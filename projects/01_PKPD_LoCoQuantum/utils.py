import numpy as np
import torch
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt
from models import *
from data import T, T_scale

def train_model(model, noise_prediction, train_loader, val_loader=None, num_epochs=10, device='cuda', eval_every=500, model_save_path="./model_save_best.pt"):
    model.to(device)
    test_loss_record = 100000.

    # with noise prediction
    if noise_prediction: 
        flow_optimizer, noise_optimizer = model.configure_optimizers()
    # without noise prediction
    else: 
        optimizer = model.configure_optimizers()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x0, x0_class, x1, x0_time, x1_time = [b.to(device) for b in batch]

            x, ut, t_model, futuretime, t = model.__x_processing__(x0, x1, x0_time, x1_time)

            in_tensor = torch.cat([x, x0_class, t_model], dim=-1)

            # with noise prediction
            if noise_prediction: 
                xt = model.flow_model.forward_train(in_tensor)

                if model.implementation == "SDE":
                    sde_noise = model.noise_model.forward_train(in_tensor)
                    variance = torch.sqrt(t * (1 - t)) * sde_noise
                    noise = torch.randn_like(xt[:, :model.dim]) * variance
                    loss = model.loss_fn(xt[:, :model.dim] + noise.clone().detach(), x1) + model.loss_fn(xt[:, -1], futuretime)
                    uncertainty = (xt[:, :model.dim].clone().detach() + noise)
                    noise_loss = model.loss_fn(uncertainty, x1)
                else:
                    loss = model.loss_fn(xt[:, :model.dim], x1) + model.loss_fn(xt[:, -1], futuretime)
                    uncertainty = torch.abs(xt[:, :model.dim].clone().detach() - x1)
                    noise_loss = model.loss_fn(model.noise_model.forward_train(in_tensor)[:, :model.dim], uncertainty)

                flow_optimizer.zero_grad()
                loss.backward()
                flow_optimizer.step()

                noise_optimizer.zero_grad()
                noise_loss.backward()
                noise_optimizer.step()

                train_loss += (loss + noise_loss).item()

            # without noise prediction
            else: 
                xt = model.model.forward_train(in_tensor)

                if model.implementation == "SDE":
                    variance = t * (1 - t) * model.sde_noise
                    noise = torch.randn_like(xt[:, :model.dim]) * torch.sqrt(variance)
                    loss = model.loss_fn(xt[:, :model.dim] + noise, x1) + model.loss_fn(xt[:, -1], futuretime)
                else:
                    loss = model.loss_fn(xt[:, :model.dim], x1) + model.loss_fn(xt[:, -1], futuretime)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        # validation
        if (epoch+1) % eval_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}")
            if val_loader: 
                test_loss = test_model(model, noise_prediction, val_loader, device)
                if test_loss < test_loss_record:
                    if model_save_path is not None:
                        torch.save(model.state_dict(), model_save_path)
                        test_loss_record = test_loss
                    
def test_model(model, noise_prediction, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    list_of_pairs = []
    list_of_times = []

    dict_full_trajs = {}
    dict_pred_trajs = {}

    loss_sum = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = [x.to(device) for x in batch]
            loss, pairs, metricD, noise_loss, noise_pair, full_time = test_func_step(batch, batch_idx, model, noise_prediction)
            list_of_pairs.append(pairs)
            list_of_times.append(full_time)
            full_traj = pairs[0][0]
            pred_traj = pairs[0][1]
            dict_full_trajs[batch_idx] = full_traj.detach().cpu().numpy()
            dict_pred_trajs[batch_idx] = pred_traj.detach().cpu().numpy()
            loss_sum += loss
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'lime']
        count = 0
        for _,v in dict_pred_trajs.items():
            ax.plot(T_scale[model.memory:], v, label="predicted" if count == 0 else '', color=colors[count])
            count += 1
        count = 0
        for _,v in dict_full_trajs.items():
            ax.plot(T_scale[model.memory:], v, label="ground truth" if count == 0 else '', color=colors[count], linestyle='--')
            count += 1

        ax.set_xlabel('scaled t', fontsize='10')
        ax.set_ylabel('x', fontsize='10')
        ax.legend(fontsize='10')
        plt.tight_layout()
        plt.show()
        plt.close()
    print(f"validation loss: {loss_sum/len(test_loader)}")
    return loss_sum/len(test_loader)

def test_func_step(batch, batch_idx, model, noise_prediction):
    """Assuming each batch is one patient"""
    total_loss = []
    traj_pairs = []

    total_noise_loss = []
    noise_pairs = []

    x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
    times_x0 = times_x0.squeeze()
    times_x1 = times_x1.squeeze()

    full_traj = torch.cat([x0_values[0, 0, :model.dim].unsqueeze(0), x1_values[0, :, :model.dim]], dim=0)
    full_time = torch.cat([times_x0[0].unsqueeze(0), times_x1], dim=0)

    if model.implementation == "ODE":
        ind_loss, pred_traj, noise_mse, noise_pred = test_trajectory_ode(batch, model, noise_prediction)
    elif model.implementation == "SDE":
        ind_loss, pred_traj, noise_mse, noise_pred = test_trajectory_sde(batch, model, noise_prediction)

    total_loss.append(ind_loss)
    traj_pairs.append([full_traj, pred_traj])
    noise_pairs.append([full_traj, noise_pred])
    total_noise_loss.append(noise_mse)

    # Optionally detach and move tensors to CPU for further processing or visualization
    full_traj = full_traj.detach().cpu().numpy()
    pred_traj = pred_traj.detach().cpu().numpy()
    full_time = full_time.detach().cpu().numpy()
    # Calculate metrics
    metricD = metrics_calculation(pred_traj, full_traj)
    return np.mean(total_loss), traj_pairs, metricD, np.mean(total_noise_loss), noise_pairs, full_time
    
def test_trajectory_ode(batch, model, noise_prediction): # have to squeeze here to adjust for bs=1
    # with noise prediction
    if noise_prediction:
        node = NeuralODE(torch_wrapper_tv(model.flow_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        node_noise = NeuralODE(torch_wrapper_tv(model.noise_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    else:
        node = NeuralODE(torch_wrapper_tv(model.model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    total_pred, noise_pred = [], []
    mse, noise_mse = [], []
    total_pred = []
    mse = []

    x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
    x0_values = x0_values.squeeze(0)
    x0_classes = x0_classes.squeeze(0)
    x1_values = x1_values.squeeze(0)
    times_x0 = times_x0.squeeze()
    times_x1 = times_x1.squeeze()

    total_pred.append(x0_values[0].unsqueeze(0))
    len_path = x0_values.shape[0]
    if model.memory > 0:
        time_history = x0_classes[0][-(model.memory * model.dim):]

    for i in range(len_path):
#         time_span = torch.linspace(times_x0[i], times_x1[i], 100).to(x0_values.device)
        time_span = torch.linspace(times_x0[i], times_x1[i], 10).to(x0_values.device)

        if model.memory > 0:
#             print(f"cond memory>0 part 0 {x0_classes[i].unsqueeze(0)}")
#             print(f"cond memory>0 part 1 shape {x0_classes[i][:-(model.memory * model.dim)].unsqueeze(0).shape}")
#             print(f"cond memory>0 part 1 {x0_classes[i][:-(model.memory * model.dim)].unsqueeze(0)}")
#             print(f"cond memory>0 part 2 shape {time_history.unsqueeze(0).shape}")
#             print(f"cond memory>0 part 2 {time_history.unsqueeze(0)}")
            new_x_classes = torch.cat([x0_classes[i][:-(model.memory * model.dim)].unsqueeze(0), time_history.unsqueeze(0)], dim=1)
        else:
            new_x_classes = x0_classes[i].unsqueeze(0)
            print(f"cond memory=0 {new_x_classes.shape}")

        with torch.no_grad():
            if i == 0: # TODO: here is the issue to fix. Seems to be wrong for the paper figure generation
                testpt = torch.cat([x0_values[i].unsqueeze(0), new_x_classes], dim=1)
#                 testpt = torch.cat([x0_values[i].unsqueeze(0), x0_classes[i].unsqueeze(0)], dim=1)
            else:
                testpt = torch.cat([pred_traj, new_x_classes], dim=1)
#                 testpt = torch.cat([pred_traj, x0_classes[i].unsqueeze(0)], dim=1)
                
            traj = node.trajectory(testpt, t_span=time_span)
            if noise_prediction:
                noise_traj = node_noise.trajectory(testpt, t_span=time_span)

        pred_traj = traj[-1, :, :model.dim]
        total_pred.append(pred_traj)
        if noise_prediction:
            noise_traj = noise_traj[-1, :, :model.dim]
            noise_pred.append(noise_traj)
        

        ground_truth_coords = x1_values[i]
        mse_traj = model.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
        mse.append(mse_traj)
        if noise_prediction:
            uncertainty_traj = ground_truth_coords - pred_traj
            noise_mse_traj = model.loss_fn(noise_traj, uncertainty_traj).detach().cpu().numpy()
            noise_mse.append(noise_mse_traj)

        
        if model.memory > 0:
            flattened_coords = pred_traj.flatten()
            time_history = torch.cat([time_history[model.dim:].unsqueeze(0), flattened_coords.unsqueeze(0)], dim=1).squeeze()

    mse_all = np.mean(mse)
    total_pred_tensor = torch.stack(total_pred).squeeze(1)
    if noise_prediction:
        noise_pred = torch.stack(noise_pred).squeeze(1)
    
    return mse_all, total_pred_tensor, noise_mse, noise_pred

def test_trajectory_sde(batch, model, noise_prediction):
    if noise_prediction:
        sde = SDE_func_solver(model.flow_model, noise=model.noise_model)
    else:
        sde = SDE(model.model, noise=0.1)
    total_pred, noise_pred = [], []
    mse, noise_mse = [], []

    x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
    x0_values = x0_values.squeeze(0)
    x0_classes = x0_classes.squeeze(0)
    x1_values = x1_values.squeeze(0)
    times_x0 = times_x0.squeeze()
    times_x1 = times_x1.squeeze()

    total_pred.append(x0_values[0].unsqueeze(0))
    len_path = x0_values.shape[0]
    assert len_path == x1_values.shape[0]

    if model.memory > 0:
        time_history = x0_classes[0][-(model.memory * model.dim):]

    for i in range(len_path):
        if model.time_varying:
            time_span = torch.linspace(times_x0[i], times_x1[i], 10).to(x0_values.device)
        else:
            time_span = torch.linspace(0., 1./len(T), 10).to(x0_values.device)

        if model.memory > 0:
            new_x_classes = torch.cat([x0_classes[i][:-(model.memory * model.dim)].unsqueeze(0), time_history.unsqueeze(0)], dim=1)
        else:
            new_x_classes = x0_classes[i].unsqueeze(0)

        with torch.no_grad():
            if i == 0:
                testpt = torch.cat([x0_values[i].unsqueeze(0), new_x_classes], dim=1)
            else:
                testpt = torch.cat([pred_traj, new_x_classes], dim=1)
                
            traj, noise_traj = sde_solver(sde, testpt, time_span)
        pred_traj = traj[-1, :, :model.dim]
        noise_traj = noise_traj[-1, :, :model.dim]

        total_pred.append(pred_traj)
        if noise_prediction:
            noise_pred.append(noise_traj)

        ground_truth_coords = x1_values[i]
        mse_traj = model.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
        mse.append(mse_traj)
        noise_mse.append(mse_traj)

        if model.memory > 0:
            flattened_coords = pred_traj.flatten()
            time_history = torch.cat([time_history[model.dim:].unsqueeze(0), flattened_coords.unsqueeze(0)], dim=1).squeeze()

    mse_all = np.mean(mse)
    noise_mse_all = np.mean(noise_mse)
    total_pred_tensor = torch.stack(total_pred).squeeze(1)
    if noise_prediction:
        noise_pred = torch.stack(noise_pred).squeeze(1)
    return mse_all, total_pred_tensor, noise_mse_all, noise_pred

def sde_solver(sde, initial_state, time_span):
    dt = time_span[1] - time_span[0]
    current_state = initial_state
    trajectory = [current_state]
    noise_trajectory = []

    for t in time_span[1:]:
        drift = sde.f(t, current_state)
        diffusion = sde.g(t, current_state)
        noise = torch.randn_like(current_state) * torch.sqrt(dt)
        current_state = current_state + drift * dt + diffusion * noise
        trajectory.append(current_state)
        noise_trajectory.append(diffusion * noise)

    return torch.stack(trajectory), torch.stack(noise_trajectory)
@torch.no_grad()
def simulate(model, sde, x0_classes, x0_values, time_history, len_path):
    total_pred = []
    total_pred.append(x0_values[0].unsqueeze(0))
    for i in range(len_path):
        time_span = torch.linspace(0., 1./len_path, 10).to(x0_values.device)
        if model.memory > 0:
            new_x_classes = torch.cat([x0_classes[i].unsqueeze(0), time_history.unsqueeze(0)], dim=1)
        else:
            new_x_classes = x0_classes[i].unsqueeze(0)
        with torch.no_grad():
            if i == 0:
                testpt = torch.cat([x0_values[i].unsqueeze(0), new_x_classes], dim=1)
            else:
                testpt = torch.cat([pred_traj, new_x_classes], dim=1)
        traj, noise_traj = sde_solver(sde, testpt, time_span)
        pred_traj = traj[-1, :, :model.dim]
        noise_traj = noise_traj[-1, :, :model.dim]
        if model.memory > 0:
            flattened_coords = pred_traj.flatten()
            time_history = torch.cat([time_history[model.dim:].unsqueeze(0), flattened_coords.unsqueeze(0)], dim=1).squeeze()
        total_pred.append(pred_traj)
    return total_pred