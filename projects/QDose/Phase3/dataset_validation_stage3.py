import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error

# Import your physics engine
from pkpd_physics_stage3 import PKPDSimulator

class ClinicalDatasetLoader:
    """
    Parses NONMEM/RTF clinical datasets for validation.
    Expected Columns: ID, BW, COMED, TIME, DV, AMT, EVID, DVID
    """
    def __init__(self, filepath):
        # Flexible separator parsing (comma or tab)
        try:
            self.df = pd.read_csv(filepath, sep=r'[,\t]', engine='python')
        except:
            self.df = pd.read_csv(filepath)
        
        # Clean column names
        self.df.columns = [c.strip().upper() for c in self.df.columns]

    def get_patient_ids(self):
        return self.df['ID'].unique()

    def get_patient_covariates(self):
        """Extracts unique X (BW, COMED) for every patient ID."""
        patients = self.df.groupby('ID').first().reset_index()
        # Ensure columns exist, default to 0 if missing
        if 'COMED' not in patients.columns: patients['COMED'] = 0
        
        X_val = patients[['BW', 'COMED']].values
        ids = patients['ID'].values
        return ids, X_val

    def get_patient_events(self, patient_id):
        return self.df[self.df['ID'] == patient_id].sort_values('TIME')

def simulate_clinical_schedule(params, events_df):
    """Simulates the EXACT dosing/observation schedule found in the clinical dataset."""
    CL, Vc, Vp, Q, kin, kout, IC50 = params
    
    # ODE Constants
    k10, k12, k21 = CL/Vc, Q/Vc, Q/Vp
    
    def model_odes(y, t):
        Ac, Ap, R = y
        Cp = Ac / Vc
        # Inhibition PD Model
        inhibition = Cp / (IC50 + Cp)
        dAc_dt = -k10*Ac - k12*Ac + k21*Ap
        dAp_dt =  k12*Ac - k21*Ap
        dR_dt  = kin * (1 - inhibition) - kout*R
        return [dAc_dt, dAp_dt, dR_dt]

    # Initial State (Steady State Baseline)
    R0 = kin / kout
    y_current = [0.0, 0.0, R0]
    t_current = 0.0
    
    predictions = []
    
    # High-resolution time points for smooth plotting
    # We will record both the exact event times (for residuals) AND intermediate times (for plotting)
    
    for _, row in events_df.iterrows():
        t_next = row['TIME']
        amt = row['AMT']
        evid = row['EVID'] 
        
        # Simulate gap between events
        if t_next > t_current:
            # Create smooth steps between events
            t_smooth = np.linspace(t_current, t_next, num=max(2, int(t_next - t_current)*2))
            sol = odeint(model_odes, y_current, t_smooth)
            
            # Store smooth trajectory for plotting
            for t_val, y_val in zip(t_smooth[:-1], sol[:-1]):
                predictions.append({
                    'TIME': t_val, 'CP': y_val[0]/Vc, 'RESP': y_val[2], 'TYPE': 'SIM'
                })
            
            y_current = sol[-1]
            t_current = t_next
            
        # Handle Event
        if evid == 1: # Dose
            y_current[0] += amt # Assuming AMT units match Model units
            
        elif evid == 0: # Observation
            # Record Exact Prediction at this Time Point for Error Calc
            cp_pred = y_current[0] / Vc
            resp_pred = y_current[2]
            
            # Use DVID to determine what was observed
            dvid = row.get('DVID', 1)
            obs_val = row['DV']
            
            predictions.append({
                'TIME': t_current,
                'CP': cp_pred,
                'RESP': resp_pred,
                'DV': obs_val,
                'DVID': dvid,
                'TYPE': 'OBS'
            })
            
    return pd.DataFrame(predictions)

def plot_individual_fit(patient_id, sim_df, output_dir):
    """Generates a dual-axis plot: PK (Top) and PD (Bottom)."""
    
    # Filter Data
    sim_line = sim_df[sim_df['TYPE'] == 'SIM']
    obs_points = sim_df[sim_df['TYPE'] == 'OBS']
    
    obs_pk = obs_points[obs_points['DVID'] == 1]
    obs_pd = obs_points[obs_points['DVID'] == 2]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot PK (Concentration)
    ax1.plot(sim_line['TIME'], sim_line['CP'], 'b-', alpha=0.6, label='QNN Prediction')
    ax1.scatter(obs_pk['TIME'], obs_pk['DV'], color='black', s=40, label='Observed Data')
    ax1.set_ylabel('Concentration (ng/mL)')
    ax1.set_title(f'Patient {int(patient_id)}: Pharmacokinetics')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot PD (Biomarker)
    ax2.plot(sim_line['TIME'], sim_line['RESP'], 'g-', alpha=0.6, label='QNN Prediction')
    ax2.scatter(obs_pd['TIME'], obs_pd['DV'], color='red', s=40, label='Observed Data')
    
    ax2.axhline(y=3.3, color='r', linestyle='--', alpha=0.5, label='Threshold (3.3)')
    ax2.set_ylabel('Biomarker Response')
    ax2.set_xlabel('Time (Hours)')
    ax2.set_title(f'Patient {int(patient_id)}: Pharmacodynamics')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save Plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/Patient_{int(patient_id)}_Fit.png")
    plt.close()

def run_dataset_validation(dataset_path, qnn_model, max_plots=5):
    print(f"\n--- Loading Validation Dataset: {dataset_path} ---")
    loader = ClinicalDatasetLoader(dataset_path)
    ids, X_val = loader.get_patient_covariates()
    
    print(f"   Predicting parameters for {len(ids)} patients...")
    Y_pred = qnn_model.predict(X_val)
    
    results = []
    
    print(f"   Simulating and Plotting (Top {max_plots} patients)...")
    
    for i, pid in enumerate(ids):
        # 1. Simulate
        events = loader.get_patient_events(pid)
        params = Y_pred[i]
        sim_res = simulate_clinical_schedule(params, events)
        
        # 2. Calculate Error
        obs_rows = sim_res[sim_res['TYPE'] == 'OBS']
        pk_rows = obs_rows[obs_rows['DVID'] == 1]
        pd_rows = obs_rows[obs_rows['DVID'] == 2]
        
        # Safe Error Calculation (Handle missing data)
        rmse_pk = np.sqrt(mean_squared_error(pk_rows['DV'], pk_rows['CP'])) if not pk_rows.empty else None
        rmse_pd = np.sqrt(mean_squared_error(pd_rows['DV'], pd_rows['RESP'])) if not pd_rows.empty else None
        
        # Store Result
        results.append({
            'ID': pid,
            'RMSE_PK': rmse_pk,
            'RMSE_PD': rmse_pd,
            'Weight': X_val[i][0],
            'Meds': X_val[i][1]
        })

        # 3. Plot
        if i < max_plots:
            plot_individual_fit(pid, sim_res, output_dir="validation_plots")
            
    # Calculate Averages (ignoring Nones)
    all_pk = [r['RMSE_PK'] for r in results if r['RMSE_PK'] is not None]
    all_pd = [r['RMSE_PD'] for r in results if r['RMSE_PD'] is not None]
    
    print(f"\n   Plots saved to 'validation_plots/' directory.")
    print(f"   Mean RMSE (PK): {np.mean(all_pk):.4f}")
    print(f"   Mean RMSE (PD): {np.mean(all_pd):.4f}")
    
    return results