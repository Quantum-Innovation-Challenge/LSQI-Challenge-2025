from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import pickle

T = np.arange(0, 1176 + 1, 24)
T_scale = np.linspace(0., 1., len(T))
DV_rescale = 10.
DOSE_rescale = 10.
BW_rescale = 100.

def prepare_dataframe(data_path, file_path):
    # Load data
    data = pd.read_csv(data_path)
    # CMT and DVID are redundant
    print((data[data["CMT"]==3]["DVID"] ==2).all())
    print((data[data["CMT"]==2]["DVID"] ==1).all())
    print((data[data["CMT"]==1]["DVID"] ==0).all())
    # MDV and EVID are redundant
    print((data[data["MDV"]==1]["EVID"] ==1).all())
    print((data[data["MDV"]==0]["EVID"] ==0).all())
    # Remove redundances
    data = data.drop('CMT', axis=1)
    data = data.drop('MDV', axis=1)

    hadm_id = []
    X = []
    c = {'dose': [], 'bw': [], 'comed': [], 'init': []}
    T_train = []
    hadm_id_test = []
    X_test = []
    c_test = {'dose': [], 'bw': [], 'comed': [], 'init': []}
    T_test = []
    for id in range(48):
        # Extract single trajectory
        single_data = data[data["ID"] == (id+1)]
        # Extract the ID of the patiant
        hadm_id_i = single_data["ID"].iloc[0]
        # Extract the BW and COMED of the patiant
        c_i_BW = single_data["BW"].iloc[0]
        c_i_COMED = single_data["COMED"].iloc[0]

        # Extract the Measurements
        data_observe = single_data[single_data["EVID"] == 0]
        # Split PK and PD
        data_observe_PK = data_observe[data_observe["DVID"] == 1]
        data_observe_PD = data_observe[data_observe["DVID"] == 2]
        # Extract the Dosing
        data_dose = single_data[single_data["EVID"] == 1]

        # Interpolation
        DV = np.interp(T, data_observe_PD["TIME"], data_observe_PD["DV"]) / DV_rescale
        c_i_dose = np.zeros_like(DV)
        for i, j in enumerate(T):
            if j in list(data_dose["TIME"]): 
                c_i_dose[i] = data_dose["AMT"].iloc[0] / DOSE_rescale
        c_i_bw = np.ones_like(DV) * c_i_BW / BW_rescale
        c_i_comed = np.ones_like(DV) * c_i_COMED
        c_i_init = np.ones_like(DV) * DV[0]

        # Append everything
        if id % 12 < 10:
            hadm_id += [hadm_id_i] * len(T)
            X.append(DV)
            T_train.append(T_scale)
            c['dose'].append(c_i_dose)
            c['bw'].append(c_i_bw)
            c['comed'].append(c_i_comed)
            c['init'].append(c_i_init)
        else:
            hadm_id_test += [hadm_id_i] * len(T)
            X_test.append(DV)
            T_test.append(T_scale)
            c_test['dose'].append(c_i_dose)
            c_test['bw'].append(c_i_bw)
            c_test['comed'].append(c_i_comed)
            c_test['init'].append(c_i_init)

    df = pd.DataFrame({
        'HADM_ID': hadm_id,
        'x': torch.tensor(np.concatenate(X), dtype=torch.float32).numpy(),
        'c_dose': torch.tensor(np.concatenate(c['dose']), dtype=torch.float32).numpy(),
        'c_bw': torch.tensor(np.concatenate(c['bw']), dtype=torch.float32).numpy(),
        'c_comed': torch.tensor(np.concatenate(c['comed']), dtype=torch.float32).numpy(),
        'c_init': torch.tensor(np.concatenate(c['init']), dtype=torch.float32).numpy(),
        't': torch.tensor(np.concatenate(T_train), dtype=torch.float32).numpy()
    })
    df_test = pd.DataFrame({
        'HADM_ID': hadm_id_test,
        'x': torch.tensor(np.concatenate(X_test), dtype=torch.float32).numpy(),
        'c_dose': torch.tensor(np.concatenate(c_test['dose']), dtype=torch.float32).numpy(),
        'c_bw': torch.tensor(np.concatenate(c_test['bw']), dtype=torch.float32).numpy(),
        'c_comed': torch.tensor(np.concatenate(c_test['comed']), dtype=torch.float32).numpy(),
        'c_init': torch.tensor(np.concatenate(c_test['init']), dtype=torch.float32).numpy(),
        't': torch.tensor(np.concatenate(T_test), dtype=torch.float32).numpy()
    })

    # # Prepare data dictionary for PyTorch Lightning
    # # for this overfitting demo, using the same data for train/val/test
    df_all = {'train': df, 'val': df_test, 'test': df_test}
    with open(file_path, 'wb') as f:
        pickle.dump(df_all, f)

class TrainingDataset(Dataset):
    def __init__(self, x0_values, x0_classes, x1_values, times_x0, times_x1):
        self.x0_values = x0_values
        self.x0_classes = x0_classes
        self.x1_values = x1_values
        self.times_x0 = times_x0
        self.times_x1 = times_x1

    def __len__(self):
        return len(self.x0_values)

    def __getitem__(self, idx):
        return (self.x0_values[idx], self.x0_classes[idx], self.x1_values[idx], self.times_x0[idx], self.times_x1[idx])


class PatientDataset(Dataset):
    def __init__(self, patient_data):
        self.patient_data = patient_data

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        return self.patient_data[idx]


class eICUDataLoader:
    def __init__(self, 
                 file_path, 
                 t_headings,
                 x_headings,
                 cond_headings,
                 memory = 0,
                 batch_size=256, 
                 groupby = 'HADM_ID',
                 train_consecutive=False):
        
        self.batch_size = batch_size
        self.file_path = file_path
        self.x_headings = x_headings
        self.cond_headings = cond_headings
        self.t_headings = t_headings
        self.input_dim = len(self.x_headings) + len(self.cond_headings)
        self.output_dim = len(self.x_headings)
        self.memory = memory
        self.min_timept = 5 + self.memory
        self.train_consecutive = train_consecutive
        self.groupby = groupby

        # Load and setup data
        self.data = pd.read_pickle(self.file_path)
        self.train_data = self.__filter_data(self.data['train'])
        self.val_data = self.__filter_data(self.data['val'])
        self.test_data = self.__filter_data(self.data['test'])

    def __filter_data(self, data_set):
        return data_set.groupby(self.groupby).filter(lambda x: len(x) > self.min_timept)

    def __unpack__(self, data_set):
        x = data_set[self.x_headings].values
        cond = data_set[self.cond_headings].values
        t = data_set[self.t_headings].values
        return x, cond, t
    
    def __sort_group__(self, data_set):
        grouped = data_set.groupby(self.groupby)
        grouped_sorted = grouped.apply(lambda x: x.sort_values([self.t_headings], ascending=True)).reset_index(drop=True)
        return grouped_sorted

    def create_pairs(self, df):
        x0_values = []
        x0_classes = []
        x1_values = []
        times_x0 = []
        times_x1 = []

        for _, group in df.groupby(self.groupby):
            sorted_group = group.sort_values(by=self.t_headings)

            for i in range(self.memory, len(sorted_group) - 1):
                x0 = sorted_group.iloc[i]
                x0_class = x0[self.cond_headings].values
                x0_value = x0[self.x_headings].values

                x1 = sorted_group.iloc[i + 1]
                x1_value = x1[self.x_headings].values

                if self.memory > 0:
                    x0_memory = sorted_group.iloc[i - self.memory:i]
                    x0_memory_flatten = x0_memory[self.x_headings].values.flatten()
                    x0_class = np.append(x0_class, x0_memory_flatten)

                x0_values.append(x0_value)
                x0_classes.append(x0_class)
                x1_values.append(x1_value)
                times_x0.append(x0[self.t_headings])
                times_x1.append(x1[self.t_headings])

        x0_values = np.array(x0_values).squeeze().astype(np.float32)
        x0_classes = np.array(x0_classes).squeeze().astype(np.float32)
        x1_values = np.array(x1_values).squeeze().astype(np.float32)
        times_x0 = np.array(times_x0).squeeze().astype(np.float32)
        times_x1 = np.array(times_x1).squeeze().astype(np.float32)

        # if ==0, shape of (bs, 0) will be kept; 
        # if >=2, shape of (bs, cond_num+dim*memory) will be kept
        if len(self.cond_headings) + self.memory == 1: 
            x0_classes = np.expand_dims(x0_classes, axis=1)
        
        if len(self.x_headings) < 2:
            x0_values = np.expand_dims(x0_values, axis=1)
            x1_values = np.expand_dims(x1_values, axis=1)

        return x0_values, x0_classes, x1_values, times_x0, times_x1

    def create_patient_data(self, df):
        patient_lst = []
        for _, group in df.groupby(self.groupby):
            sorted_group = group.sort_values(by=self.t_headings)
            x0_values, x0_classes, x1_values, times_x0, times_x1 = self.create_pairs(sorted_group)

            patient_lst.append((x0_values,
                                x0_classes,
                                x1_values,
                                times_x0,
                                times_x1))
        return patient_lst

    def create_patient_data_t0(self, df):
        patient_lst = []
        for _, group in df.groupby(self.groupby):
            sorted_group = group.sort_values(by=self.t_headings)
            x0_values, x0_classes, x1_values, times_x0, times_x1 = self.create_pairs(sorted_group)

            if len(self.cond_headings) < 2:
                x0_classes = np.expand_dims(x0_classes, axis=1)
            else:
                x0_classes = x0_classes.squeeze().astype(np.float32)

            x0_values = np.repeat(x0_values[0][None, :], len(x0_values), axis=0)
            x0_classes = np.repeat(x0_classes[0][None, :], len(x0_values), axis=0)
            times_x0 = np.repeat(times_x0[0], len(x0_values))

            patient_lst.append((x0_values.squeeze().astype(np.float32),
                                x0_classes,
                                x1_values.squeeze().astype(np.float32),
                                times_x0.squeeze().astype(np.float32),
                                times_x1.squeeze().astype(np.float32)))
        return patient_lst

    def get_dataloader(self, data, shuffle=True, for_training=True):
        if for_training and self.train_consecutive:
            data = self.create_patient_data_t0(data)
            dataset = PatientDataset(data)
            return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
        elif for_training:
            data = self.create_pairs(data)
            dataset = TrainingDataset(*data)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)
        else:
            data = self.create_patient_data(data)
            dataset = PatientDataset(data)
            return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)

    def get_train_loader(self, shuffle=True):
        return self.get_dataloader(self.train_data, shuffle=shuffle, for_training=True)

    def get_val_loader(self):
        return self.get_dataloader(self.val_data, shuffle=False, for_training=False)

    def get_test_loader(self):
        return self.get_dataloader(self.test_data, shuffle=False, for_training=False)