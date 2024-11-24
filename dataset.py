import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# class for the dataset
class PatientsDataset(Dataset):
    def __init__(self, file_paths, metadata_file, is_train=True):
        self.data = []
        self.labels = []
        self.metadata = pd.read_csv(metadata_file)

        # store id and metadata in metadata_dict
        self.metadata_dict = self.metadata.set_index('participant_id').to_dict(orient='index')
        self.is_train = is_train

        print(f"Loaded metadata with {len(self.metadata_dict)} participants.")

        for file in file_paths:
            participant_id = None
            for id in self.metadata_dict.keys():
                if id in file:
                    participant_id = id
                    break

            if participant_id is not None:
                participant_metadata = self.metadata_dict[participant_id]

                #  read .tsv info about the current participant
                try:
                    df = pd.read_csv(file, sep="\t", header=None)

                    # vectorize upper diagonal of the matrix
                    features = df.where(~np.tril(np.ones(df.shape)).astype(bool)).stack().values
                    self.data.append(features)

                    # TODO: Concatenate Training and Test dataframes with respective metadata after the metadata is
                    # TODO: cleared

                    if self.is_train:
                        self.labels.append(participant_metadata['age'])
                    else:
                        self.labels.append(None)

                except Exception as e:
                    print(f"Error loading file {file}: {e}")

        # normalize the data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

        if self.is_train:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]
