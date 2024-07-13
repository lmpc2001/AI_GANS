import torch
from torch.utils.data import Dataset

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # data = self.df.iloc[index].values.astype(np.float32)
        # return torch.tensor(data.values)
        return torch.tensor(self.dataframe.iloc[index].values)
