### Main module to preprocess dataset

import torch
import random
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import CHARAS_LIST
from torch.utils.data import Dataset, DataLoader

class StockData(Dataset):
    def __init__(self, return_file, data_file, mask_prob = 0, split = "train"):
        super().__init__()

        assert mask_prob <= 1.0, "mask probability should be less than 1.0"
        self.split = split
        self.scaler = StandardScaler()
        self.return_file = return_file
        self.data_file = data_file


        
        self.data, self.ret = self.load_data()
        self.merged_data = self._merge()
        self.mask_prob = mask_prob
        self._create_data()
        
        

    def _masker(self, x):
        
        """
            x : unmasked dataset containing 94 characteristics
            returns : masked(x), target(x), attention_mask(x)
        """ 
        attention_mask = torch.where(x == torch.tensor(123456789), -1e16, 0)
        return x, attention_mask
    
    def _target_masker(self, y):
        attention_mask = torch.zeros_like(torch.from_numpy(y))
        sample_size = int(self.mask_prob * attention_mask.shape[0])
        sample_ = torch.randint(low = 0, high = attention_mask.shape[0] + 1, size = (sample_size))
        for index in sample_:
            attention_mask[index] = -torch.tensor(-1e16)
        
        return y, attention_mask



    def load_data(self):
        with open(self.data_file, "rb") as fp:
            data = pkl.load(fp)
            if self.split == "train":
                data = data[data.DATE <= 19701231] ### Compressing data
            elif self.split == "validation":
                data = data[ 19861231 <= data.DATE <= 20001231]
            elif self.split == "test":
                data = data[data.DATE > 20001231]
            fp.close()

        with open(self.return_file, "rb") as fp:
            ret = pkl.load(fp)
            if self.split  == "train":
                ret = ret[ret.date <= 19701231] ### Compressing data
            elif self.split == "validation":
                ret = ret[ 19861231 <= ret.date <= 20001231]
            elif self.split == "test":
                ret = ret[ret.date > 20001231]
            fp.close()

        return data, ret
    
    def _merge(self):

        self.scaler.set_output(transform='pandas')
        self.data = self.data.rename(columns = {"DATE" : "date"})
        cols = list(self.data.columns)
        cols = [e for e in cols if e not in ['permno', 'date', 'sic2']]
        self.data[cols] = self.scaler.fit_transform(self.data[cols])

        merged_data =  pd.merge(self.data, self.ret, how = "inner", on=['permno', 'date'])
        merged_data.drop(columns=["permno", "month", "date", "sic2"], axis = 1, inplace= True)
        merged_data.fillna(123456789, inplace=True)
        return merged_data
    
    def _create_data(self):
        self.X = self.merged_data[[l for l in self.merged_data.columns if l != "ret-rf"]].values
        self.y = self.merged_data["ret-rf"].values


    def __len__(self):
        return self.merged_data.shape[0]

    def __getitem__(self, index):
        return self._masker(self.X[index]), self.y[index]


    