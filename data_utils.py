from pyexpat import features
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data

import config

def load_all(test_num=100):
    train_data = pd.read_csv(
        config.train_rating, sep="\t", header=None,
        names=["user", "item"], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    
    user_num = train_data["user"].max() + 1 # user index가 0부터 시작
    item_num = train_data["item"].max() + 1 # item index가 0부터 시작
    
    train_data = train_data.values.tolist()
    
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[[x[0], x[1]]] = 1.0
    
    test_data = []
    with open(config.test_negative, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u = eval(arr[0])[0]
            i = eval(arr[0])[1]
            test_data.append([u, i])
            
            for j in arr[1:]:
                test_data.append([u, int(j)])
            
            line = f.readline()
    return train_data, test_data, user_num, item_num, train_mat

class load_dataset(data.Dataset):
    def __init__(self, feautres, num_item, train_mat=None, num_ng=0, is_training=None):
        super(load_dataset, self).__init__()
        
        self.feature_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.label = [0 for _ in range(features)]
        
    def ng_sample(self):
        assert self.is_training, "No need to sampling when test phase"
        
        self.feature_ng = []
        
        for x in self.feature_ps:
            u = x[0]
            for t  in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
        
        labels_ps = [1 for _ in range(self.feature_ps)]
        labels_ng = [0 for _ in range(self.feature_ng)]
        
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng
        
    def __len__(self):
        return (self.num_ng + 1) * len(self.labes)
    
    def __getitem__(self, idx):
        features = self.festures_fill if self.is_training else self.feature_ps
        labels = self.labels_fill if self.is_training else self.labels_fill
        
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        
        return user, item, label