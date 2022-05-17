import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers):
        super(NCF,self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.num_layers = num_layers
        
        self.user_embedding_GMF = nn.Embedding(user_num, factor_num)
        self.item_embedding_GMF = nn.Embedding(item_num, factor_num)
        
        self.user_embedding_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.item_embedding_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))
