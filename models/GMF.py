import torch
import torch.nn as nn
import torch.nn.functional as F

class GMF(nn.Module):
    def __init__(self, num_user, num_item, num_factor, model):
        super(GMF, self).__init__()
        
        self.num_user = num_user
        self.num_item = num_item
        self.num_factor = num_factor
        self.model = model
        
        self.user_embedding_GMF = nn.Embedding(num_user, num_factor)
        self.item_embedding_GMF = nn.Embedding(num_item, num_factor)
        
    def _init_weight_(self):
        if not self.model == "GMF-pre":
            nn.init.normal_(self.user_embedding_GMF.weight, std=0.01)
            nn.init.normal_(self.item_embedding_GMF.weight, std=0.01)
        
        else:
            self.user_embedding_GMF.weight.data.copy_(self.model.user_embedding_GMF.weight)
            self.item_embedding_GMF.weight.data.copy_(self.model.item_embedding_MLP.weight)
    
    def forward(self, user, item):
        user_embedding_GMF = self.user_embedding_GMF(user)
        item_embedding_GMF = self.item_embedding_GMF(item)
        
        GMF_layer = torch.mul(user_embedding_GMF, item_embedding_GMF)
        
        return GMF_layer
        