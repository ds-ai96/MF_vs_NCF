import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_user, num_item, num_factor, num_layers, dropout, model):
        super(MLP, self).__init__()
        
        self.num_user = num_user
        self.num_item = num_item
        self.num_factor = num_factor
        self.dropout = dropout
        self.model = model
        
        self.user_embedding_MLP = nn.Embedding(num_user, num_factor * (2 ** (num_layers - 1)))
        self.item_embedding_MLP = nn.Embedding(num_item, num_factor * (2 ** (num_layers - 1)))
        
        MLP_modules = []
        
        for i in range(num_layers):
            input_size = num_factor * (2 ** (num_layers - i))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=self.dropout))
            
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        self._init_weight()
        
    def _init_weight_(self):
        if not self.model == "MLP-pre":
            nn.init.normal_(self.user_embedding_MLP.weight, std=0.01)
            nn.init.normal_(self.item_embedding_MLP.weight, std=0.01)
            
            for layer in self.MLP_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.normal_(layer.bias, std=0.01)
                    
        else:
            self.user_embedding_MLP.weight.data.copy_(self.model.user_embedding_MLP.weight)
            self.item_embedding_MLP.weight.data.copy_(self.model.item_embedding_MLP.weight)
            
            for (layer1, layer2) in zip(self.MLP_layers, self.model.MLP_layers):
                if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                    layer1.weight.data.copy_(layer2.weight)
                    layer1.bias.data.copy_(layer2.bias)
        
    def forward(self, user, item):
        user_embedding_MLP = self.user_embedding_MLP(user)
        item_embedding_MLP = self.item_embedding_MLP(item)
        
        embedding_MLP = torch.cat((user_embedding_MLP, item_embedding_MLP), -1)
        MLP_layer = self.MLP_layers(embedding_MLP)
        
        return MLP_layer.view(-1)
        