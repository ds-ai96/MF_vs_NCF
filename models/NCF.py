import torch
import torch.nn as nn
import torch.nn.functional as F
import GMF, MLP

class NCF(nn.Module):
    def __init__(self, num_user, num_item, num_factor, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        
        self.num_user = num_user
        self.num_item = num_item
        self.num_factor = num_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = model
        
        if GMF_model == None:
            self.GMF_model = GMF(num_user, num_item, num_factor, model)
        else:
            self.GMF_model = GMF_model
        
        if MLP_model == None:
            self.MLP_model = MLP(num_user, num_item, num_factor, num_layers, dropout, model)
        else:
            self.MLP_model = MLP_model
        
        predict_size = num_factor * 2 
        self.predict_layer = nn.Linear(predict_size, 1)
        
    def forward(self, user, item):
        GMF_layer = GMF.forward(user, item)
        MLP_layer = MLP.forward(user, item)
        
        NeuMF = torch.cat((GMF_layer, MLP_layer), -1)
        
        prediction = self.predict_layer(NeuMF)
        prediction = nn.Sigmoid(prediction)
        
        return prediction.view(-1)