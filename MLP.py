import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import config
import evaluate
import data_utils

class MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model):
        super(MLP, self).__init__()
        
        self.dropout = dropout
        self.model = model
        
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))
        
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=self.dropout))
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = factor_num
        
        self.predict_layer = nn.Sequential(nn.Linear(predict_size, 1),
                                           nn.Sigmoid())
        self._init_weight_()
        
    def _init_weight_(self):

        nn.init.normal_(self.embed_user_MLP.weight, mean=0, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, mean=0, std=0.01)
            
        for m in self.MLP_layers:
            if np.isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            
        nn.init.normal_(self.predict_layer.weight, mean=0, std=0.01)
            
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
            
    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.view(-1)
    
if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="training epoches")
    parser.add_argument("--top_K", type=int, default=10,
                        help="compute mtrics@top_K")
    parser.add_argument("--factor_num", type=int, default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of layers in MLP model")
    parser.add_argument("--num_ng", type=int, default=3,
                        help="sample negative items for training")
    parser.add_argument("--test_num_ng", type=int, default=99,
                        help="sample part of negative items for testing")
    parser.add_argument("--out", default=True,
                        help="save model or not")
    parser.add_argument("--gpu", type=str, default="0",
                        help="gpu card ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # Dataset
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num+1, shuffle=False, num_workers=0)

    model = MLP(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    count, best_hr = 0, 0

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()
        
        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            
            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            count += 1
            
        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        end_time = time.time()
        
        print("The time elapse of epoch {:03d}".format(epoch) + " is : " +
              time.strftime("%H: %ML %S", time.gmtime(end_time-start_time)))
        print(f"HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}")
        
        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(model, "{}{}.pth".format(config.model_path, config.model))

print(f"End. Best epoch {best_epoch:03d}: HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}")