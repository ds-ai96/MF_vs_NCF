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

class MF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, reg):
        super(MF, self).__init__()
        self.embed_user_MF = np.random.normal(0, 0.01, (user_num, factor_num))
        self.embed_utem_MF = np.random.normal(0, 0.01, (item_num, factor_num))
        self.user_bias = np.zeros([user_num])
        self.item_bias = np.zeros([item_num])
        self.bias =0.0
        self.reg = reg        
                
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        
        concat = torch.cat((output_GMF, output_MLP), -1)
        
        prediction = self.predict_layer(concat)
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

    # Model
    if config.model == "NeuMF-pre":
        assert os.path.exists(config.GMF_model_path), "GMF model이 존재하지 않습니다."
        assert os.path.exists(config.MLP_model_path), "MLP model이 존재하지 않습니다."
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
        
    else:
        GMF_model = None
        MLP_model = None

    model = NCF(user_num, item_num, args.factor_num, args.num_layers,
                args.dropout, config.model, GMF_model, MLP_model)
    
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