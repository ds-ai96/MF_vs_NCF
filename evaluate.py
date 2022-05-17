import numpy as np
import torch

def HR(gt_item, pred_item_list):
    if gt_item in pred_item_list:
        return 1
    return 0

def NDCG(gt_item, pred_item_list):
    if gt_item in pred_item_list:
        item_index = pred_item_list.index(gt_item)
        return np.reciprocal(np.log2(item_index + 2))
    return 0

def metrics(model, test_loader, top_k):
    hr, ndcg = [], []
    
    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()
        
        predictions = model(user, item)
        _, indices =torch.topk(predictions, top_k)
        pred_item_list = torch.tack(item, indices).cpu().numpy().tolist()
        
        gt_item = item[0].item()
        hr.append(HR(gt_item, pred_item_list))
        ndcg.append(NDCG(gt_item, pred_item_list))
    return np.mean(hr), np.mean(ndcg)
        