import torch

def BPR_loss(user_embedding, positive_item_embedding, negative_item_embedding):
    
    s_pos = torch.sum(user_embedding * positive_item_embedding, dim=1)
    s_neg = torch.sum(user_embedding * negative_item_embedding, dim=1)
    
    mid_sigmoid_value = torch.sigmoid(s_pos - s_neg)
    loss_value = (-1) * torch.log(mid_sigmoid_value)

    return loss_value.mean()
