import torch
import torch.nn.functional as F

def balance_bce_loss(pred, gt, mask):
    negative_ratio = 3.0
    eps = 1e-6
    positive = (gt * mask)
    negative = ((1 - gt) * mask)
    positive_count = int(positive.float().sum())
    negative_count = min(
        int(negative.float().sum()),
        int(positive_count * negative_ratio))

    assert gt.max() <= 1 and gt.min() >= 0
    assert pred.max() <= 1 and pred.min() >= 0
    loss = F.binary_cross_entropy(pred, gt, reduction='none')
    positive_loss = loss * positive.float()
    negative_loss = loss * negative.float()

    negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

    balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + eps)

    return balance_loss

def balance_bce_loss_m(pred, gt, mask):
    negative_ratio = 3.0
    eps = 1e-6
    positive = (gt * mask)
    negative = ((1 - gt) * mask)
    positive_count = int(positive.float().sum())
    # negative_count = min(
    #     int(negative.float().sum()),
    #     int(positive_count * negative_ratio))
    negative_count = 700

    assert gt.max() <= 1 and gt.min() >= 0
    assert pred.max() <= 1 and pred.min() >= 0
    loss = F.binary_cross_entropy(pred, gt, reduction='none')
    positive_loss = loss * positive.float()
    negative_loss = loss * negative.float()

    negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

    balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + eps)

    return balance_loss

# def balance_bce_loss_m(pred, gt, mask):
#     negative_ratio = 3.0
#     eps = 1e-6
#     positive = (gt * mask)
#     negative = ((1 - gt) * mask)
#     # positive_count = int(positive.float().sum())
#     # negative_count = min(
#     #     int(negative.float().sum()),
#     #     int(positive_count * negative_ratio))
#
#     assert gt.max() <= 1 and gt.min() >= 0
#     assert pred.max() <= 1 and pred.min() >= 0
#     loss = F.binary_cross_entropy(pred, gt, reduction='none')
#     positive_loss = loss * positive.float()
#     negative_loss = loss * negative.float()
#
#     total_loss=torch.cat([positive_loss,negative_loss],0)
#     total_loss,_=torch.topk(total_loss.view(-1),1000,sorted=True)
#
#
#     # negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
#
#     balance_loss = total_loss.sum() / (1000+eps)
#
#     return balance_loss