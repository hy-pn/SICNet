import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from typing import Optional
# Recommend
from torch.nn.modules.loss import _Loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
    
def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    
    if weight_pos is not None:        
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss
class FocalLoss2(nn.Module):
    def __init__(self, alpha, gamma=0., size_average=True):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, y_pred, y_true):

        logpt = F.log_softmax(y_pred)

        y_true = torch.argmax(y_true, dim=-1)
        y_true = torch.unsqueeze(y_true, dim=-1)
        logpt = logpt.gather(1, y_true)
        pt = Variable(logpt.data.exp())
        # y_true的第1维是通道维度，也即预测是哪一类的标签。gather取axis=1的索引，相当于y_true的值替换了列索引
        # 为了取到具体的值，还需要行索引，根据gather的算法，y_true中的每个值本身的位置，提供行索引。如果dim=0
        # 则本身的值替换行索引，本身的值所在的位置，提供列索引

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred.to(dtype=torch.float32))
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss
class MultiClass_DiceLoss(nn.Module):
    def __init__(self, 
                weight: torch.Tensor, 
                batch: Optional[bool] = True, 
                ignore_index: Optional[int] = -1,
                do_sigmoid: Optional[bool] = False,
                **kwargs,
                )->torch.Tensor:
        super(MultiClass_DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        if weight is None:
            self.weight = [1, 1 , 1 , 1 , 1 , 1 , 1]
        else:
            self.weight = weight
        self.do_sigmoid = do_sigmoid
        self.binary_diceloss = dice_loss(batch)

    def __call__(self, y_pred, y_true):
        if self.do_sigmoid:
            y_pred = torch.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true.long(), y_pred.shape[1])
        y_true = y_true.permute(0,3,1,2)
        total_loss = 0.0
        tmp_i = 0.0
        for i in range(y_pred.shape[1]):
            if i != self.ignore_index:
                diceloss = self.binary_diceloss(y_pred[:, i, :, :], y_true[:, i, :, :])
                total_loss += torch.mul(diceloss, self.weight[i])
                tmp_i += 1.0
        return total_loss / tmp_i


class mc_dice_bce_loss(nn.Module):
    """multi-class"""
    def __init__(self, weight=None, do_sigmoid = True,ignore_index=-1):
        super(mc_dice_bce_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight,ignore_index=ignore_index)
        # self.dice = MultiClass_DiceLoss(weight,ignore_index=-1, do_sigmoid = do_sigmoid)

    def __call__(self, scores, labels):

        if len(scores.shape) < 4:
            scores = scores.unsqueeze(1)
        # if len(labels.shape) < 4:
        #     labels = labels.unsqueeze(1)
        # diceloss = self.dice(scores, labels)+ diceloss
        bceloss = self.ce_loss(scores, labels)
        return bceloss

class unChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(unChangeSimilarity, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, p_out, q_out, mask_bin):
        p = torch.argmax(p_out, dim=1)
        q = torch.argmax(q_out, dim=1)

        p[mask_bin == 1] = -1
        q[mask_bin == 1] = -1
        loss = 0.5*self.criterion(p_out, q.long()) + 0.5*self.criterion(q_out, p.long())
        return loss




def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos
        
    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
class ContrastiveLoss1(nn.Module):
    def __init__(self, margin1 = 0.3, margin2=2.2, eps=1e-6):
        super(ContrastiveLoss1, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1-x2)
        dist_sq = torch.pow(diff+self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)

        # print(y.data.type(), mdist_pos.data.type(), mdist_neg.data.type())
        loss_pos =(1- y)*(mdist_pos.pow(2))
        loss_neg = y*(mdist_neg.pow(2))

        loss = torch.mean(loss_pos + loss_neg)


        return loss



