# 2025.03.07 add some loss from ShenSiCong
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss


class SingleClassificationAccuracyLoss(torch.nn.Module):
    def __init__(self):
        super(SingleClassificationAccuracyLoss, self).__init__()

    def forward(self, x, y):
        # x and y is onehot, and float
        # can be used in any number of classes, but only one classes is target for a task
        # [B, classes], x is float, and the max is 1 and other is 0
        # e.g.: (batch size is 3, and classes is 2)
        # x: [[0.2, 0.8], [0.7, -1.5], [1.2, 0.8]]
        # y: [[1, 0], [0, 1], [0, 1]]
        loss = (x.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        return loss


class GetBinaryConfusionMatrixTPFPTNFN(torch.nn.Module):
    def __init__(self):
        super(GetBinaryConfusionMatrixTPFPTNFN, self).__init__()

    def forward(self, x, y):
        # x and y is onehot, and float
        # can be used in any number of classes, but only one classes is target for a task
        # [B, classes], x is float, and the max is 1 and other is 0
        # e.g.: (batch size is 3, and classes is 2)
        # x: [[0.2, 0.8], [0.7, -1.5], [1.2, 0.8]]
        # y: [[1, 0], [0, 1], [0, 1]]
        xc = x.argmax(dim=1)  # xc means class of x
        yc = y.argmax(dim=1)  # yc means class of y
        
        tn = ((xc == 0) & (yc == 0)).float().sum()
        fp = ((xc == 1) & (yc == 0)).float().sum()
        fn = ((xc == 0) & (yc == 1)).float().sum()
        tp = ((xc == 1) & (yc == 1)).float().sum()
        
        return tn, fp, fn, tp
    

# 2025.03.07 add some loss from ShenSiCong start ==========================================
class ST_CE_loss(nn.Module):
    """
        CE loss, timm implementation for mixup
    """
    def __init__(self):
        super(ST_CE_loss, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
    def __init__(self, args):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(args.cls_num)
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = 1 * prior
        self.prior = self.prior.to(args.device)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior
        x = x + prior
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class LS_CE_loss(nn.Module):
    """
        label smoothing without mixup
    """
    def __init__(self, smoothing=0.1):
        super(LS_CE_loss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # 2025.03.15 debug, target should be index label but not one-hot label
        target = torch.argmax(target, dim=1)
        
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MiSLAS_loss(nn.Module):
    ''' 
        Paper: Improving Calibration for Long-Tailed Recognition
        Code: https://github.com/Jia-Research-Lab/MiSLAS
    '''
    def __init__(self, args, shape='concave', power=None):
        super(MiSLAS_loss, self).__init__()

        cls_num_list = args.cls_num
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        smooth_head = 0.3
        smooth_tail = 0.0

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float().to(args.device)

    def forward_oneway(self, x, target):
        # smooth = self.smooth.to(x.device)
        smooth = self.smooth
        smoothing = smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            '''
                x.shape: batch * nClass
                target: one hot [0, 0, 0, 0.4, 0, 0, 0.6, 0, 0, 0]
            '''
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class LADE_loss(nn.Module):
    '''NOTE can not work with mixup, plz set mixup=0 and cutmix=0
        Paper: Disentangling Label Distribution for Long-tailed Visual Recognition
        Code: https://github.com/hyperconnect/LADE
    '''
    def __init__(self, args, remine_lambda=0.1):
        super().__init__()
        cls_num = torch.tensor(args.cls_num)
        self.prior = cls_num / torch.sum(cls_num)
        self.num_classes = args.nb_classes
        self.balanced_prior = torch.tensor(1. / self.num_classes).float()
        self.remine_lambda = remine_lambda
        self.cls_weight = (cls_num.float() / torch.sum(cls_num.float()))
        
        # to device
        self.prior = self.prior.to(args.device)
        self.balanced_prior = self.balanced_prior.to(args.device)
        self.cls_weight = self.cls_weight.to(args.device)

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, x, target, q_pred=None):
        """
            x: N x C
            target: N
        """
        # prior = self.prior.to(x.device)
        # balanced_prior = self.balanced_prior.to(x.device)
        # cls_weight = self.cls_weight.to(x.device)
        
        # 2025.03.15 debug, target should be index label but not one-hot label
        target = torch.argmax(target, dim=1)
        
        prior = self.prior
        balanced_prior = self.balanced_prior
        cls_weight = self.cls_weight
        per_cls_pred_spread = x.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (x - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T  # C x N
        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, _, _ = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        return - torch.sum(estim_loss * cls_weight)


class LDAM_loss(nn.Module):
    '''
        Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
        Code: https://github.com/kaidic/LDAM-DRW
    '''
    def __init__(self, args):
        super(LDAM_loss, self).__init__()
        cls_num_list = args.cls_num
        self.drw = False
        self.epoch = 0
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).to(args.device)
        self.s = 30

    def forward_oneway(self, x, target):
        # m_list = self.m_list.to(x.device)
        m_list = self.m_list
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, reduction='none')

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class CB_CE_loss(nn.Module):
    '''
        Paper: Class-Balanced Loss Based on Effective Number of Samples
        Code: https://github.com/richardaecn/class-balanced-loss
    '''
    def __init__(self, args):
        super(CB_CE_loss, self).__init__()
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, args.cls_num)
        weight = (1.0 - beta) / np.array(effective_num)
        weight = weight / np.sum(weight) * len(args.cls_num)
        self.weight = torch.FloatTensor(weight).to(args.device)

    def forward(self, x, target):
        # weight = self.weight.to(x.device)
        weight = self.weight
        return F.cross_entropy(input = x, target = target, weight = weight)
    
# 2025.03.07 add some loss from ShenSiCong end ==========================================

# 2025.03.21, add diff loss, start ======================================
# diff loss from ShenSiCong
class Diff_Label01_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, labels, datas):
        """
        Contrast learning loss. 
        It is assumed that the features of label0 are consistent, while those of label1 are far away from label0

        Args:
            labels (_type_): one-hot label 
            datas (_type_): Features of each sample, such as the input of the last FC_layer

        Returns:
            _type_: Contrast learning loss
        """
        # one-hot label to index label
        labels = torch.argmax(labels, dim=1)
        
        # get feature vectors with label 0
        normal_vectors = datas[labels == 0]
        except_vectors = datas[labels == 1]
        sim_loss = 0
        if normal_vectors.numel() == 0:
            # Make sure the output are tensor
            sim_loss = torch.tensor(0.0).to(datas.device)
            differ_loss = torch.tensor(0.0).to(datas.device)
            total_loss = sim_loss + differ_loss
            return total_loss, sim_loss, differ_loss
        
        # get mean
        mean_vector = torch.mean(normal_vectors, dim=0)

        # get sim loss (Theoretically the feature vector of label0 should be similar to each other)
        for i in range(len(normal_vectors)):
            sim_loss += (1 - torch.abs(F.cosine_similarity(normal_vectors[i], mean_vector, dim=0)))
        sim_loss = sim_loss / len(normal_vectors)
        if except_vectors.numel() == 0:
            # Make sure the output are tensor
            differ_loss = torch.tensor(0.0).to(datas.device)
            total_loss = sim_loss + differ_loss
            
            return total_loss, sim_loss, differ_loss

        # get diff loss (Theoretically the feature vectors of label1 should be different from label0's)
        differ_loss = 0
        for i in range(len(except_vectors)):
            differ_loss += torch.abs(F.cosine_similarity(except_vectors[i], mean_vector, dim=0))
        differ_loss = differ_loss / len(except_vectors)
        
        # sim_loss and differ_loss are Tensor
        total_loss = sim_loss + differ_loss
            
        return total_loss, sim_loss, differ_loss


class DynamicByEpoch_DiffLoss(nn.Module):
    def __init__(self, main_loss, diff_loss,
                 epoch_add_diff_loss=0, diff_loss_weight=1,
                 features_name_in_hook_tool="last_fc_input"):
        super().__init__()
        self.main_loss = main_loss
        self.diff_loss = diff_loss
        self.epoch_add_diff_loss = epoch_add_diff_loss
        self.diff_loss_weight = diff_loss_weight
        self.features_name_in_hook_tool = features_name_in_hook_tool
        
        # 2025.04.20 update, for net_run
        self.is_need_hook_tool = True
        
    def get_is_using_mix_loss(self, epoch):
        return epoch >= self.epoch_add_diff_loss

    def forward(self, x, target, is_using_mix_loss, hook_tool):
        main_loss_value = self.main_loss(x, target)
        # We assume that the output of diff loss consists of total_loss, sim_loss, differ_loss
        # diff_loss_value_tuple[0-2] is total_loss (for self.diff_loss), sim_loss, differ_loss
        if is_using_mix_loss:
            diff_loss_value_tuple = self.diff_loss(target, hook_tool.features[self.features_name_in_hook_tool])    
        else:
            diff_loss_value_tuple = (
                torch.tensor(0.0).to(x.device),
                torch.tensor(0.0).to(x.device),
                torch.tensor(0.0).to(x.device))
            
        total_loss_value = main_loss_value + diff_loss_value_tuple[0] * self.diff_loss_weight
        
        # 2025.03.28, Multiple losses are allowed for return, and the first of which is used for backpropagation
        return (total_loss_value,
                main_loss_value, 
                diff_loss_value_tuple[0], 
                diff_loss_value_tuple[1], 
                diff_loss_value_tuple[2])

# 2025.03.21, add diff loss, end ======================================


# 2025.04.16, add BSS loss, start =====================
class BatchSpectralShrinkage_Loss(nn.Module):
    r"""
    The regularization term in `Catastrophic Forgetting Meets Negative Transfer:
    Batch Spectral Shrinkage for Safe Transfer Learning (NIPS 2019) <https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf>`_.


    The BSS regularization of feature matrix :math:`F` can be described as:torch svd

    .. math::
        L_{bss}(F) = \sum_{i=1}^{k} \sigma_{-i}^2 ,

    where :math:`k` is the number of singular values to be penalized, :math:`\sigma_{-i}` is the :math:`i`-th smallest singular value of feature matrix :math:`F`.

    All the singular values of feature matrix :math:`F` are computed by `SVD`:

    .. math::
        F = U\Sigma V^T,

    where the main diagonal elements of the singular value matrix :math:`\Sigma` is :math:`[\sigma_1, \sigma_2, ..., \sigma_b]`.


    Args:
        k (int):  The number of singular values to be penalized. Default: 1

    Shape:
        - Input: :math:`(b, |\mathcal{f}|)` where :math:`b` is the batch size and :math:`|\mathcal{f}|` is feature dimension.
        - Output: scalar.

    """
    def __init__(self, k=1):
        super(BatchSpectralShrinkage_Loss, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result


class DynamicByEpoch_BSSLoss(nn.Module):
    def __init__(self, main_loss, bss_loss,
                 epoch_add_bss_loss=0, bss_loss_weight=1,
                 features_name_in_hook_tool="last_fc_input"):
        super().__init__()
        self.main_loss = main_loss
        self.bss_loss = bss_loss
        self.epoch_add_bss_loss = epoch_add_bss_loss
        self.bss_loss_weight = bss_loss_weight
        self.features_name_in_hook_tool = features_name_in_hook_tool
        
        # 2025.04.20 update, for net_run
        self.is_need_hook_tool = True

    def get_is_using_mix_loss(self, epoch):
        return epoch >= self.epoch_add_bss_loss

    def forward(self, x, target, is_using_mix_loss, hook_tool):
        main_loss_value = self.main_loss(x, target)
        # TODO ??? 2025.04.20,  maybe "bss_loss_value" is not a tuple...
        # We assume that the output of bss loss consists of total_loss, sim_loss, bsser_loss
        # bss_loss_value_tuple[0-2] is total_loss (for self.bss_loss), sim_loss, bsser_loss
        if is_using_mix_loss:
            bss_loss_value = self.bss_loss(hook_tool.features[self.features_name_in_hook_tool])
        else:
            bss_loss_value = torch.tensor(0.0).to(x.device)

        total_loss_value = main_loss_value + bss_loss_value * self.bss_loss_weight

        # 2025.03.28, Multiple losses are allowed for return, and the first of which is used for backpropagation
        return (total_loss_value,
                main_loss_value,
                bss_loss_value)
# 2025.04.16, add BSS loss, end =====================


