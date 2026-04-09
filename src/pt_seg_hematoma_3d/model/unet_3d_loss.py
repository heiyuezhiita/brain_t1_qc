from torch import nn
import torch
from einops import rearrange
from pt_seg_hematoma_3d.model.unet_3d_modules import PassLayer


# dice batch and channel-----------------------------------------------------------------------------------
def flatten_for_channel(tensor):
    """
    Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceBC(nn.Module):
    def __init__(self, nonlinear='Softmax', smooth=1e-8, is_batch_dice=False, is_channel_dice=True,
                 is_ignore_background=False):
        # The dimensions of the output are related to is_batch_dice and is_channel_dice
        super(DiceBC, self).__init__()
        self.smooth = smooth
        self.is_ignore_background = is_ignore_background

        # if nonlinear used, which will be applied to input
        if nonlinear == 'Softmax':
            self.nonlinear = nn.Softmax(dim=1)  # dim 1 is channel
        elif nonlinear == 'Sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif nonlinear == '':
            self.nonlinear = PassLayer()  # do nothing
        else:
            raise RuntimeError('Unsupport nonlinear in DiceBC: {}'.format(nonlinear))

        # 'flt' is 'flatten'
        if is_batch_dice and is_channel_dice:
            self.flt = 'b c z y x -> b c (z y x)'
        elif is_batch_dice and not is_channel_dice:
            self.flt = 'b c z y x -> b (c z y x)'
        elif not is_batch_dice and is_channel_dice:
            self.flt = 'b c z y x -> c (b z y x)'
        elif not is_batch_dice and not is_channel_dice:
            self.flt = 'b c z y x -> (b c z y x)'

    def forward(self, input, target):
        """
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
        """
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # apply nonlinear
        input = self.nonlinear(input)

        # dim 1 is channel (0 is first), channel 0 is background
        if self.is_ignore_background:
            input = input[:, 1:, ...]
            target = target[:, 1:, ...]

        input = rearrange(input, self.flt).float()
        target = rearrange(target, self.flt).float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        # denominator = (input * input).sum(-1) + (target * target).sum(-1)
        denominator = (input + target).sum(-1)

        # epsilon: prevents division by zero
        dice_each = (2 * intersect + self.smooth) / (denominator + self.smooth)

        return dice_each


class ArgmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    my input target is one-hot, so applying argmax
    """
    def forward(self, input, target):
        target = target.argmax(dim=1)  # [B, C, z, y, x] -> [B, z, y, x]
        return super().forward(input, target.long())


# dice batch/channel loss (can calculate dice in each batch/channel or not)
class DiceBCWeightLoss(nn.Module):
    def __init__(self, classes, nonlinear='Softmax',
                 weight=None, smooth=1e-8, is_batch_dice=False, is_channel_dice=True,
                 is_ignore_background=False):
        # default loss is mean channel loss
        super(DiceBCWeightLoss, self).__init__()
        self.classes = classes
        self.smooth = smooth

        self.is_channel_dice = is_channel_dice
        self.is_batch_dice = is_batch_dice

        # only is_channel is True, weight will be used
        if weight is None:
            self.weight = torch.ones(self.classes)
        elif torch.is_tensor(weight):
            self.weight = weight
        elif isinstance(weight, list) or isinstance(weight, tuple):
            self.weight = torch.FloatTensor(weight)
        else:
            raise RuntimeError(
                "Type of 'weight' must in: [torch.tensor, list, tuple, None], input type is {}".format(type(weight)))

        # remove weight of background
        if is_ignore_background:
            self.weight = self.weight[1:]

        self.DiceC = DiceBC(nonlinear=nonlinear, smooth=self.smooth,
                            is_batch_dice=is_batch_dice, is_channel_dice=is_channel_dice,
                            is_ignore_background=is_ignore_background)

    def forward(self, input, target):
        # weight of class
        dice_each = self.DiceC(input=input, target=target)  # dice (may get by each batch/channel)

        if self.is_channel_dice:
            weight = self.weight.to(dice_each)  # weight of each classes (channel), [C]

            # if is_batch_dice and is_channel_dice, out shape is [B, C]
            # else if only is_channel_dice, out shape is [C]
            if self.is_batch_dice:
                dice_each = torch.mean(dice_each, 0)  # [B, C] -> [C]
            weight_dice = (weight * dice_each).sum() / weight.sum()  # [C] -> [1]
        else:
            weight_dice = torch.mean(dice_each)

        return -weight_dice


class CEDiceWeightLoss(nn.Module):
    # Linear combination of CE and Dice losses
    def __init__(self, classes, ce_kwargs, dice_kwargs,
                 weight_ce=1, weight_dice=1):
        # alpha is weight of BCE, beta is weight of dice
        # dice_class_weight is weight of each class in dice
        super(CEDiceWeightLoss, self).__init__()
        self.weight_ce = weight_ce
        self.ce = ArgmaxCrossEntropyLoss(**ce_kwargs)  # predict is not be softmax

        self.weight_dice = weight_dice
        self.dice_loss = DiceBCWeightLoss(classes=classes, **dice_kwargs)

    def forward(self, input, target):
        # add 1 to match raw BCEDice loss
        return self.weight_ce * self.ce(input, target) + self.weight_dice * (1 + self.dice_loss(input, target))


# Focal ----------------------------------------------------------------------------------------
class MultiFocalLoss(nn.Module):
    def __init__(self, classes, alpha=None, gamma=2):
        """
        Calculate Focal loss for each channel (class), and return mean
        FL = -alpha * (1-pt)^gamma * log(pt)
        pt is p if y=1, or (1-p) otherwise
        default alpha and gamma is recommended by raw paper
        Input predict_mask and true_mask is one-hot data, and predict mask is softmax data
        :param alpha: tensor with length is class. alpha of Focal loss, which is class weight
        :param gamma: int. Adjust weight of the difficult and easy sample. 0 is CE
        """
        super(MultiFocalLoss, self).__init__()
        self.classes = classes
        self.gamma = gamma
        # alpha is weight of class
        if alpha is None:
            self.alpha = torch.ones(self.classes)
        elif torch.is_tensor(alpha):
            self.alpha = alpha
        elif isinstance(alpha, list) or isinstance(alpha, tuple):
            self.alpha = torch.FloatTensor(alpha)
        else:
            raise RuntimeError(
                "Type of 'alpha' must in: [torch.tensor, list, tuple, None], input type is {}".format(type(alpha)))

    def forward(self, input, target):
        # change to [C, B * D1 * D2 * D3]
        input = flatten_for_channel(input)
        input = torch.clamp(input, min=1e-7)  # log(0) is Inf, so min = 1e-7

        target = flatten_for_channel(target)
        target = target.float()

        # get alpha [D1 * D2 * D3], weight of each voxel (class)
        weight = self.alpha[target.argmax(dim=0)].to(target)

        # - alpha * (1 - p) ^ gamma * y * log(p)
        focal_losses = - torch.pow((1 - input), self.gamma) * target * input.log()  # [C, D1 * D2 * D3]
        focal_losses = weight * focal_losses.sum(0)  # [D1 * D2 * D3]
        focal_loss = focal_losses.mean()  # float

        return focal_loss


# Bad loss, Focal is out of sync (maybe conflict) with Dice -------------------------------------------------------
# Not use
class FocalDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""
    def __init__(self, classes, w_focal=1, w_dice=1,
                 focal_cl_w=None, focal_gamma=2, dice_cl_w=None, dice_smooth=1,
                 is_batch_dice=False, is_channel_dice=True):
        # alpha is weight of BCE, beta is weight of dice
        # dice_class_weight is weight of each class in dice
        super(FocalDiceLoss, self).__init__()
        self.w_focal = w_focal
        self.focal = MultiFocalLoss(classes=classes, alpha=focal_cl_w, gamma=focal_gamma)  # predict has been softmax
        self.w_dice = w_dice
        self.dice_loss = DiceBCWeightLoss(classes=classes, weight=dice_cl_w, smooth=dice_smooth,
                                          is_batch_dice=is_batch_dice, is_channel_dice=is_channel_dice)

    def forward(self, input, target):
        return self.w_focal * self.focal(input, target) + self.w_dice * self.dice_loss(input, target)


# Deep Supervision loss ----------------------------------------------------------------------------
# adjusting from nnUnet
class DeepSupervisionLoss(nn.Module):
    # get deep supervision loss from base loss function
    def __init__(self, loss, weights=None):
        # alpha is weight of BCE, beta is weight of dice
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = loss
        self.weights = weights

    def forward(self, input, target):
        # input and target is tensor list/tuple
        assert isinstance(input, (tuple, list)), "input must be either tuple or list"
        assert isinstance(target, (tuple, list)), "target must be either tuple or list"
        if self.weights is None:
            weights = [1] * len(input)
        else:
            weights = self.weights

        loss = weights[0] * self.base_loss(input[0], target[0])
        for i in range(1, len(input)):
            if weights[i] != 0:
                loss += weights[i] * self.base_loss(input[i], target[i])

        return loss



