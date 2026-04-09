# 2021.11.07 Unet 3d modules
import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from einops import rearrange
from copy import deepcopy


class Double3dConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='l', padding=1, kernel_size=3, stride=1):
        super(Double3dConv, self).__init__()
        comb_layer = partial(_combine_layers, out_ch=out_ch, modules_list=['c', norm_l, act_l],
                             padding=padding, kernel_size=kernel_size, stride=stride)

        self.conv = nn.Sequential(
            comb_layer(in_ch=in_ch),
            comb_layer(in_ch=out_ch),
        )

    def forward(self, x):
        return self.conv(x)


# 2 conv: (conv3d + InsNorm + LeakyReLu)*2, the 2nd conv will pool image by stride=2
class Double3dConvPoolStride(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(Double3dConvPoolStride, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            # nn.BatchNorm3d(out_ch),  # add BN
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, stride=stride),
            nn.InstanceNorm3d(out_ch, affine=True),
            # nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# modules for Res-Unet ------------------------------------------------
# Res conv block (BN + LeakyReLU + Conv)
class ResProDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='r'):
        super(ResProDouble3dConv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.BatchNorm3d(in_ch),  # add BN
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv3d(in_ch, out_ch, 3, padding=1),
        #     nn.BatchNorm3d(out_ch),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv3d(out_ch, out_ch, 3, padding=1),
        # )

        self.conv1 = nn.Sequential(
            _combine_layers(in_ch, in_ch, [norm_l, act_l]),
            _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

        self.conv111 = nn.Conv3d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        return self.conv1(x) + self.conv111(x)


# Res conv block (Conv + BN + LeakyReLU)
class U3dResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='r'):
        super(U3dResBlock, self).__init__()
        # conv
        self.conv1 = nn.Sequential(
            _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1),
            _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        )

        # short
        self.conv111 = _combine_layers(in_ch, out_ch, ['c'], padding=0, kernel_size=1, stride=1)
        self.act = _combine_layers(out_ch, out_ch, [act_l])

    def forward(self, x):
        x1 = self.conv111(x)  # short
        x2 = self.conv1(x)  # conv

        return self.act(x1 + x2)


# modules for Vnet ==========================================================
# conv, normalise, activate_function
def _combine_layers(in_ch, out_ch, modules_list, padding=1, kernel_size=3, stride=1, bias=True,
                    gn_group_num=16, se_reduction=16):
    """
    get combine conv layers (Conv - normalise - activate)
    Conv: 'c' is conv3d, 't' is ConvTranspose3d, 'L' is Linear
    Normalise: 'b' is BN, 'i' is InstanceNorm, 'g' is GroupNorm
    Active function: 'l' is LeakyRelu, 'r' is Relu, 'e' is ELU, 'p' is PReLU, 's' is Sigmoid, 'ge' is GELU
    Other Layer: 'se' is SE layer
    :param in_ch: int, used in conv
    :param out_ch: int, used in conv and other layers
    :param modules_list: list or str, the key arg of this function.
                         Each element in this arg will be used to add a layer
                         If all modules name will be used is single char, you can use str, else input list or tuple
    :param padding: int, used in conv
    :param kernel_size: int, used in conv
    :param stride: int, used in conv
    """
    layer_list = []

    for layer_name in modules_list:
        # Conv
        if layer_name == 'c':
            layer_list.append(nn.Conv3d(in_ch, out_ch,
                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif layer_name == 't':
            layer_list.append(nn.ConvTranspose3d(in_ch, out_ch,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif layer_name == 'L':
            layer_list.append(nn.Linear(in_ch, out_ch))
        # Normalise
        elif layer_name == 'b':
            layer_list.append(nn.BatchNorm3d(out_ch))
        elif layer_name == 'i':
            layer_list.append(nn.InstanceNorm3d(out_ch, affine=True))
        elif layer_name == 'g':
            layer_list.append(nn.GroupNorm(num_groups=math.ceil(out_ch / gn_group_num), num_channels=out_ch))
        # Active
        elif layer_name == 'l':
            layer_list.append(nn.LeakyReLU(inplace=True))
        elif layer_name == 'r':
            layer_list.append(nn.ReLU(inplace=True))
        elif layer_name == 'e':
            layer_list.append(nn.ELU(inplace=True))
        elif layer_name == 'p':
            layer_list.append(nn.PReLU(num_parameters=out_ch))
        elif layer_name == 's':
            layer_list.append(nn.Sigmoid())
        elif layer_name == 'ge':
            layer_list.append(nn.GELU())
        # other
        elif layer_name == 'se':
            layer_list.append(SELayer(in_ch=out_ch, reduction=se_reduction))
        else:
            raise RuntimeError('Create Combine Conv layers Error! Unsupported layer name: {}!'.format(layer_name))

    return nn.Sequential(*layer_list)


# multi conv layer (each layer contains conv + norm + act)
def _multi_conv_layer3d(in_ch, out_ch, n_conv, modules_list, padding=1, kernel_size=2, stride=1):
    layer_list = []
    for i in range(n_conv):
        if i > 0:
            in_ch = out_ch  # debug, if n_conv > 1 and in_ch != out_ch, will get error
        layer_list.append(_combine_layers(in_ch, out_ch, modules_list,
                                          padding=padding, kernel_size=kernel_size, stride=stride))
    return nn.Sequential(*layer_list)


# Input modules of Vnet, raw image will be copy n times as out channel
class VnetInputConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='p'):
        # norm_l is norm layer, act_l is active layer (active function)
        self.out_ch = out_ch

        super(VnetInputConv, self).__init__()

        self.conv1 = _combine_layers(in_ch=in_ch, out_ch=out_ch, modules_list=['c', norm_l, act_l],
                                     padding=1, kernel_size=3, stride=1)
        # in Vnet, copy raw image n times to match channel, but this trick not suitable of multi-channel input
        # So, use 1*1*1 conv replace copy raw image
        self.conv111 = _combine_layers(in_ch=in_ch, out_ch=out_ch, modules_list=['c', norm_l, act_l],
                                       padding=0, kernel_size=1, stride=1)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        # x_direct = self.cat(tuple([x] * self.out_ch), 0)  # which is raw Vnet method
        x_skip = self.conv111(x)
        out = self.act(x1 + x_skip)

        return out


# down sample by stride
class VnetDownConv(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv, norm_l='b', act_l='p'):
        super(VnetDownConv, self).__init__()
        # down sample conv
        self.down_conv = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=0, kernel_size=2, stride=2)
        # conv
        self.conv_layers = _multi_conv_layer3d(out_ch, out_ch, n_conv=n_conv,
                                               modules_list=['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        x1 = self.down_conv(x)
        x2 = self.conv_layers(x1)
        out = self.act(x1 + x2)  # res connect, with in down block

        return out


# up sample by stride
class VnetUpConv(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv, norm_l='b', act_l='p'):
        super(VnetUpConv, self).__init__()

        # need to concatenate down and up, so set the channel of up is out_channel / 2
        # ???? why should to /2 ?
        self.up_conv = _combine_layers(in_ch, out_ch // 2, ['t', norm_l, act_l], padding=0, kernel_size=2, stride=2)

        self.conv_layers = _multi_conv_layer3d(out_ch, out_ch, n_conv=n_conv,
                                               modules_list=['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.act = nn.PReLU(out_ch)

    def forward(self, x, x_skip):
        # channel of x_skip should be half of out_channel
        x1 = self.up_conv(x)
        x2 = torch.cat((x1, x_skip), dim=1)  # skip connect, down + up

        x3 = self.conv_layers(x2)

        out = self.act(x2 + x3)  # res connect, within up block

        return out


# out of Vnet
class VnetOutputConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='p'):
        super(VnetOutputConv, self).__init__()

        # need to concatenate down and up, so set the channel of up is out_channel / 2
        # cbe is Conv + BN + PReLU;
        self.conv1 = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        # cb is Conv + BN, not us relu because Softmax will be applied
        # self.conv2 = _combine_layers(out_ch, out_ch, ['c', norm_l], padding=0, kernel_size=1, stride=1)  # remove norm
        self.conv2 = _combine_layers(out_ch, out_ch, 'c', padding=0, kernel_size=1, stride=1)

        # self.act = nn.Softmax(dim=1)  # dim 1 is channel

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.act(x2)

        return out


# Unet layer by _combine_layers -------------------------------------------------------------------
class PassLayer(nn.Module):
    # not do anything
    def __init__(self):
        super(PassLayer, self).__init__()

    def forward(self, x):
        return x


class U3dDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='b', act_l='l'):
        # pool_l must in ['m', 'c', 'c3', None]
        super(U3dDownBlock, self).__init__()

        if pool_l == 'm':
            self.pool = nn.MaxPool3d(kernel_size=2)
        elif pool_l == 'c':
            self.pool = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=2)
        elif pool_l == 'c3':  # conv1 not as PassLayer in 'c3'
            self.pool = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=2)
        elif pool_l is None:
            self.pool = PassLayer()  # used in input block

        # cbe is Conv + BN + LeakyReLU;
        if pool_l == 'c':  # conv1 was replaced by pool (conv with stride=2)
            self.conv1 = PassLayer()
        elif pool_l == 'c3':
            self.conv1 = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        else:
            self.conv1 = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv2 = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2(out)

        return out


class U3dUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l', ct_out_bias=True):
        super(U3dUpBlock, self).__init__()
        self.up_conv = _combine_layers(in_ch, out_ch, 't', padding=0, kernel_size=2, stride=2, bias=ct_out_bias)

        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(out_ch*2, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv2 = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)

    def forward(self, x, x_pass):
        # channel of x_pass must same as out channel
        x1 = self.up_conv(x)

        out = torch.cat((x1, x_pass), dim=1)  # new channel: out_ch + x_pass channel
        out = self.conv1(out)
        out = self.conv2(out)

        return out
    
    
# attention Unet -------------------------------------------------------------------------------
class U3dUpAttentionBlock(nn.Module):
    def __init__(self, in_up_ch, in_sh_ch, out_ch, norm_l='b', act_l='l'):
        """
        Up block of U3d, 'attention' means applying the weight to short image
        :param in_up_ch: int, channel of up image
        :param in_sh_ch: int, channel of short image, which must same as out_ch
        :param out_ch: int, channel of out image
        :param norm_l: str, name of normalization layer
        :param act_l: str, name of active layer
        """
        super(U3dUpAttentionBlock, self).__init__()
        # up sample
        self.up_conv = _combine_layers(in_up_ch, out_ch, 't', padding=0, kernel_size=2, stride=2)

        # Get weight of short image
        # out of up_conv and sh_conv will be add, and using sigmoid convert added image to 0-1 weight
        self.up_img_conv = _combine_layers(out_ch, out_ch, ['c', norm_l], padding=0, kernel_size=1, stride=1)
        self.sh_img_conv = _combine_layers(in_sh_ch, out_ch, ['c', norm_l], padding=0, kernel_size=1, stride=1)
        # the weight will be multiple to short image
        self.add_act = _combine_layers(out_ch, out_ch, act_l)
        self.weight_conv = _combine_layers(out_ch, 1, ['c', norm_l, 's'],
                                           padding=0, kernel_size=1, stride=1)

        # Conv block * 2  (concatenated up image and weighted short image)
        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(out_ch*2, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv2 = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        
    def forward(self, x, x_pass):
        # up
        x1 = self.up_conv(x)

        # attention, get weight
        x2 = self.up_img_conv(x1)
        x_pass2 = self.sh_img_conv(x_pass)
        x_add = self.add_act(x2 + x_pass2)
        x_weight = self.weight_conv(x_add)

        x_pass3 = x_weight * x_pass

        # conv
        x3 = torch.cat((x1, x_pass3), dim=1)
        out = self.conv1(x3)
        out = self.conv2(out)

        return out


# Recurrent conv --------------------------------------------------------------------------------
class RecBlock(nn.Module):
    def __init__(self, out_ch, t, norm_l='b', act_l='l'):
        # must have same channel
        super(RecBlock, self).__init__()
        self.t = t
        self.conv_r = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                out = self.conv_r(x)
            out = self.conv_r(out + x)
        return out


class R2U3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t, norm_l='b', act_l='l'):
        # must have same channel
        super(R2U3dBlock, self).__init__()
        self.t = t
        self.conv111 = nn.Conv3d(in_ch, out_ch, 1, padding=0)

        self.conv_r = nn.Sequential(
            RecBlock(out_ch=out_ch, t=t, norm_l=norm_l, act_l=act_l),
            RecBlock(out_ch=out_ch, t=t, norm_l=norm_l, act_l=act_l),
        )

    def forward(self, x):
        x = self.conv111(x)
        x1 = self.conv_r(x)
        out = x + x1
        return out


# SE layer, attention of channel ---------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, in_ch, reduction=16, act_l='r'):
        super(SELayer, self).__init__()

        self.pool = nn.AdaptiveAvgPool3d(output_size=1)  # set out shape is [B * C], global average pool

        # get attention of channels
        self.fc = nn.Sequential(
            nn.Linear(in_ch, math.ceil(in_ch // reduction), bias=False),
            _combine_layers(None, None, [act_l]),  # default is relu
            # nn.ReLU(inplace=True),
            nn.Linear(math.ceil(in_ch // reduction), in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        return x * y.expand_as(x)


class U3dSEDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_pool=True, norm_l='b', act_l='l', se_reduction=16):
        super(U3dSEDownBlock, self).__init__()

        if is_pool:
            self.pool = nn.MaxPool3d(kernel_size=2)
        else:
            self.pool = PassLayer()  # used in input block

        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv2_se = _combine_layers(out_ch, out_ch, ['c', norm_l, 'se', act_l],
                                        padding=1, kernel_size=3, stride=1, se_reduction=se_reduction)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2_se(out)

        return out


class U3dSEUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l', se_reduction=16):
        super(U3dSEUpBlock, self).__init__()
        self.up_conv = _combine_layers(in_ch, out_ch, 't', padding=0, kernel_size=2, stride=2)

        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(out_ch*2, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv2_se = _combine_layers(out_ch, out_ch, ['c', norm_l, 'se', act_l],
                                        padding=1, kernel_size=3, stride=1, se_reduction=se_reduction)

    def forward(self, x, x_pass):
        # channel of x_pass must same as out channel
        x1 = self.up_conv(x)

        out = torch.cat((x1, x_pass), dim=1)  # new channel: out_ch + x_pass channel
        out = self.conv1(out)
        out = self.conv2_se(out)

        return out


# Unet++ -----------------------------------------------------------------------------------------
class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(UpSample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


# Trans Unet --------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_nums = head_num
        self.dk = (embedding_dim // head_num) ** (-1/2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.sf = nn.Softmax(dim=-1)

        self.out_att = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        qkv = self.qkv_layer(x)

        # quary, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_nums))
        qkv = rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_nums)
        quary = qkv[0, ...]
        key = qkv[1, ...]
        value = qkv[2, ...]

        # softmax((Q * KT) / sqrt(dk)) * V
        energy = torch.einsum('... t d, ... j d -> ... t j', quary, key) * self.dk
        attention = self.sf(energy)

        x_out = torch.einsum('...t j, ... j d -> ... t d', attention, value)

        # rearrange to b c embedding_dim
        x_out = rearrange(x_out, 'b h t d -> b t (h d)')
        x_out = self.out_att(x_out)

        return x_out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1),
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        x_att = self.multi_head_attention(x)
        x_att = self.dropout(x_att)

        x_out = x + x_att  # res
        x_out = self.layer_norm1(x_out)

        x_mlp = self.mlp(x_out)
        x_out = x_out + x_mlp  # res
        x_out = self.layer_norm2(x_out)

        return x_out


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layers = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)]
        )

    def forward(self, x):
        return self.layers(x)


class ViT3d(nn.Module):
    def __init__(self, img_shape, in_ch, embedding_dim, mlp_dim, head_num, block_num):
        """
        patch dim is 1, i.e., each voxel is a patch
        :param img_shape: input is [b, c, z, y, x], img_shape is z*y*x, same as tokens number
        :param in_ch:
        :param embedding_dim:
        :param mlp_dim:
        :param head_num:
        :param block_num:
        """
        super(ViT3d, self).__init__()

        self.projection = nn.Linear(in_ch, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(img_shape, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

    def forward(self, x):
        z_dim, y_dim, x_dim = x.size()[-3:]
        x_patch = rearrange(x, 'b c z y x -> b (z y x) c')

        x_project = self.projection(x_patch)

        x_patches = x_project + self.embedding
        x_patches = self.dropout(x_patches)

        x_trans = self.transformer(x_patches)

        x_out = rearrange(x_trans, 'b (z y x) c -> b c z y x', z=z_dim, y=y_dim, x=x_dim)

        return x_out


# DR-Unet ------------------------------------------------------------------------------------------
# which DRBlock1 is dimension reduction, out channel will be set to out_ch / 4
# in raw paper, block1 is same to block2 (using 1 to down and 2 to up)
class DRBlock1(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l'):
        # pool_l must in ['m', 'c', 'c2', None]
        super(DRBlock1, self).__init__()

        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=0, kernel_size=1, stride=1)
        self.conv2 = _combine_layers(out_ch, out_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv3 = _combine_layers(out_ch, out_ch, ['c', norm_l], padding=1, kernel_size=3, stride=1)

        self.conv_res = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=0, kernel_size=1, stride=1)

        self.active = _combine_layers(out_ch, out_ch, ['e'])  # ELU

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)

        out_res = self.conv_res(x)

        out = self.active(out1 + out_res)

        return out


# which DRBlock1 is dimension reduction, out channel will be set to out_ch / 4
class DRBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l'):
        # pool_l must in ['m', 'c', 'c2', None]
        super(DRBlock2, self).__init__()

        # cbe is Conv + BN + LeakyReLU;
        self.conv1 = _combine_layers(in_ch, in_ch, ['c', norm_l, act_l], padding=0, kernel_size=1, stride=1)
        self.conv2 = _combine_layers(in_ch, in_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv3 = _combine_layers(in_ch, out_ch, ['c', norm_l], padding=1, kernel_size=3, stride=1)

        self.conv_res = _combine_layers(in_ch, out_ch, ['c', norm_l, act_l], padding=0, kernel_size=1, stride=1)

        self.active = _combine_layers(out_ch, out_ch, ['e'])  # ELU

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)

        out_res = self.conv_res(x)

        out = self.active(out1 + out_res)

        return out


class DRU3dDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='b', act_l='l'):
        # pool_l must in ['m', 'c', 'c2', None]
        super(DRU3dDownBlock, self).__init__()

        if pool_l == 'm':
            self.pool = nn.MaxPool3d(kernel_size=2)
        elif pool_l == 'c2':  # conv1 not as PassLayer in 'c3'
            self.pool = _combine_layers(in_ch, in_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=2)
        elif pool_l is None:
            self.pool = PassLayer()  # used in input block

        # using block1 and block2 (in raw paper is block 1 and 3)
        self.block1 = DRBlock1(in_ch, out_ch // 4, norm_l=norm_l, act_l=act_l)  # out ch is out_ch // 4
        self.block2 = DRBlock2(out_ch // 4, out_ch, norm_l=norm_l, act_l=act_l)

    def forward(self, x):
        out = self.pool(x)
        out = self.block1(out)
        out = self.block2(out)

        return out


class DRU3dUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l', is_norm_act_concate=False):
        # up_l must in ['t', ['t', norm_l, act_l]]
        super(DRU3dUpBlock, self).__init__()

        self.up_conv = _combine_layers(in_ch, out_ch, 't', padding=0, kernel_size=2, stride=2)

        if is_norm_act_concate:
            self.norm_act = _combine_layers(out_ch*2, out_ch*2, [norm_l, act_l], padding=0, kernel_size=2, stride=2)
        else:
            self.norm_act = PassLayer()

        # using block1 and block2 (in raw paper is block 1 and 3)
        self.block1 = DRBlock1(out_ch*2, out_ch // 4, norm_l=norm_l, act_l=act_l)
        self.block2 = DRBlock2(out_ch // 4, out_ch, norm_l=norm_l, act_l=act_l)

    def forward(self, x, x_pass):
        x1 = self.up_conv(x)

        out = torch.cat((x1, x_pass), dim=1)  # new channel: out_ch + x_pass channel
        out = self.norm_act(out)

        out = self.block1(out)
        out = self.block2(out)

        return out


# MultiResUnet --------------------------------------------------------------------------------------
class MultiResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l', is_act_first=True):
        """
        Same as split out_ch to 0.167, 0.333, 0.5, and concatenate it
        :param in_ch:
        :param out_ch: which is 'W' in raw code
        :param norm_l:
        :param act_l:
        """
        super(MultiResBlock, self).__init__()
        ch_1 = int(out_ch * 0.167)
        ch_2 = int(out_ch * 0.333)
        ch_3 = out_ch - (ch_1 + ch_2)  # maintain out channel = out_ch, and ch_3 similar to int(out_ch * 0.5)

        # cbe is Conv + BN + LeakyReLU;
        # 333, 555, 777 means which is equivalent to the convolution kernel of size 3, 5, 7
        self.conv333 = _combine_layers(in_ch, ch_1, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv555 = _combine_layers(ch_1, ch_2, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)
        self.conv777 = _combine_layers(ch_2, ch_3, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)

        self.conv_concate_norm = _combine_layers(out_ch, out_ch, [norm_l])

        # no activate in short cut
        self.conv_res = _combine_layers(in_ch, out_ch, ['c', norm_l], padding=0, kernel_size=1, stride=1)

        if is_act_first:
            self.active_norm = _combine_layers(out_ch, out_ch, [act_l, norm_l])  # why which is first active and norm
        else:
            self.active_norm = _combine_layers(out_ch, out_ch, [norm_l, act_l])

    def forward(self, x):
        # conv and concate
        out1 = self.conv333(x)
        out2 = self.conv555(out1)
        out3 = self.conv777(out2)

        # out_conv = torch.cat((out1, out2, out3), dim=1)  # concate out 123
        # out_conv = self.conv_concate_norm(out_conv)
        #
        # # res
        # out_res = self.conv_res(x)
        #
        # # add and active+norm
        # out = self.active_norm(out_conv + out_res)

        # add and active+norm
        out = self.active_norm(self.conv_concate_norm(torch.cat((
            out1, out2, out3), dim=1)) + self.conv_res(x))

        return out


class ResPathBaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='b', act_l='l', is_act_first=True):
        super(ResPathBaseBlock, self).__init__()

        # conv 1*1*1 and 3*3*3
        self.conv111 = _combine_layers(in_ch, out_ch,
                                       ['c', norm_l], padding=0, kernel_size=1, stride=1)
        self.conv333 = _combine_layers(in_ch, out_ch,
                                       ['c', norm_l, act_l], padding=1, kernel_size=3, stride=1)

        if is_act_first:
            self.active_norm = _combine_layers(out_ch, out_ch, [act_l, norm_l])
        else:
            self.active_norm = _combine_layers(out_ch, out_ch, [norm_l, act_l])

    def forward(self, x):
        # conv and concate
        # out1 = self.conv111(x)
        # out2 = self.conv333(x)
        #
        # out = self.active_norm(out1 + out2)

        # not use intermediate variable can save gpu memory
        out = self.active_norm(torch.add(self.conv111(x), self.conv333(x)))
        return out


class ResPathBlock(nn.Module):
    def __init__(self, in_ch, out_ch, length, norm_l='b', act_l='l', is_act_first=True):
        super(ResPathBlock, self).__init__()

        res_path_list = list()

        # There's at least one ResPathConv
        res_path_list.append(ResPathBaseBlock(in_ch, out_ch, norm_l, act_l, is_act_first=is_act_first))
        for i in range(length-1):
            res_path_list.append(ResPathBaseBlock(out_ch, out_ch, norm_l, act_l, is_act_first=is_act_first))

        self.res_path = nn.Sequential(*res_path_list)

    def forward(self, x):
        return self.res_path(x)


class MultiResU3dDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='b', act_l='l', is_act_first=True):
        """
        :param in_ch:
        :param out_ch:
        :param pool_l: pool_l must in ['m', 'c', 'c2', None]
        :param norm_l:
        :param act_l:
        """
        super(MultiResU3dDownBlock, self).__init__()

        if pool_l == 'm':
            self.pool = nn.MaxPool3d(kernel_size=2)
        elif pool_l == 'c2':  # using conv to down sample, and the in_ch is not actual input channel
            self.pool = _combine_layers(in_ch, in_ch, ['c', norm_l, act_l], padding=1, kernel_size=3, stride=2)
        elif pool_l is None:
            self.pool = PassLayer()  # used in input block

        # MultiResBlock
        self.block1 = MultiResBlock(in_ch, out_ch, norm_l=norm_l, act_l=act_l, is_act_first=is_act_first)

    def forward(self, x):
        out = self.pool(x)
        out = self.block1(out)

        return out


class MultiResU3dUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, res_path_length,
                 norm_l='b', act_l='l', is_norm_act_concate=False, is_act_first=True):
        super(MultiResU3dUpBlock, self).__init__()
        self.up_conv = _combine_layers(in_ch, out_ch, 't', padding=0, kernel_size=2, stride=2, bias=False)

        if is_norm_act_concate:
            self.norm_act = _combine_layers(out_ch*2, out_ch*2, [norm_l, act_l], padding=0, kernel_size=2, stride=2)
        else:
            self.norm_act = PassLayer()

        # res path layer
        self.res_path = ResPathBlock(out_ch, out_ch, length=res_path_length, norm_l=norm_l, act_l=act_l,
                                     is_act_first=is_act_first)

        # cbe is Conv + BN + LeakyReLU;
        self.block1 = MultiResBlock(out_ch*2, out_ch, norm_l=norm_l, act_l=act_l, is_act_first=is_act_first)

    def forward(self, x, x_pass):
        # channel of x_pass must same as out channel
        # x1 = self.up_conv(x)
        # x_pass = self.res_path(x_pass)

        # out = torch.cat((x1, x_pass), dim=1)  # new channel: out_ch + x_pass channel
        out = torch.cat((self.up_conv(x), self.res_path(x_pass)), dim=1)  # reduce gpu memory

        out = self.norm_act(out)

        out = self.block1(out)

        return out


# --------------------------------------------------------------------------------------------------------------------
# Similar to '_combine_layers', but can input more kwargs
def combine_3d_layers(in_ch, out_ch, layer_name_list, conv_kwargs=None,
                      norm_kwargs=None, act_kwargs=None, drop_kwargs=None, liner_kwargs=None):
    # set default flag
    # conv_kw_None_flag = False
    # linear_kw_None_flag = False
    norm_kw_None_flag = False
    act_kw_None_flag = False
    # drop_kw_None_flag = False

    # set default args
    if conv_kwargs is None:
        # conv_kw_None_flag = True
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
    if norm_kwargs is None:
        norm_kw_None_flag = True
        norm_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
    if act_kwargs is None:
        act_kw_None_flag = True
        act_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    if drop_kwargs is None:
        # drop_kw_None_flag = True
        drop_kwargs = {'p': 0, 'inplace': True}
    if liner_kwargs is None:
        # linear_kw_None_flag = True
        liner_kwargs = dict()

    # get layer list
    layer_list = []
    for layer_name in layer_name_list:
        # Conv
        if layer_name == 'c':
            layer_list.append(nn.Conv3d(in_ch, out_ch, **conv_kwargs))
        elif layer_name == 't':
            layer_list.append(nn.ConvTranspose3d(in_ch, out_ch, **conv_kwargs))
        # Linear
        elif layer_name == 'L':
            layer_list.append(nn.Linear(in_ch, out_ch, **liner_kwargs))
        # Normalise
        elif layer_name == 'b':
            layer_list.append(nn.BatchNorm3d(out_ch, **norm_kwargs))
        elif layer_name == 'i':
            layer_list.append(nn.InstanceNorm3d(out_ch, **norm_kwargs))
        elif layer_name == 'g':
            if norm_kw_None_flag:
                norm_kwargs = {'num_groups': math.ceil(out_ch / 16), 'num_channels': out_ch}
            layer_list.append(nn.GroupNorm(**norm_kwargs))  # must input num_groups and num_channels
        # Active
        elif layer_name == 'l':
            layer_list.append(nn.LeakyReLU(**act_kwargs))
        elif layer_name == 'r':
            layer_list.append(nn.ReLU(**act_kwargs))
        elif layer_name == 'e':
            layer_list.append(nn.ELU(**act_kwargs))
        elif layer_name == 'p':
            if act_kw_None_flag:
                act_kwargs = {'num_parameters': out_ch}
            layer_list.append(nn.PReLU(**act_kwargs))  # must input num_parameters
        elif layer_name == 's':
            layer_list.append(nn.Sigmoid())  # no args
        elif layer_name == 'ge':
            layer_list.append(nn.GELU())  # no args
        # drop out
        elif layer_name == 'd':
            if drop_kwargs['p'] != 0:  # if p=0, the drop is not actually used, so skip it
                layer_list.append(nn.Dropout3d(**drop_kwargs))
            else:
                pass
        # skip
        elif layer_name is None or layer_name == '':
            pass  # skip this layer name
        # error
        else:
            raise RuntimeError('Create Combine layers Error! Unsupported layer name: {}!'.format(layer_name))

    return nn.Sequential(*layer_list)


# Similar the U3dDownBlock, but can input more args
class U3dEncodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l', drop_l='', conv_num=2,
                 pool_kwargs=None, conv_kwargs=None, norm_kwargs=None, act_kwargs=None, drop_kwargs=None):
        # pool_l must in ['m', 'c', 'c3', None], 'm' is max_pool
        # 'c' will using the first conv to pool, but 'c3' will add an extra conv layer for pool
        super(U3dEncodeBlock, self).__init__()

        # get pool layer
        if pool_l == 'm':
            self.pool = nn.MaxPool3d(pool_kwargs)
        elif pool_l == 'c':
            pool_conv_kwargs = deepcopy(conv_kwargs)
            pool_conv_kwargs['stride'] = pool_kwargs['kernel_size']
            self.pool = combine_3d_layers(in_ch, out_ch, ['c', drop_l, norm_l, act_l],
                                          pool_conv_kwargs, norm_kwargs, act_kwargs, drop_kwargs)
            conv_num -= 1  # the conv pool will occupy 1 conv numbers
            in_ch = out_ch  # input channel has been changed
        elif pool_l is None or pool_l == '':
            self.pool = PassLayer()  # used in input block
        else:
            raise RuntimeError('Unsupport pool name: {}'.format(pool_l))

        # get conv layer
        conv_list = []
        for i in range(conv_num):
            conv_list.append(combine_3d_layers(in_ch, out_ch, ['c', drop_l, norm_l, act_l],
                                               conv_kwargs, norm_kwargs, act_kwargs, drop_kwargs))
            in_ch = out_ch  # new input channel has been changed out_channel

        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


# Similar the U3dUpBlock, but can input more args
class U3dDecodeBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, up_l='t', norm_l='i', act_l='l', drop_l='', conv_num=2,
                 up_pool_kwargs=None, conv_kwargs=None, norm_kwargs=None, act_kwargs=None, drop_kwargs=None,):
        # up_pool_kwargs will be used to upsample, which is same as pool_kwargs in EncodeBlock
        super(U3dDecodeBlock, self).__init__()

        # get upsample layer
        if up_l == 't':
            up_conv_kwargs = {'kernel_size': up_pool_kwargs['kernel_size'],
                              'stride': up_pool_kwargs['kernel_size'],
                              'bias': False}
            self.up_conv = combine_3d_layers(in_ch, skip_ch, [up_l, drop_l, norm_l, act_l],
                                             up_conv_kwargs, norm_kwargs, act_kwargs, drop_kwargs)
        else:
            raise RuntimeError('Unsupport upsample name: {}'.format(up_l))

        # get conv layer
        conv_list = []
        for i in range(conv_num):
            if i == 0:
                in_ch = skip_ch * 2  # first input will stack the pass and up-sample image
            conv_list.append(combine_3d_layers(in_ch, out_ch, ['c', drop_l, norm_l, act_l],
                                               conv_kwargs, norm_kwargs, act_kwargs, drop_kwargs))
            in_ch = out_ch  # new input channel has been changed out_channel

        self.conv = nn.Sequential(*conv_list)

    def forward(self, x, x_pass):
        x = self.up_conv(x)
        x = torch.cat((x, x_pass), dim=1)  # new channel: out_ch + x_pass channel
        x = self.conv(x)

        return x
