import torch
import torch.nn as nn
import torch.nn.functional as F
from dl_brain.dl_run.model_multi_modality.model.mm_modules import SelfAttentionMM


# Copy from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/model_files/sfcn.py
# has some modify
class SFCN(nn.Module):
    def __init__(self, input_ch=1, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, drop_p=0.5):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = input_ch
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        # avg_shape = [5, 6, 5]
        # self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        
        # In raw paper, the input is 160*192*160, and the output in there os [5, 6, 5]
        # So, this average pool with kernel [5, 6, 5] was same as global average pool
        # using nn.AdaptiveAvgPool3d([1, 1, 1]) as Global Average Pool
        # self.classifier.add_module('global_average_pool', nn.AdaptiveAvgPool3d([1, 1, 1]))
        self.classifier.add_module('global_average_pool', nn.AdaptiveAvgPool3d(1))
        # setting dropout can be modified
        # if dropout:
        if drop_p > 0:
            self.classifier.add_module('dropout', nn.Dropout(drop_p))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        # out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = torch.squeeze(x, dim=(2,3,4))
        # I'm using CrossEntropyLoss, which will applying softmax automatically
        # x = F.log_softmax(x, dim=1)
        # out.append(x)
        return x


def SFCN_adjust_model_input_and_output_layer(net, in_c, out_c):
    """
    will replace the number of last FC layer to classes
    Args:
        net: model needed to be adjust
        in_c (int): needed input channels
        out_c (int): needed output channels
        drop_p (float): needed drop probability, 0~1
    Returns:
        nn.model: input models with modified input/output number
    """
    if net.feature_extractor.conv_0[0].in_channels != in_c:
        # l_out_c = net.feature_extractor.conv_0[0].out_channels
        net.feature_extractor.conv_0[0] = nn.Conv3d(
            in_channels=in_c,
            out_channels=net.feature_extractor.conv_0[0].out_channels,
            padding=net.feature_extractor.conv_0[0].padding,
            kernel_size=net.feature_extractor.conv_0[0].kernel_size
            )
        net.feature_extractor.conv_0[0].reset_parameters()
    
    # reset output channels
    if net.classifier[2].out_channels != out_c:
        net.classifier[2] = nn.Conv3d(
            in_channels=net.classifier[2].in_channels,
            out_channels=out_c,
            padding=net.classifier[2].padding,
            kernel_size=net.classifier[2].kernel_size
            )
        net.classifier[2].reset_parameters()
        
    return net

# To correspond to multi-modal model (CrossAtt)
# has some modify
class SFCN_SelfAttention(nn.Module):
    def __init__(self, classes, 
                 cnn_input_ch,
                 cnn_channel_number=[32, 64, 128, 256, 256, 64], cnn_drop_p=0.5,
                 att_embed_dim=256, att_num_heads=4, att_dropout=0.0,
                 fin_drop_p=0.0,
                 ):
        super(SFCN_SelfAttention, self).__init__()
        # CNN modules
        n_layer = len(cnn_channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = cnn_input_ch
            else:
                in_channel = cnn_channel_number[i-1]
            out_channel = cnn_channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        # avg_shape = [5, 6, 5]
        # self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        
        # In raw paper, the input is 160*192*160, and the output in there os [5, 6, 5]
        # So, this average pool with kernel [5, 6, 5] was same as global average pool
        # using nn.AdaptiveAvgPool3d([1, 1, 1]) as Global Average Pool
        # self.classifier.add_module('global_average_pool', nn.AdaptiveAvgPool3d([1, 1, 1]))
        self.classifier.add_module('global_average_pool', nn.AdaptiveAvgPool3d(1))
        # setting dropout can be modified
        # if dropout:
        if cnn_drop_p > 0:
            self.classifier.add_module('dropout', nn.Dropout(cnn_drop_p))
        # i = n_layer
        # in_channel = cnn_channel_number[-1]
        # out_channel = cnn_output_dim
        # self.classifier.add_module('conv_%d' % i,
        #                            nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        
        # attention --------------------------------------------------------------------------------
        self.cnn_att_linear = nn.Linear(cnn_channel_number[-1], att_embed_dim)
        
        # self attention
        self.cnn_self_att = SelfAttentionMM(embed_dim=att_embed_dim, num_heads=att_num_heads, dropout=att_dropout)
        
        # fin last drop layer 
        self.fin_drop_layer = nn.Dropout(p=fin_drop_p)
            
        # linear to get predict results
        self.out_linear = nn.Linear(att_embed_dim, classes)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer
        
    def forward(self, cnn_x):
        # SFCN ----------------------------------------------------------
        cnn_x = self.feature_extractor(cnn_x)
        cnn_x = self.classifier(cnn_x)
        cnn_x = torch.squeeze(cnn_x, dim=(2,3,4))  # [b, c, x, y, z] -> [b, c]
        
        # attention -----------------------------------------------------
        # self attention
        cnn_x = self.cnn_att_linear(cnn_x)
        cnn_x = self.cnn_self_att(cnn_x)

        # drop
        cnn_x = self.fin_drop_layer(cnn_x)
        
        # out linear, predict --------------------------------------------
        cnn_x = self.out_linear(cnn_x)

        return cnn_x

