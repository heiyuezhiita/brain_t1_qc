import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *


def ResNet_adjust_model_input_and_output_layer(net, in_c, out_c, drop_p=0):
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
    if net.conv1.in_channels != in_c:
        net.conv1 = nn.Conv2d(in_c, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
    
    # get fc input channel of pretrained model
    if isinstance(net.fc, nn.Linear):
        fc_in_c = net.fc.in_features
    elif isinstance(net.fc, nn.Sequential):
        # net.fc is [nn.Dropout, nn.Linear]
        fc_in_c = net.fc[1].in_features
    else:
        msg = "net.fc must be 'nn.Linear' or 'nn.Sequential', but not '{}'".format(type(net.fc))
        RuntimeError(msg)
        
    # pertrained fc will be removed
    if drop_p > 0:
        net.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=drop_p),
            nn.Linear(fc_in_c, out_c)
        )
    else:
        net.fc = nn.Linear(fc_in_c, out_c)
    
    return net


def get_transfer_resnet18(in_c, classes, is_pretrained=True, drop_p=0):
    """
    will replace the number of last FC layer to classes
    Args:
        classes (int): output number of resnet18 will be replaced to this
        pretrained (bool, optional): is using pretrained resnet from internet. Defaults to True.

    Returns:
        nn.model: resnet18 with modified output classes number
    """
    net = models.resnet18(pretrained=is_pretrained)
    
    net = ResNet_adjust_model_input_and_output_layer(net, in_c, classes, drop_p=drop_p)
    
    return net


def get_transfer_resnet34(in_c, classes, is_pretrained=True, drop_p=0):
    """
    will replace the number of last FC layer to classes
    Args:
        classes (int): output number of resnet18 will be replaced to this
        pretrained (bool, optional): is using pretrained resnet from internet. Defaults to True.

    Returns:
        nn.model: resnet18 with modified output classes number
    """
    net = models.resnet34(pretrained=is_pretrained)
    
    net = ResNet_adjust_model_input_and_output_layer(net, in_c, classes, drop_p=drop_p)

    return net
