### Main code taking from torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
### modified slightly to allow for different number of filters for input image numbers
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torchvision.models import resnet
from sbi import utils as sbi_utils


def build_default_NN(
        num_filter: int,
        num_summary: Optional[int] = 16):
    """Return the default posterior NN used in the initial silkscreen paper

    Parameters
    ----------
    num_filters : int
        Number of filters
    num_sumamry : int
        Number of learned summary statistics
    """
    embedding_net = silkscreen_resnet(num_summary=num_summary, num_filter=num_filter)
    flow_kwargs = {'z_score_theta':'independent', 'z_score_x':'structured', 'hidden_features': 50, 'num_transforms':5, 'num_bins':8, 'dropout':0.2}
    
    posterior_nn = sbi_utils.posterior_nn('nsf', embedding_net=embedding_net, **flow_kwargs )
    return posterior_nn

class silkscreen_resnet(resnet.ResNet):
    def __init__(
        self,
        num_summary: int = 32,
        num_filter: int = 3,
    ) -> None:
        super().__init__(block = resnet.BasicBlock, layers=[0,0,0,0])#dummy init
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.block = resnet.Bottleneck
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        layers = [2,2,2,2]
        self.groups = 1
        self.base_width = 32
        self.inplanes = 16

        self.conv1 = nn.Conv2d(num_filter, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 16, layers[0])
        self.layer2 = self._make_layer(self.block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128*self.block.expansion, num_summary)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

