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
from sbi.neural_nets.embedding_nets import FCEmbedding, calculate_filter_output_size

def get_new_cnn_output_size_double(
    input_shape: Tuple,
    conv_layer_1: Union[nn.Conv1d, nn.Conv2d],
    conv_layer_2: Union[nn.Conv1d, nn.Conv2d],
    pool: Union[nn.MaxPool1d, nn.MaxPool2d],
        ) -> Union[Tuple[int], Tuple[int, int]]:
    """Returns new output size after applying a given convolution and pooling.
    Args:
        input_shape: tup.
        conv_layer: applied convolutional layers
        pool: applied pooling layer
    Returns:
        new output dimension of the cnn layer.
    """
    assert isinstance(input_shape, Tuple), "input shape must be Tuple."
    assert 0 < len(input_shape) < 3, "input shape must be 1 or 2d."
    assert isinstance(conv_layer_1.padding, Tuple), "conv layer 1 attributes must be Tuple."
    assert isinstance(conv_layer_2.padding, Tuple), "conv layer 2 attributes must be Tuple."

    assert isinstance(pool.padding, int), "pool layer attributes must be integers."

    out_after_conv_1 = [
        calculate_filter_output_size(
            input_shape[i],
            conv_layer_1.padding[i],
            conv_layer_1.dilation[i],
            conv_layer_1.kernel_size[i],
            conv_layer_1.stride[i],
        )
        for i in range(len(input_shape))
    ]
    
    out_after_conv_2 = [
        calculate_filter_output_size(
            out_after_conv_1[i],
            conv_layer_2.padding[i],
            conv_layer_2.dilation[i],
            conv_layer_2.kernel_size[i],
            conv_layer_2.stride[i],
        )
        for i in range(len(input_shape))
    ]
    
    out_after_pool = [
        calculate_filter_output_size(
            out_after_conv_2[i],
            pool.padding,
            1,
            pool.kernel_size,
            pool.stride,
        )
        for i in range(len(input_shape))
    ]
    return tuple(out_after_pool)

def get_new_cnn_output_size_single(
    input_shape: Tuple,
    conv_layer_1: Union[nn.Conv1d, nn.Conv2d],
    pool: Union[nn.MaxPool1d, nn.MaxPool2d],
        ) -> Union[Tuple[int], Tuple[int, int]]:
    """Returns new output size after applying a given convolution and pooling.
    Args:
        input_shape: tup.
        conv_layer: applied convolutional layers
        pool: applied pooling layer
    Returns:
        new output dimension of the cnn layer.
    """
    assert isinstance(input_shape, Tuple), "input shape must be Tuple."
    assert 0 < len(input_shape) < 3, "input shape must be 1 or 2d."
    assert isinstance(conv_layer_1.padding, Tuple), "conv layer 1 attributes must be Tuple."

    assert isinstance(pool.padding, int), "pool layer attributes must be integers."

    out_after_conv_1 = [
        calculate_filter_output_size(
            input_shape[i],
            conv_layer_1.padding[i],
            conv_layer_1.dilation[i],
            conv_layer_1.kernel_size[i],
            conv_layer_1.stride[i],
        )
        for i in range(len(input_shape))
    ]
    
    out_after_pool = [
        calculate_filter_output_size(
            out_after_conv_1[i],
            pool.padding,
            1,
            pool.kernel_size,
            pool.stride,
        )
        for i in range(len(input_shape))
    ]
    return tuple(out_after_pool)

class SilkScreenCNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: List = [6, 12],
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
        num_linear_units: int = 50,
        output_dim: int = 20,
        kernel_size: Union[int,List] = 5,
        pool_kernel_size: int = 2,
    ):
        """Convolutional embedding network.
        First two layers are convolutional, followed by fully connected layers.
        Automatically infers whether to apply 1D or 2D convolution depending on
        input_shape.
        Allows usage of multiple (color) channels by passing in_channels > 1.
        Args:
            input_shape: Dimensionality of input, e.g., (28,) for 1D, (28, 28) for 2D.
            in_channels: Number of image channels, default 1.
            out_channels_per_layer: Number of out convolutional out_channels for each
                layer. Must match the number of layers passed below.
            num_cnn_layers: Number of convolutional layers.
            num_linear_layers: Number fully connected layer.
            num_linear_units: Number of hidden units in fully-connected layers.
            output_dim: Number of output units of the final layer.
            kernel_size: Kernel size for both convolutional layers.
            pool_size: pool size for MaxPool1d operation after the convolutional
                layers.
        """
        super(SilkScreenCNN, self).__init__()

        assert isinstance(
            input_shape, Tuple
        ), "input_shape must be a Tuple of size 1 or 2, e.g., (width, [height])."
        assert (
            0 < len(input_shape) < 3
        ), """input_shape must be a Tuple of size 1 or 2, e.g.,
            (width, [height]). Number of input channels are passed separately"""
        assert num_linear_layers >=2
            
        use_2d_cnn = len(input_shape) == 2
        conv_module = nn.Conv2d if use_2d_cnn else nn.Conv1d
        pool_module = nn.AvgPool2d if use_2d_cnn else nn.AvgPool1d
        activ = torch.nn.GELU
        
        assert (
            len(out_channels_per_layer) == num_conv_layers
        ), "out_channels needs as many entries as num_cnn_layers."
        
        if isinstance(kernel_size,int):
            kernel_size = [kernel_size]*num_conv_layers
        else:
            assert len(kernel_size) == num_conv_layers 
            
        # define input shape with channel
        self.input_shape = (in_channels, *input_shape)
        stride = 1
        padding = 1
        
        # Construct CNN feature extractor.
        cnn_layers = []
        cnn_output_size = input_shape
        conv_layer_init =  conv_module(
                in_channels=in_channels,
                out_channels=out_channels_per_layer[0],
                kernel_size=11,
                stride=stride,
                padding=padding,
            )
        pool_init = pool_module(kernel_size=pool_kernel_size)
        cnn_layers += [conv_layer_init, activ(), pool_init]
        
        cnn_output_size  = get_new_cnn_output_size_single(cnn_output_size, conv_layer_init, pool_init)

        for ii in range(num_conv_layers):
            # Defining another 2D convolution layer
            conv_layer_1 = conv_module(
                in_channels=out_channels_per_layer[0] if ii == 0 else out_channels_per_layer[ii - 1],
                out_channels=out_channels_per_layer[ii],
                kernel_size=kernel_size[ii],
                stride=stride,
                padding=padding,
            )
            conv_layer_2 = conv_module(
                in_channels=out_channels_per_layer[ii],
                out_channels=out_channels_per_layer[ii],
                kernel_size=kernel_size[ii],
                stride=stride,
                padding=padding,
            )
            pool = pool_module(kernel_size=pool_kernel_size)
            cnn_layers += [conv_layer_1, activ(),conv_layer_2, activ(), pool]
            # Calculate change of output size of each CNN layer
            cnn_output_size = get_new_cnn_output_size_double(cnn_output_size, conv_layer_1,conv_layer_2, pool)

        self.cnn_subnet = nn.Sequential(*cnn_layers)
        
        # Construct linear post processing net.
        drop_p_lin = 1./3.
        linear_layers = []
        num_cnn_output = out_channels_per_layer[-1] * torch.prod(torch.tensor(cnn_output_size))

        for i in range(num_linear_layers-1):
            in_features = num_cnn_output if i==0 else num_linear_units
            lin_layer = nn.Linear(in_features, num_linear_units)
            linear_layers += [lin_layer, activ(), nn.Dropout(p = drop_p_lin)]
        
        linear_layers += [nn.Linear(num_linear_units, output_dim),] # final layer
        self.linear_subnet = nn.Sequential(*linear_layers)

    # Defining the forward pass
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # reshape to account for single channel data.
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))
        # flatten for linear layers.
        x = x.view(batch_size, -1)
        x = self.linear_subnet(x)
        return x


def build_default_NN(
        num_filter: int,
        num_summary: Optional[int] = 64):
    """_summary_

    Parameters
    ----------
    num_filters : int
        Number of filters
    """
    embedding_net = silkscreen_resnet(num_summary=num_summary, num_filter=num_filter)
    flow_kwargs = {'z_score_theta':'independent', 'z_score_x':'structured', 'hidden_features': 50, 'num_transforms':5, 'num_bins':8, 'dropout':0.2}
    
    posterior_nn = sbi_utils.posterior_nn('nsf', embedding_net=embedding_net, **flow_kwargs )
    return posterior_nn

### Maybe get rid of all this stuff:
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

