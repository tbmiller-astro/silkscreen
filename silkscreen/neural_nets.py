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
        activ = partial(nn.ReLU, inplace = True)
        
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
        drop_p = 1./3.
        linear_layers = []
        num_cnn_output = out_channels_per_layer[-1] * torch.prod(torch.tensor(cnn_output_size))

        for i in range(num_linear_layers-1):
            in_features = num_cnn_output if i==0 else num_linear_units
            lin_layer = nn.Linear(in_features, num_linear_units)
            linear_layers += [lin_layer, activ(), nn.Dropout(p = drop_p)]
        
        linear_layers += [nn.Linear(num_linear_units, output_dim),] # final layer
        self.linear_subnet = nn.Sequential(*linear_layers)

    # Defining the forward pass
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # reshape to account for single channel data.
        x = torch.asinh(x)
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))
        # flatten for linear layers.
        x = x.view(batch_size, -1)
        x = self.linear_subnet(x)
        return x


def build_default_NN(
        shape,
        num_summary: Optional[int] = 16):
    """_summary_

    Parameters
    ----------
    num_filters : int
        Number of filters
    """
    im_shape = shape[1:]
    num_channels = shape[0]
    embedding_net = SilkScreenCNN(
        input_shape = im_shape,
        in_channels = num_channels,
        out_channels_per_layer = [16,32],
        num_conv_layers =2,
        num_linear_layers = 3,
        num_linear_units = 1024,
        output_dim = num_summary,
        kernel_size = [5,3],
        pool_kernel_size = 3,
    )
    flow_kwargs = {'z_score_theta':'independent', 'z_score_x':'structured', 'hidden_features': 32, 'num_transforms':3, 'dropout_probability':0.0, 'num_blocks':2,'hidden_layers_spline_context':2}
    
    posterior_nn = sbi_utils.posterior_nn('maf', embedding_net=embedding_net, **flow_kwargs )
    return posterior_nn

### Maybe get rid of all this stuff:

class silkscreen_resnet(resnet.ResNet):
    def __init__(self,num_classes: int):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2],num_classes=num_classes)
    def forward(self, x):
        return self._forward_impl(torch.asinh(x))
    
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        num_filters (int) - number of inital filters
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 16,
        block_config: Tuple[int, int, int, int] = (3, 6, 12, 8),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.1,
        num_classes: int = 16,
        num_filters: int = 3,
        memory_efficient: bool = False,
        norm_asinh: bool =  True,
    ) -> None:

        super().__init__()
        self.norm_asinh = norm_asinh
        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(num_filters, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_asinh: x = torch.asinh(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
