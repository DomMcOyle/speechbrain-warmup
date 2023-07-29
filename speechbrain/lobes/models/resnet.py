"""
ResNet-based model for audio classification.
The implementation is based on the original architecture of the original paper
(Deep Resitual Learning for Image Recognition https://arxiv.org/abs/1512.03385)

Authors
 * Domenico Dell'Olio 2023
 * Mirco Ravanelli 2020
"""


import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import AdaptivePool, Pooling1d
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class ResNet(torch.nn.Module):
    """ResNet-based model extracting features for audio classification

    Arguments
    ---------
    activation : torch class
        A class for constructing the activation layers.
    stem_params : list of three ints
        A list containing the parameters for the initial stem layer, in order:
        kernel size, stride and output channels. If None, no stem layer is applied on
        the input.
    stem_pooling_params: A list containing the parameters for the pooling in the stem layer, in order:
        kernel size, stride and padding. If None, no stem layer is applied on
        the input.
    resnet_stages : int
        Number of stages to add to the network.
        Each stage can have different number of blocks and channels
    block_per_stage : list of ints
        number of ResNet blocks (conv+BN+act.+conv+BN+act.) to add to each stage
    stages_channels : list of ints
        number of channels for the activation maps for each stage
    bottleneck_reduction : int
        reduction ratio of channels for bottleneck blocks, e.g. 4 means the number of channels
        are reduced by a factor of 4 in the bottleneck. 1 means no bottleneck.
    in_channels : int
        Number of input channels

    Example
    -------
    >>> compute_resnet_features = ResNet()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = _resnet_features(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.ReLU,
        stem_params=[7, 2, 64],
        stem_pooling_params=[3,2,1],
        resnet_stages=4,
        block_per_stage=[2,2,2,2],
        stages_channels=[64, 128, 256, 512],
        bottleneck_reduction=1,
        in_channels=40,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()
        
        assert len(stages_channels)==resnet_stages
        assert len(block_per_stage)==resnet_stages
        
        if stem_params is not None and stem_pooling_params is not None:
            # if the parameters for the stem are provided, then it is added
            # to the network
            self.blocks.extend([Conv1d(in_channels=in_channels,
                                       kernel_size=stem_params[0],
                                       stride=stem_params[1],
                                       out_channels=stem_params[2]),
                                BatchNorm1d(input_size=stem_params[2]),
                                activation(),
                                Pooling1d('max', kernel_size=stem_pooling_params[0],
                                          stride=stem_pooling_params[1],
                                          padding=stem_pooling_params[2])
                                ])

            in_channels = stem_params[2]

        # Resnet is composed of different stages, each of which of different blocks
        # composed of conv+BN+activation+conv+BN+(residual)+activation
        # We here loop over all the stages and blocks in order to add them.
        for stage_index in range(resnet_stages):
            for block_index in range(block_per_stage[stage_index]):
                out_channels = stages_channels[stage_index]
                self.blocks.extend(
                    [
                        ResNetBlock(device, activation, out_channels, bottleneck_reduction, in_channels)
                    ]
                )
                in_channels = stages_channels[stage_index]

        # AdaptivePooling. Computes the average
        self.blocks.append(AdaptivePool(1))

    def forward(self, x):
        """Returns the x-vectors.

        Arguments
        ---------
        x : torch.Tensor
        """

        for layer in self.blocks:
            x = layer(x)
        return x

class ResNetBlock(torch.nn.Module):
    """Function creating a single ResNet Block for the ResNet Network

        Arguments
        ---------
        activation : torch class
            A class for constructing the activation layers.
        out_channels : int
            number of channels for the output of the block
        bottleneck_reduction : int
            reduction ratio of channels for bottleneck blocks, e.g. 4 means the number of channels
            are reduced by a factor of 4 in the bottleneck. 1 means no bottleneck.
        in_channels : int
            Number of input channels
        """

    def __init__(
            self,
            device="cpu",
            activation=torch.nn.ReLU,
            out_channels=4,
            bottleneck_reduction=1,
            in_channels=40,
    ):
        super().__init__()
        # if the block inputs and outputs a different number of channel, then it requires
        # to downsample the input and apply a shortcut layer on the residual signal to even out
        # the number of channels
        if in_channels != out_channels:
            first_stride = 2
            self.shortcut = Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=2)
            self.shortcut_bn = BatchNorm1d(input_size=out_channels)
        else:
            first_stride = 1
            self.shortcut = None
            self.shortcut_bn = None

        # if there is a bottleneck, then the third convolutional layer must be added,
        # together with its batch norm
        if bottleneck_reduction == 1:
            self.conv3 = None
            self.bn3 = None
        else:
            self.conv3 = Conv1d(in_channels=out_channels//bottleneck_reduction,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1)
            self.bn3 = BatchNorm1d(input_size=out_channels)
            out_channels = out_channels//bottleneck_reduction

        # here the two main conv+bn are declared
        self.conv1 = Conv1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=first_stride)
        self.bn1 = BatchNorm1d(input_size=out_channels)
        self.conv2 = Conv1d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1)
        self.bn2 = BatchNorm1d(input_size=out_channels)
        self.activation = activation()

    def forward(self, x):
        """implements the forward behaviour of the ResNet Block

        Arguments
        ---------
        x : torch.Tensor
        """
        h = self.conv1(x) # TODO: to test
        h = self.bn1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.bn2(h)
        if self.conv3 is not None:
            h = self.activation(h)
            h = self.conv3(h)
            h = self.bn3(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
            x = self.shortcut_bn(x)
        return self.activation(x+h)



class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of ResNet features.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_resnet = ResNet()
    >>> resnet = compute_resnet(input_feats)
    >>> classify = Classifier(input_shape=resnet.shape)
    >>> output = classify(resnet)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.ReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):
        super().__init__(input_shape=input_shape)

        #self.append(activation(), layer_name="act")
        #self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            #self.DNN[block_name].append(
            #    sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            # )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )
