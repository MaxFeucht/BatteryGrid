import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    """
    TemporalBlock is a module that represents a single temporal block in a Temporal Convolutional Network (TCN).
    It consists of two convolutional layers with residual connections and dropout.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolutional operation.
        dilation (int): Dilation rate of the convolutional operation.
        padding (int): Padding size for the convolutional operation.
        dropout (float, optional): Dropout probability. Default is 0.2.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        """
        Initializes the weights of the convolutional layers.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        """
        Performs forward pass through the temporal block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    
    

class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block. Applies a depthwise convolution followed by a pointwise convolution to reduce the number of parameters.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride value for the convolution.
        dilation (int): Dilation value for the convolution.
        padding (int): Padding value for the convolution.
        dropout (float, optional): Dropout probability. Default is 0.2.
        downsample (bool, optional): Whether to apply downsampling. Default is False.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, downsample=False):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_conv1 = weight_norm(nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, groups=n_inputs))
        self.pointwise_conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.depthwise_conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, groups=n_outputs))
        self.pointwise_conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 1))
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.depthwise_conv1, self.pointwise_conv1, self.chomp1, self.elu1, self.dropout1,
                                 self.depthwise_conv2, self.pointwise_conv2, self.chomp2, self.elu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None        
        self.elu = nn.ELU()
        self.init_weights()


    def init_weights(self):
        """
        Initializes the weights of the convolutional layers.
        """
        self.depthwise_conv1.weight.data.normal_(0, 0.01)
        self.pointwise_conv1.weight.data.normal_(0, 0.01)
        self.depthwise_conv2.weight.data.normal_(0, 0.01)
        self.pointwise_conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass of the Depthwise Separable Convolution Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.elu(out + res)




class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) # num_channels is a list of the number of channels for each layer 
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)



class TCN(nn.Module):
    def __init__(self, seq_len, num_inputs, num_channels, out_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(seq_len*num_channels[-1], out_channels)

    def forward(self, x):
        tcn_output = self.tcn(x).flatten(start_dim=1) #Flatten over the features and timestep dimensions, preserve batch dimension (dim=0)
        return self.dense(self.dropout(tcn_output))
    
