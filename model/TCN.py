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
    #     self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0.05, 0.01)
    #     self.conv2.weight.data.normal_(0.05, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0.05, 0.01)

    def forward(self, x):
        # print(self.conv1.weight.data.device)

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        # self.layers = nn.ModuleList(self.layers)
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

        in_conv = x
        # out = torch.zeros(self.num_levels-1)
        for i in range(self.num_levels):
            # print(in_conv.shape)
            # print(in_conv[0,:,0])
            # print(self.layers[i].weight.data)
            out_conv = self.layers[i](in_conv)
            in_conv = out_conv
            # print(out_conv[0,:,0])

            # if i!=self.num_levels-1:
            #     out[self.num_levels-2-i] = out_conv[-1]

        # out = torch.cat((out_conv, out))
        return out_conv