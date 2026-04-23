import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=2, activation='lrelu', norm='BN',
                                 pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=2, activation='lrelu', norm='BN',
                                 pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=32, stride=2, activation='lrelu', norm='BN',
                                 pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x1 = self.pool1(self.conv1(x))
        x2 = self.pool2(self.conv2(x1))
        f_map = self.pool3(self.conv3(x2))
        f_vec = self.flatten(f_map)
        return f_map, f_vec

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=576, out_features=100)   #1600   fft576
        self.linear2 = nn.Linear(in_features=100, out_features=3)
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(1, (64, 1))
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.deconv1 = Conv1dBlock(in_chan=32 * 2, out_chan=32, kernel_size=31, stride=1, activation='lrelu', norm='BN',
                                   pad_type='reflect', padding=35)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.deconv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=63, stride=1, activation='lrelu', norm='BN',
                                   pad_type='reflect', padding=151)
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.deconv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=127, stride=1, activation='lrelu', norm='BN',
                                   pad_type='reflect', padding=695)
        self.deconv7 = Conv1dBlock(in_chan=32, out_chan=1, kernel_size=127, stride=1, activation='sigmoid', norm='BN',
                                 pad_type='reflect', padding=63)

    def forward(self, f_map):
        x1 = self.deconv1(self.up1(f_map))
        x2 = self.deconv2(self.up2(x1))
        x3 = self.deconv3(self.up3(x2))
        x_rec =  self.deconv7(x3)
        return x_rec




class Conv1dBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride, activation='lrelu', norm='LN', pad_type='reflect',
                 padding=0):
        super(Conv1dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad1d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad1d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ConstantPad1d(padding, 0.0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        norm_dim = out_chan
        if norm == 'bn' or norm == 'BN':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in' or norm == 'IN':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln' or norm == 'LN':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm is None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none' or activation is None:
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, stride, bias=self.use_bias, padding='valid')
        nn.init.kaiming_normal_(self.conv.weight.data, nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor':  # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x



