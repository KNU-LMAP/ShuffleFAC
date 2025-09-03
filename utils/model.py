import torch.nn as nn
import torch
import yaml
import torchaudio
"""
model name : FAC(Frequency Aware Convolution)
Author : Dongjun Kim, DongHyeon Lee

"""
class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class Positional_Encoding(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.freq_bins = freq_bins
        freq = torch.arange(self.freq_bins)
        pos_freq = torch.sin((torch.pi/2) * (freq / self.freq_bins))
        self.register_buffer('pos_freq', pos_freq.view(1,1,-1,1))
    def forward(self, x):
        return self.pos_freq


class self_attention(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.fcn = nn.Linear(in_features=freq_bins, out_features=1)
        self.activ = nn.Sigmoid()
    def forward(self, x):
        x = torch.mean(x,dim=2)
        x = self.fcn(x)
        x = self.activ(x)
        x = x.squeeze(-1)
        return x

class FAC(nn.Module):
    def __init__(self,freq_bins):
        super().__init__()
        self.attn = self_attention(freq_bins=freq_bins)
        self.pos = Positional_Encoding(freq_bins=freq_bins)
    def forward(self, x):
        attn_x = self.attn(x)
        attn_x = attn_x.view(attn_x.size(0), attn_x.size(1), 1, 1)
        pos_x = self.pos(x)
        x = (attn_x + pos_x).transpose(2,3) + x
        return x

class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="cg",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        fac_layers=[1, 1, 1, 1, 1, 1, 1], 
        freq_bins=[128,64,32,16,8,4,2,1],
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]

            if fac_layers[i]==1:
                cnn.add_module(f"fac_conv{i}", FAC(freq_bins=freq_bins[i])),
                cnn.add_module(f"conv{i}", 
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
                )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x

class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 specaugm_t_p=0.2,
                 specaugm_t_l=5,
                 specaugm_f_p=0.2,
                 specaugm_f_l=10,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.specaugm_t_p = specaugm_t_p
        self.specaugm_t_l = specaugm_t_l
        self.specaugm_f_p = specaugm_f_p
        self.specaugm_f_l = specaugm_f_l

        self.cnn = CNN(n_in_channel=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def apply_specaugment(self, x):
        timemask = torchaudio.transforms.TimeMasking(
            self.specaugm_t_l, True, self.specaugm_t_p
        )
        freqmask = torchaudio.transforms.TimeMasking(
            self.specaugm_f_l, True, self.specaugm_f_p
        )  # use time masking also here
        x = timemask(freqmask(x.transpose(1, -1)).transpose(1, -1))
        return x

    def forward(self, x): 
        x = x.transpose(2, 3)
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frame, ch*freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        #rnn
        # x = self.rnn(x) #x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        # CrossEntropyLoss를 사용하므로 시그모이드를 제거하고 로짓을 그대로 사용합니다.
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return weak

if __name__=="__main__":
    path = "Your Yaml file path"
    with open(path, "r") as f:
        configs = yaml.safe_load(f)
    x = torch.randn([24,1,626,128])
    net = CRNN(**configs['CRNN'])
    print(net(x).shape)