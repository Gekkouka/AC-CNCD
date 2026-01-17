import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, m):
        ctx.m = m
        return x.view_as(x)  # 返回与输入 x 相同的视图

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.m  # 反向梯度乘以 m
        return output, None  # 返回反向梯度和 None（因为 m 不是可学习的参数）

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """
        self.w_linear = nn.Parameter(torch.randn(d_core, d_core))
        self.u_linear = nn.Parameter(torch.randn(d_core))

        self.gen1 = nn.Linear(d_series, d_core)
        self.gen2 = nn.Linear(d_series + d_core, d_series)

    def forward(self, input):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = self.gen1(input)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            aug_combined_mean = torch.gather(combined_mean, 1, indices)
            aug_combined_mean = aug_combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            aug_combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # attention combined_mean
        combined_mean_shape = combined_mean.shape
        combined_mean_reshape = torch.reshape(combined_mean, [-1, combined_mean.shape[-1]])
        attn_softmax = F.softmax(torch.mm(combined_mean_reshape, self.w_linear) + self.u_linear, 1)
        aug_combined_mean = torch.reshape(aug_combined_mean, [-1, aug_combined_mean.shape[-1]])
        combined_mean = torch.mul(attn_softmax, aug_combined_mean)
        combined_mean = torch.reshape(combined_mean, combined_mean_shape)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        output = self.gen2(combined_mean_cat)

        return output

# The ABP module
class Attention(nn.Module):
    def __init__(self, cuda, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res

class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=2, location=-1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location
    def forward(self, x):
        # self.lstm.flatten_parameters()
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn

class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2):
        super(Encoder, self).__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)
    def forward(self, x):
        x_h = self.theta(x)
        return x_h

class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2,output_dim=310):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell, time_steps):
        out =[]
        out1 = self.fc_out(input)
        out.append(out1)
        out1= out1.unsqueeze(0)  # input = [batch size] to [1, batch size]
        for i in range(time_steps-1):
            output, (hidden, cell) = self.rnn(out1,
                                              (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]
            out_cur = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
            out.append(out_cur)
            out1 = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1,0) #batch first
        return out, hidden, cell

'''
#namely The Subject Classifier SD
class DomainClassifier(nn.Module):
    def __init__(self, input_dim =64, output_dim=14):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.classifier(x)
        return x
'''


class LaplaceDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(LaplaceDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

'''
class LaplaceDiscriminator(nn.Module):
    def __init__(self, input_dim=64):
        super(LaplaceDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),                      # 使用 LeakyReLU 替代 Softplus
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),                                             # 使用 ReLU 激活函数
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),                 # 最后一层，输出维度为 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
'''

class Taskout(nn.Module):
    def __init__(self, hidden_layer=64, n_class=3):
        super(Taskout, self).__init__()
        # self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer,n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MMD_AAE(nn.Module):
    def __init__(self, nClass, input_dim=310, hid_dim=64, n_layers=1, d_output_dim=310):
        super(MMD_AAE,self).__init__()
        self.E = Encoder(input_dim, hid_dim, n_layers)
        self.D = Decoder(input_dim, hid_dim, n_layers, d_output_dim)
        self.T = Taskout(hid_dim, nClass)
        return

    def forward(self, x, time_steps):
        e, hidden, cell = self.E(x)
        d, _, _  = self.D(e, hidden, cell, time_steps)
        t = self.T(e)
        return e,d,t