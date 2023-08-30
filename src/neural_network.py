# -*- coding: utf-8 -*-
import sys
import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class KataGPool(nn.Module):
    def __init__(self):
        super(KataGPool, self).__init__()
        self.layer_mean = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Specify output size
        self.layer_max = nn.AdaptiveMaxPool2d(output_size=(1,1))   # Specify output size

    def forward(self, x):
        out_mean = self.layer_mean(x)
        out_max = self.layer_max(x)
        out = torch.cat((out_mean, out_max), dim=1)
        return out

class KataConvAndGPool(nn.Module):
    def __init__(self, c_in, c_out, c_gpool):
        super(KataConvAndGPool, self).__init__()
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding="same", bias=False)
        self.actg = nn.Mish(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = torch.nn.Linear(c_gpool * 2, c_out, bias=False)

        init.trunc_normal_(self.conv1r.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.conv1g.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.linear_g.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)

    def forward(self, x):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        outr = self.conv1r(out)
        outg = self.conv1g(out)
        outg = self.actg(outg)
        outg = self.gpool(outg).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)

        out = outr + outg
        return out

class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.Mish(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.Mish(inplace=True)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
            init.trunc_normal_(self.downsample_conv.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)

        init.trunc_normal_(self.conv1.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.conv2.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu2(out)
        return out


class PolicyHead(torch.nn.Module):
    def __init__(self, c_in, c_p1, c_g1):
        super(PolicyHead, self).__init__()
        self.conv1p = torch.nn.Conv2d(c_in, c_p1, kernel_size=1, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_g1, kernel_size=1, padding="same", bias=True)
        self.actg = nn.Mish(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = torch.nn.Linear(2 * c_g1, c_p1, bias=False)
        self.act2 = nn.Mish(inplace=True)
        self.conv2p = torch.nn.Conv2d(c_p1, 2, kernel_size=1, padding="same", bias=True)

        init.trunc_normal_(self.conv1p.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.conv1g.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.linear_g.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.conv2p.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)

    def forward(self, x):
        outp = self.conv1p(x)
        outg = self.conv1g(x)

        outg = self.actg(outg)
        outg = self.gpool(outg).squeeze(-1).squeeze(-1) # NC

        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1) # NCHW

        outp = outp + outg
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        outpolicy = outp

        # mask out parts outside the board by making them a huge neg number, so that they're 0 after softmax
        # outpolicy = outpolicy - (1.0 - mask) * 5000.0
        # NC(HW) concat with NC1

        return outpolicy.view(outpolicy.shape[0],outpolicy.shape[1],-1)

class ValueHead(torch.nn.Module):
    def __init__(self, c_in, c_v1, c_v2):
        super(ValueHead, self).__init__()
        self.conv1 = torch.nn.Conv2d(c_in, c_v1, kernel_size=1, padding="same", bias=True)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.act1 = nn.Mish(inplace=True)
        self.gpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear2 = torch.nn.Linear(c_v1, c_v2, bias=True)
        self.act2 = nn.Mish(inplace=True)

        self.linear_valuehead = torch.nn.Linear(c_v2, 3, bias=True)

        init.trunc_normal_(self.conv1.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.linear2.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)
        init.trunc_normal_(self.linear_valuehead.weight, mean=0, std=0.02, a=-2.0*0.02, b=2.0*0.02)

    def forward(self, x):
        outv1 = x
        outv1 = self.conv1(outv1)
        outv1 = self.act1(outv1)
        outpooled = self.gpool(outv1).squeeze(-1).squeeze(-1)
        outv2 = self.linear2(outpooled)
        outv2 = self.act2(outv2)

        # Different subheads
        out_value = self.linear_valuehead(outv2)
        return out_value

class NeuralNetWork(nn.Module):
    """Policy and Value Network
    """

    def __init__(self, num_layers, num_channels, n, action_size):
        super(NeuralNetWork, self).__init__()
        self.c_p1 = 32
        self.c_g1 = 32
        self.c_v1 = 32
        self.c_v2 = 64


        # residual block
        # res_list = [ResidualBlock(3, num_channels)] + [ResidualBlock(num_channels, num_channels) for _ in range(num_layers - 1)]
        # self.res_layers = nn.Sequential(*res_list)
        self.resblock = ResidualBlock(3, num_channels)
        self.resblock1 = ResidualBlock(num_channels, num_channels)
        self.resblock2 = ResidualBlock(num_channels, num_channels)
        self.resblock3 = ResidualBlock(num_channels, num_channels)
        self.resblock4 = ResidualBlock(num_channels, num_channels)
        self.resblock5 = ResidualBlock(num_channels, num_channels)
        self.resblock6 = ResidualBlock(num_channels, num_channels)
        self.resblock7 = ResidualBlock(num_channels, num_channels)
        self.gpool1 = KataConvAndGPool(num_channels, num_channels, num_channels // 2)
        self.gpool2 = KataConvAndGPool(num_channels, num_channels, num_channels // 2)
        self.gpool3 = KataConvAndGPool(num_channels, num_channels, num_channels // 2)

        # policy head
        self.policy_head = PolicyHead(num_channels, self.c_p1, self.c_g1)

        # self.p_conv = nn.Conv2d(num_channels, 4, kernel_size=1, padding=0, bias=False)
        # self.p_bn = nn.BatchNorm2d(num_features=4)
        # self.relu = nn.ReLU(inplace=True)

        # self.p_fc = nn.Linear(4 * n ** 2, action_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # value head
        self.value_head = ValueHead(num_channels, self.c_v1, self.c_v2)

        # self.v_conv = nn.Conv2d(num_channels, 2, kernel_size=1, padding=0, bias=False)
        # self.v_bn = nn.BatchNorm2d(num_features=2)

        # self.v_fc1 = nn.Linear(2 * n ** 2, 256)
        # self.v_fc2 = nn.Linear(256, 1)
        # self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        # out = self.res_layers(inputs)
        out = self.resblock(inputs)

        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.gpool1(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.gpool2(out)
        out = self.resblock6(out)
        out = self.resblock7(out)
        out = self.gpool2(out)

        # policy head
        p = self.policy_head(out)
        p1 = self.log_softmax(p[:,0,:])
        p2 = self.log_softmax(p[:,1,:])
        # value head
        v = self.value_head(out)
        v = self.log_softmax(v)

        return p1, p2, v


class AlphaLoss(nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, log_ps, log_nxt_ps, log_vs, target_ps, target_nxt_ps, target_vs):
        value_loss = -torch.mean(torch.sum(target_vs * log_vs, 1))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))
        nxt_policy_loss = -torch.mean(torch.sum(target_nxt_ps * log_nxt_ps, 1))
        return 1.5 * value_loss + policy_loss + 0.15 * nxt_policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, lr, l2, num_layers, num_channels, n, action_size, train_use_gpu=True, libtorch_use_gpu=True):
        """ init
        """
        self.lr = lr
        self.l2 = l2
        self.num_channels = num_channels
        self.n = n

        self.libtorch_use_gpu = libtorch_use_gpu
        self.train_use_gpu = train_use_gpu

        self.neural_network = NeuralNetWork(num_layers, num_channels, n, action_size)
        if self.train_use_gpu:
            self.neural_network.cuda()

        self.optim = SGD(self.neural_network.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00003)
        # self.optim = AdamW(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.alpha_loss = AlphaLoss()

    def train(self, example_buffer, batch_size, epochs):
        """train neural network
        """
        for epo in range(1, epochs + 1):
            self.neural_network.train()

            # sample
            train_data = random.sample(example_buffer, batch_size)

            # extract train data
            board_batch, last_action_batch, cur_player_batch, p_batch, nxt_p_batch, v_batch = list(zip(*train_data))

            state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
            p_batch = torch.Tensor(p_batch).cuda() if self.train_use_gpu else torch.Tensor(p_batch)
            nxt_p_batch = torch.Tensor(nxt_p_batch).cuda() if self.train_use_gpu else torch.Tensor(nxt_p_batch)
            v_batch = torch.Tensor(v_batch).cuda() if self.train_use_gpu else torch.Tensor(v_batch)
            # v_batch = torch.Tensor(v_batch).unsqueeze(
            #     1).cuda() if self.train_use_gpu else torch.Tensor(v_batch).unsqueeze(1)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward + backward + optimize
            log_ps, nxt_log_ps, vs = self.neural_network(state_batch)
            loss = self.alpha_loss(log_ps,nxt_log_ps, vs, p_batch, nxt_p_batch, v_batch)
            loss.backward()

            self.optim.step()

            # calculate entropy
            new_p, new_nxt_p, new_v = self._infer(state_batch)

            new_p_entropy = -np.mean(
                np.sum(new_p * np.log(new_p + 1e-10), axis=1)
            )

            new_nxt_p_entropy = -np.mean(
                np.sum(new_nxt_p * np.log(new_nxt_p + 1e-10), axis=1)
            )

            new_v_entropy = -np.mean(
                np.sum(new_v * np.log(new_v + 1e-10), axis=1)
            )
            print("EPOCH: {}, LOSS: {} \n Policy Entropy: {:.3f}, Auxiliary Policy Entropy: {:.3f}, Value Entropy: {:.3f}".format(epo, loss.item(), new_p_entropy, 0.15 * new_nxt_p_entropy, 1.5 * new_v_entropy))

    def infer(self, feature_batch):
        """predict p and v by raw input
           return numpy
        """
        board_batch, last_action_batch, cur_player_batch = list(zip(*feature_batch))
        states = self._data_convert(board_batch, last_action_batch, cur_player_batch)

        self.neural_network.eval()
        log_ps, nxt_log_ps, log_vs = self.neural_network(states)

        return np.exp(log_ps.cpu().detach().numpy()), np.exp(nxt_log_ps.cpu().detach().numpy()), np.exp(log_vs.cpu().detach().numpy())

    def _infer(self, state_batch):
        """predict p and v by state
           return numpy object
        """

        self.neural_network.eval()
        log_ps, nxt_log_ps, log_vs = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), np.exp(nxt_log_ps.cpu().detach().numpy()), np.exp(log_vs.cpu().detach().numpy())

    def _data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """convert data format
           return tensor
        """
        n = self.n

        board_batch = torch.Tensor(board_batch).unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        state2 = torch.zeros((len(last_action_batch), 1, n, n)).float()

        for i in range(len(board_batch)):
            if cur_player_batch[i] == -1:
                temp = state0[i].clone()
                state0[i].copy_(state1[i])
                state1[i].copy_(temp)

            last_action = last_action_batch[i]
            if last_action != -1:
                x, y = last_action // self.n, last_action % self.n
                state2[i][0][x][y] = 1

        res =  torch.cat((state0, state1, state2), dim=1)
        # res = torch.cat((state0, state1), dim=1)
        return res.cuda() if self.train_use_gpu else res

    def set_learning_rate(self, lr):
        """set learning rate
        """

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def load_model(self, folder="models", filename="checkpoint"):
        """load model from file
        """

        filepath = os.path.join(folder, filename)
        state = torch.load(filepath)
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])

    def save_model(self, folder="models", filename="checkpoint"):
        """save model to file
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        state = {'network':self.neural_network.state_dict(), 'optim':self.optim.state_dict()}
        torch.save(state, filepath)

        # save torchscript
        filepath += '.pt'
        self.neural_network.eval()

        if self.libtorch_use_gpu:
            self.neural_network.cuda()
            example = torch.rand(1, 3, self.n, self.n).cuda()
        else:
            self.neural_network.cpu()
            example = torch.rand(1, 3, self.n, self.n).cpu()

        traced_script_module = torch.jit.trace(self.neural_network, example)
        traced_script_module.save(filepath)

        if self.train_use_gpu:
            self.neural_network.cuda()
        else:
            self.neural_network.cpu()
