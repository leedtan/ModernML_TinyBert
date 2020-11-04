# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Cheolhyoung Lee
# Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
# Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import Parameter
from torch.autograd.function import InplaceFunction

import numpy as np
import copy


class Mixout_normal(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):

        #target is pretrained
        #input is fine-tuned
        # p is probability to use pre-trained (dropout probability)
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[:, 0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[:, 0].repeat(input.size()[1], 1)
            ctx.noise = torch.transpose(ctx.noise, 0, 1)
        # import pdb
        # print(ctx.noise.shape)
        # print(input.shape)
        # pdb.set_trace()
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output) * torch.norm(
                output) / torch.norm((1 - ctx.noise) * target + ctx.noise * output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p)
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None.
    # I modified the code of dropout in PyTorch.
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise *
                      output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout_normal.apply(input, target, p, training, inplace)


class mixout_layer(nn.Module):
    def __init__(self, linear, p, norm_flag=True):
        super().__init__()
        self.layer = linear
        self.norm_flag = norm_flag
        self.p = p
        self.layer_frozen = copy.deepcopy(linear)
        for param in self.layer_frozen.parameters():
            param.requires_grad = False

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if not self.training or self.p == 0:
            return self.layer(x)

        x_shape = x.shape
        x = torch.flatten(x, end_dim=-2)
        learned_layer_output = self.layer(x)
        frozen_layer_output = self.layer_frozen(x)
        self.noise = torch.FloatTensor(
            x.shape[0], self.layer.out_features).uniform_(0, 1)
        self.mask = (self.noise < self.p).type(torch.FloatTensor)
        self.masked_learned = learned_layer_output * (1-self.mask)
        self.masked_frozen = frozen_layer_output * self.mask
        self.raw_output = self.masked_learned + self.masked_frozen
        # self.num_scale = self.normalize(
        #     learned_layer_output, frozen_layer_output)
        # self.denom_scale = self.normalize(
        #     self.raw_output, frozen_layer_output, keepdim=True, dim=[1])
        self.desired_norm = self.normalize(
            learned_layer_output, frozen_layer_output)
        self.raw_norm = self.normalize(self.raw_output, frozen_layer_output,
                                       keepdim=True, dim=[1])
        delta = (self.raw_output - frozen_layer_output)
        self.output = delta * (self.desired_norm / self.raw_norm) + frozen_layer_output
        self.output = self.output.view(*x_shape[:-1], -1)
        return self.output

    def normalize(self, x, x_frozen, dim=None, keepdim=False):
        return torch.norm(x - x_frozen, dim=dim, keepdim=keepdim, p=1) + 1e-10


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p))
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer,
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p)

    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # fine tuned
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        # pretrained
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )
