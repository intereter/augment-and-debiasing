from transformers.file_utils import requires_backends
from random import sample
from torch.autograd import Function
import torch
import math
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class MBartForConditionalGeneration_my:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch"])


class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output[0] * -ctx.lambd, None


def cal_GradReverse_lambda(step, total_step, gamma=10):
    lamb = torch.tensor([step/total_step])
    lamb = gamma * lamb
    lamb = -lamb
    lamb = torch.exp(lamb) + 1
    lamb = 2 / lamb - 1
    return lamb.cuda()