# -*- coding: utf-8 -*-
# @Time : 2021/4/19 17:40
# @Author : Jclian91
# @File : torch_layernorm.py
# @Place : Yangpu, Shanghai
import torch as T
from torch.nn import LayerNorm, BatchNorm1d

t = T.FloatTensor([[1, 2, 4, 1],
                   [6, 3, 2, 4],
                   [2, 4, 6, 1]])

# LayerNorm
normed_layer = LayerNorm(t.size()[1])(t)
print(normed_layer)

# BatchNorm
normed_batch = BatchNorm1d(t.size()[1])(t)
print(normed_batch)