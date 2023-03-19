# -*- coding: utf-8 -*-
# @Time : 2023/3/19 18:00
# @Author : Jclian91
# @File : torch_embedding.py
# @Place : Minghang, Shanghai
import torch
import torch.nn as nn

# Embedding with padding_idx
embedding = nn.Embedding(5, 3, padding_idx=0)
print(embedding.weight)
input = torch.IntTensor([[0, 2, 0, 4]])
print(embedding(input))
