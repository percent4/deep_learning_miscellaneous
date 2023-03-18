# -*- coding: utf-8 -*-
# @Time : 2021/4/20 14:20
# @Author : Jclian91
# @File : pytorch_gather_test.py
# @Place : Yangpu, Shanghai
import torch
a = torch.randint(0, 30, (2, 3, 5))
print(a)
index = torch.LongTensor([[[0,1,2,0,2],
                          [0,0,0,0,0],
                          [1,1,1,1,1]],
                        [[1,2,2,2,2],
                         [0,0,0,0,0],
                         [2,2,2,2,2]]])
print(a.size() == index.size())
b = torch.gather(a, 1, index)
print(b)

import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.gather(a, 1, torch.LongTensor([[0, 0], [1, 0]]))
print(a)
print(b)

# view function
label = torch.tensor([2, 3, 4])
print(label.view(-1))