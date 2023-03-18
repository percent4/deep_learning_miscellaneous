# -*- coding: utf-8 -*-
# @Time : 2021/4/20 15:03
# @Author : Jclian91
# @File : query_key_value.py
# @Place : Yangpu, Shanghai
# this script explains why equation softmax(Qk^{T}/\sqrt{d_{k}})V in self attention should divide \sqrt{d_{k}}

# support we have Qk^{T} = [4, 9], d_{k} = 16
import torch
a = torch.tensor([4, 9], dtype=torch.float64)
# without divide \sqrt{d_{k}}
b = a.softmax(-1)
print(b)
# with divide \sqrt{d_{k}}
b = (a/4).softmax(-1)
print(b)
