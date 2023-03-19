# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter

with SummaryWriter('./run/onnx') as w:
    w.add_onnx_graph('iris.onnx')
