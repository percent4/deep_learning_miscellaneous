# -*- coding: utf-8 -*-
# @Time : 2021/4/19 13:44
# @Author : Jclian91
# @File : torch_loss_function_test.py
# @Place : Yangpu, Shanghai
import math
import torch as T
from sklearn.metrics import log_loss
from torch.nn import MSELoss, L1Loss, BCELoss, CrossEntropyLoss, NLLLoss, Softmax

# mse
y_true = T.tensor([[10.0], [7.0]])
y_pred = T.tensor([[8.0], [6.0]])
loss = MSELoss()(y_pred, y_true)
print(f'Value of Mean Squared Error is {loss}')

# mae
y_true = T.tensor([[10.0, 7.0]])
y_pred = T.tensor([[8.0, 6.0]])
loss = L1Loss()(y_pred, y_true,)
print(f'Value of Mean Absolute Error is {loss}')

# Binary Cross Entropy
y_true = T.tensor([[1.0], [0.0]])
y_pred = T.tensor([[0.9], [0.2]])
loss = BCELoss()(y_pred, y_true)
python_loss = (1.0 * (-math.log(0.9)) + 1.0 * (-math.log(0.8)))/2
print(f'Compute Binary Cross Entropy in python: {python_loss}')
print(f'Compute Binary Cross Entropy in Torch: {loss}')

# Categorical Cross Entropy
y_true = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
y_true_torch = T.tensor([0, 1, 2])
y_pred_tmp = T.randn(3, 3)
y_pred = Softmax(dim=1)(y_pred_tmp)
print("y_pred in softmax: ", y_pred)
y_pred_numpy = y_pred.numpy()
loss = (-math.log(y_pred_numpy[0][0])-math.log(y_pred_numpy[1][1])-math.log(y_pred_numpy[2][2]))/3
print(f"compute CCE by python: {loss}")
loss = log_loss(y_true, y_pred)
print(f"compute CCE by sklearn: {loss}")
loss = CrossEntropyLoss()(y_pred_tmp, y_true_torch)
print(f"compute CCE by Torch CrossEntropyLoss: {loss}")
log_value = T.log(y_pred)
torch_loss = T.sum(T.tensor(y_true).mul(-log_value))/3
print(f"compute CCE by Torch Operation: {torch_loss}")
torch_cce_loss = NLLLoss()(log_value, y_true_torch)
print(f"compute CCE by validation: {torch_cce_loss}")


# hinge loss
class MyHingeLoss(T.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        # output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        hinge_loss = 1 - T.sum(T.mul(output, target), dim=0)
        hinge_loss[hinge_loss < 0] = 0
        return T.mean(hinge_loss)


y_true = T.tensor([[0., 1.], [1., 0.]])
y_pred = T.tensor([[0.7, 0.3], [0.4, 0.6]])
loss = MyHingeLoss()(y_pred, y_true)
print("hinge loss: ", loss)

