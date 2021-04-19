# -*- coding: utf-8 -*-
# @Time : 2021/4/15 15:17
# @Author : Jclian91
# @File : keras_loss_function_test.py
# @Place : Yangpu, Shanghai
import math
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python import math_ops, array_ops
from keras import backend as K
from keras.losses import (mean_squared_error,
                          mean_absolute_error,
                          binary_crossentropy,
                          categorical_crossentropy,
                          sparse_categorical_crossentropy,
                          hinge
                          )


# mse
y_true = tf.constant([[10.0, 7.0]])
y_pred = tf.constant([[8.0, 6.0]])
loss = mean_squared_error(y_true, y_pred)
loss = K.eval(loss)
print(f'Value of Mean Squared Error is {loss}')

# mae
y_true = tf.constant([[10.0, 7.0]])
y_pred = tf.constant([[8.0, 6.0]])
loss = mean_absolute_error(y_true, y_pred)
loss = K.eval(loss)
print(f'Value of Mean Absolute Error is {loss}')

# Binary Cross Entropy
y_true = tf.constant([[1.0], [0.0]])
y_pred = tf.constant([[0.9], [0.2]])
loss = binary_crossentropy(y_true, y_pred)
loss = K.eval(loss)
python_loss = [1.0 * (-math.log(0.9)), 1.0 * (-math.log(0.8))]
print(f'Compute Binary Cross Entropy in python: {python_loss}')
print(f'Compute Binary Cross Entropy in Keras: {loss}')

# Categorical Cross Entropy
y_true = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
y_pred = [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]
loss = (-math.log(0.9)-math.log(0.89)-math.log(0.94))/3
print(f"compute CCE by python: {loss}")
loss = log_loss(y_true, y_pred)
print(f"compute CCE by sklearn: {loss}")
loss = K.sum(categorical_crossentropy(tf.constant(y_true), tf.constant(y_pred)))/3
loss = K.eval(loss)
print(f"compute CCE by Keras: {loss}")

# Sparse Categorical Cross Entropy
t = LabelEncoder()
y_pred = tf.constant([[0.1, 0.1, 0.8], [0.1, 0.4, 0.5], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
y_true = t.fit_transform(['Rain', 'Rain', 'High Changes of Rain', 'No Rain'])
print("transformed label: ", y_true)
y_true = tf.constant(y_true)
loss = sparse_categorical_crossentropy(y_true, y_pred)
loss = K.eval(loss)
print(f'Value of Sparse Categorical Cross Entropy is ', loss)


# hinge loss
y_true = tf.constant([[0., 1.], [0., 0.]])
y_pred = tf.constant([[0.7, 0.3], [0.4, 0.6]])
loss = hinge(y_true, y_pred)
a = K.eval(loss)
print("hinge loss: ", a)


# Custom Loss Function: categorical_crossentropy_with_label_smoothing
def categorical_crossentropy_with_label_smoothing(y_true, y_pred, label_smoothing=0.1):
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    return categorical_crossentropy(y_true, y_pred)


y_true = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
y_pred = tf.constant([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
loss = categorical_crossentropy_with_label_smoothing(y_true, y_pred, label_smoothing=0.3)
loss = K.eval(loss)
print(f"compute CCE with label smoothing by Custom Loss function in Keras: {loss}")
print("compute CCE with label smoothing by Custom Loss function in Keras: {}, {}, {}\n".format(
0.8*(-math.log(0.9))+0.1*(-math.log(0.05))+0.1*(-math.log(0.05)),
0.1*(-math.log(0.05))+0.8*(-math.log(0.89))+0.1*(-math.log(0.06)),
0.1*(-math.log(0.05))+0.1*(-math.log(0.01))+0.8*(-math.log(0.94))
))

# use Custom Loss Function: categorical_crossentropy_with_label_smoothing for classification model in Keras
import keras as K
import pandas as pd
# 1. 读取数据
df = pd.read_csv("iris.csv", header=None)
targets = df[4].unique()
targets_dict = dict(zip(targets, range(len(targets))))
df["target"] = df[4].apply(lambda x: targets_dict[x])
print(targets_dict)
print(df.head())
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)

# 2. 定义模型
init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=5, input_dim=4, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
model.compile(loss=categorical_crossentropy_with_label_smoothing, optimizer=simple_adam, metrics=['accuracy'])
# 3. 模型训练
train_x = train_df[[0, 1, 2, 3]]
train_y = train_df["target"]
test_x = test_df[[0, 1, 2, 3]]
test_y = test_df["target"]
h = model.fit(train_x, train_y, batch_size=1, epochs=100, shuffle=True, verbose=1)
# 4. 评估模型
eval = model.evaluate(test_x, test_y, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))

