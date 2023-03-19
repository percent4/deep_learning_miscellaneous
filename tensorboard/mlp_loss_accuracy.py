# -*- coding: utf-8 -*-
import torch
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_

from tensorboardX import SummaryWriter


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 5)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(5, 6)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(6, 3)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        return X


class Model(object):
    def __init__(self, file_path, model):
        self.writer = SummaryWriter('./run/mlp_demo')
        # load the dataset
        dataset = CSVDataset(file_path)
        # calculate split
        train, test = dataset.get_splits()
        # prepare data loaders
        self.train_dl = DataLoader(train, batch_size=4, shuffle=True)
        self.test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        # model
        self.model = model

    # train the model
    def train(self):
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters())
        # enumerate epochs
        for epoch in range(100):
            init_loss = torch.Tensor([0.0])
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(self.train_dl):
                targets = targets.long()
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
                init_loss += loss.data
                # update model weights
                optimizer.step()

            self.writer.add_scalar('Loss/Train', init_loss/(i+1), epoch)
            test_accuracy = self.evaluate_model()
            self.writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

    # evaluate the model
    def evaluate_model(self):
        predictions, actuals = [], []
        for i, (inputs, targets) in enumerate(self.test_dl):
            # evaluate the model on the test set
            yhat = self.model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc


if __name__ == '__main__':
    # train the model
    Model('iris.csv', MLP(4)).train()
