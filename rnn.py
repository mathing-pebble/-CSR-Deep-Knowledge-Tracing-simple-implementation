import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class RNN(nn.Module):
    def __init__(self, params):
        super(RNN, self).__init__()
        self.n_questions = params['n_questions']
        self.n_hidden = params['n_hidden']
        self.use_dropout = params['dropout']
        self.max_grad = params['maxGrad']
        self.dropout_pred = params['dropoutPred']
        self.max_steps = params['maxSteps']
        self.n_input = self.n_questions * 2

        if params.get('compressedSensing', False):
            self.n_input = params['compressedDim']
            torch.manual_seed(12345)
            self.basis = torch.randn(self.n_questions * 2, self.n_input)
        else:
            self.basis = None

        self.build(params)

    def build(self, params):
        self.start = nn.Linear(1, self.n_hidden)
        self.transfer = nn.Linear(self.n_hidden, self.n_hidden)
        self.linX = nn.Linear(self.n_input, self.n_hidden)
        self.linY = nn.Linear(self.n_hidden, self.n_questions)
        self.hidden_activation = nn.Tanh()
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout_pred)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputM, inputX, inputY, truth):
        if self.basis is not None:
            inputX = torch.mm(inputX, self.basis)
        linM = self.transfer(inputM)
        linX = self.linX(inputX)
        madd = linM + linX
        hidden = self.hidden_activation(madd)
        if self.use_dropout:
            pred_input = self.dropout_layer(hidden)
        else:
            pred_input = hidden
        linY = self.linY(pred_input)
        pred_output = self.sigmoid(linY)
        pred = (pred_output * inputY).sum(dim=1)
        return pred, hidden

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), os.path.join(dir_path, 'model.pth'))

    def load(self, dir_path):
        self.load_state_dict(torch.load(os.path.join(dir_path, 'model.pth')))
        self.eval()