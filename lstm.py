import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class LSTMModel(nn.Module):
    def __init__(self, params):
        super(LSTMModel, self).__init__() 
        self.n_questions = params['n_questions']
        self.n_hidden = params['n_hidden']
        self.use_dropout = params.get('dropout', False)
        self.dropout_pred = params.get('dropoutPred', 0.0)
        self.compressed_sensing = params.get('compressedSensing', False)
        
        if self.compressed_sensing:
            self.n_input = params['compressedDim']
            torch.manual_seed(12345)
            self.basis = nn.Parameter(torch.randn(self.n_questions * 2, self.n_input), requires_grad=False)
        else:
            self.n_input = self.n_questions * 2
            self.basis = None
        
        self.lstm = nn.LSTM(self.n_input, self.n_hidden, batch_first=True)
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout_pred)
        self.output_layer = nn.Linear(self.n_hidden, self.n_questions)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, init_states=None):
        device = x.device
        if self.basis is not None:
            batch_size, sequence_length, _ = x.size()
            x = x.reshape(-1, self.n_questions * 2)  
            x = torch.matmul(x, self.basis.to(device))  
            x = x.reshape(batch_size, sequence_length, -1)
        
        if init_states is None:
            h0 = torch.zeros(1, x.size(0), self.n_hidden).to(device)
            c0 = torch.zeros(1, x.size(0), self.n_hidden).to(device)
            init_states = (h0, c0)
        
        x, (hn, cn) = self.lstm(x, init_states)
        
        if self.use_dropout:
            x = self.dropout_layer(x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), os.path.join(dir_path, 'lstm_model.pth'))

    def load(self, dir_path):
        self.load_state_dict(torch.load(os.path.join(dir_path, 'lstm_model.pth')))
        self.eval()
