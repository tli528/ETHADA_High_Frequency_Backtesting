import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, lstm_layer=3, input_dim=6, hidden_size=32):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.lstm_layer = lstm_layer
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=lstm_layer, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):   
        out, _ = self.lstm(x) 
        out = self.out_layer(out)       
        
        return out