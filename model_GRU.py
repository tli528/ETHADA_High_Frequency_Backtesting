import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, gru_layer=3, input_dim=6, hidden_size=32):
        super(GRU, self).__init__()
        self.hidden_size=hidden_size
        self.gru_layer = gru_layer
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=gru_layer, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):   
        out, _ = self.gru(x) 
        out = self.out_layer(out)       
        
        return out