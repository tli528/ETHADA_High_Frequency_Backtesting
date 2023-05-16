import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,dropout=0.1):
        super(CNN, self).__init__()
        self.emb_layer = nn.Linear(6, 3)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU())
            ## avoid over-fitting
            #nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
            ## avoid over-fitting
            #nn.Dropout(p=dropout),
            #nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2))
        self.conv_last = nn.Conv1d(8, 1, kernel_size=1, padding=0)
        self.fc = nn.Linear(6, 1)
        
        
    def forward(self, x):
        x_ = x.unsqueeze(2)
        x_emb = x_.transpose(1,2)
        out1 = self.layer1(x_emb) 
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.conv_last(out3) + x_emb 
        out = self.fc(out.squeeze(1))

        return out