import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SymbolDataset(Dataset):
    def __init__(self, symbol, bar_gap=1, train_flag=True):
        # read data
        data_pd = pd.read_csv(f'data/single_csv/alpha_{symbol}.csv', index_col=0)   
        
        data_pd['target'] = data_pd['Close'].pct_change(bar_gap).shift(-bar_gap)
        
        self.train_flag = train_flag
        self.data_train_ratio = 0.8

        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_pd))
            data_all = np.array(data_pd[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'SFP','MaMA', 'projectedDeviation', 'target']].fillna(0))
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1-self.data_train_ratio) * len(data_pd))
            data_all = np.array(data_pd[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'SFP','MaMA', 'projectedDeviation', 'target']].fillna(0))
            self.data = data_all[-self.data_len:]
            
        if train_flag:
            print("train data size: {}".format(self.data_len))
        else:
            print("test data size: {}".format(self.data_len))
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, -1]