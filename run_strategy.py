import pandas as pd
import torch
import os
import argparse
import backtrader as bt

from model.model_CNN import CNN
from model.model_GRU import GRU
from model.model_LSTM import LSTM 
from model.model_transformer import Transformer
from model.model_train import backtest_models
from backtest import backtest, BaseStrategy


nn_model_set = {
	'GRU': GRU,
	'LSTM': LSTM, 
	'transformer': Transformer,
	'CNN': CNN
}


def run_strategy(symbol, interval, model_type):

    if ('_checkpoints') not in os.listdir('model'):
        os.mkdir('model/_checkpoints')
    
    model_path = f'model/_checkpoints/{symbol}_{model_type}_reg_{interval}.ckpt'
    if model_path.split('/')[-1] not in os.listdir('model/_checkpoints/'):
        print(f'Train Model: {symbol}_{model_type}_reg_{interval}')
        file = open(model_path,'w')
        file.close()
        backtest_models(symbol, model_type, 'reg', interval)
    
    checkpoint = torch.load(f'model/_checkpoints/{symbol}_{model_type}_reg_{interval}.ckpt')
    
    model = nn_model_set[model_type]()
    model.load_state_dict(checkpoint)
    
    df = pd.read_csv(f'data/single_csv/alpha_{symbol}.csv')
    input = torch.tensor(df[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'SFP', 'MaMA', 'projectedDeviation']].fillna(0).values)
    predictions = torch.sigmoid(model(input.to(torch.float32))).detach().numpy()
    predictions = 1.0*(predictions >= 0.6) - 1.0*(predictions <= 0.4)
    
    analysis = backtest(BaseStrategy, df, predictions)
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--mode', type=str, default='reg')
    parser.add_argument('--model_type', type=str, default='CNN')
    args = parser.parse_args()
    
    analysis = run_strategy(args.Symbol, args.interval, args.model_type)
    analysis.to_csv(f'{args.Symbol}_{args.interval}_{args.model_type}_analysis.csv')