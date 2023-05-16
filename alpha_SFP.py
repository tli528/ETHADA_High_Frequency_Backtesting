import pandas as pd
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm


def add_SFP(df, **kwargs):
    if len(kwargs.items()) == 0:
        lookback = 21
    else:
        lookback = kwargs['lookback']
    
    pivot_high = df['High'] * (df['High'] >= df['High'].rolling(lookback).max())
    pivot_low  = df['Low'] * (df['Low'] <= df['Low'].rolling(lookback).min())
    
    swing_high_failure_pattern = (-1.0 * ((pivot_high != 0) * (df['Close'] < pivot_high.rolling(lookback).max())))
    swing_low_failure_pattern = (1.0 * ((pivot_low != 0) * (df['Close'] > pivot_low.rolling(lookback).min())))
    
    df['SFP'] =  swing_high_failure_pattern + swing_low_failure_pattern

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--lookback', type=int, default=21)
    args = parser.parse_args()
    
    path = f'../data/single_csv/binance_{args.Symbol}_2018_2023.csv'
    param = {'lookback': args.lookback}
    
    df = pd.read_csv(path, index_col=0)

    df = add_SFP(df, **param)
    df.to_csv(f'./{args.Symbol}_alpha_SFP.csv')

    