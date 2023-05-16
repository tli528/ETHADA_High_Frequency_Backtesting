import pandas as pd
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm


def add_MaMA(df, **kwargs):
    if len(kwargs.items()) == 0:
        mlength, plength, length = 34, 13, 55 # default params
    else:
        mlength, plength, length = kwargs['mlength'], kwargs['plength'], kwargs['length']
    
    momentum = df['Close'].diff(mlength)
    acceleration = momentum.diff(mlength)
    probability = ((df['Close'].diff() > 0) * 1.0).rolling(plength).sum() / plength
    adjustedSource = df['Close'] + (momentum + 0.5 * acceleration) * probability
    df['MaMA'] = adjustedSource.ewm(length).mean()
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--mlength', type=int, default=34)
    parser.add_argument('--plength', type=int, default=13)
    parser.add_argument('--length', type=int, default=55)
    args = parser.parse_args()
    
    path = f'../data/single_csv/binance_{args.Symbol}_2018_2023.csv'
    param = {'mlength': args.mlength, 'plength': args.plength, 'length': args.length}

    df = pd.read_csv(path, index_col=0)

    df = add_MaMA(df, **param)
    df.to_csv(f'./{args.Symbol}_alpha_MaMA.csv')
