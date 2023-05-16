import argparse
import pandas as pd

from alpha_MACD import add_macd
from alpha_SFP import add_SFP
from alpha_MaMA import add_MaMA
from alpha_GARCH import add_GARCH

def add_default_alpha(df):
    df = df.set_index('Open time', drop=False)
    # add macd
    macd_param = {'sma': 12, 'lma': 26, 'tsp': 9}
    df = add_macd(df, **macd_param)
    # add SFP
    SFP_param = {'lookback': 21}
    df = add_SFP(df, **SFP_param)
    # add MaMA
    MaMA_param = {'mlength': 34, 'plength': 13, 'length': 55}
    df = add_MaMA(df, **MaMA_param)
    # add GARCH
    GARCH_param = {'length': 60}
    df = add_GARCH(df, **GARCH_param)
    
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    args = parser.parse_args()
    
    path = f'../data/single_csv/binance_{args.Symbol}_2018_2023.csv'
    df = pd.read_csv(path, index_col=0)

    df = add_default_alpha(df)
    df.to_csv(f'../data/single_csv/alpha_{args.Symbol}.csv', index=False)