import pandas as pd
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm

def add_macd(df, **kwarg):
    if len(kwarg.items()) == 0: # default parameters
        sma, lma, tsp = 12, 26, 9
    else:
        assert len(kwarg) == 3, 'Invaild Parameters'
        sma, lma, tsp = kwarg['sma'], kwarg['lma'], kwarg['tsp']
    
    MMEslowa = df['Close'].ewm(span=lma).mean()
    MMEslowb = MMEslowa.ewm(span=lma).mean()
    DEMAslow = (2 * MMEslowa) - MMEslowb
    
    MMEfasta = df['Close'].ewm(span=sma).mean()
    MMEfastb = MMEfasta.ewm(span=sma).mean()
    DEMAfast = (2 * MMEfasta) - MMEfastb
    
    Ligne_MACD_Zero_Lag = (DEMAfast - DEMAslow)
    
    MMEsignala = Ligne_MACD_Zero_Lag.ewm(span=tsp).mean()
    MMEsignalb = MMEsignala.ewm(span=tsp).mean()
    Ligne_signal = (2 * MMEsignala) - MMEsignalb
    
    MACD_Zero_Lag = (Ligne_MACD_Zero_Lag - Ligne_signal)

    df['macd'] = MACD_Zero_Lag
    
    return df


# tune the parameters of the macd based on correlation with the target
def tune_parameters(train_df, target, alpha, parameters_list):
    train_df = train_df.copy(deep = False)
    if alpha == 'MACD':
        df = pd.DataFrame(columns=['sma', 'lma', 'tsp', 'corr'])
        for param_set in tqdm(parameters_list, desc='Iterating'):
            param = {'sma': param_set[0], 'lma': param_set[1], 'tsp': param_set[2]}
            # add macd to the dataframe
            train_df = add_macd(train_df, **param)
            # calculate the correlation between macd and the target
            corr = abs(train_df['macd'].corr(target.fillna(0)))
            # add the results to the dataframe
            df = pd.concat([df, pd.DataFrame([param_set[0], param_set[1], param_set[2], corr], index=['sma', 'lma', 'tsp', 'corr']).T.fillna(0)])
            
    # sort the dataframe by correlation
    df = df.sort_values(by=['corr'], ascending=False)
    # return the top 5 results
    return df.head(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--tune', type=bool, default=False)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--sma', type=int, default=12)
    parser.add_argument('--lma', type=int, default=26)
    parser.add_argument('--tsp', type=int, default=9)
    args = parser.parse_args()
    
    path = f'../data/single_csv/binance_{args.Symbol}_2018_2023.csv'
    param = {'sma': args.sma, 'lma': args.lma, 'tsp': args.tsp}
    
    df = pd.read_csv(path, index_col=0)
    if not args.tune:
        df = add_macd(df, **param)
        df.to_csv(f'./{args.Symbol}_alpha_MACD.csv')
    
    else:
        sma_set = list(np.linspace(1, 20, 20, dtype=int))
        lma_set = list(np.linspace(1, 20, 20, dtype=int))
        tsp_set = list(np.linspace(1, 20, 20, dtype=int))
        macd_params_list = list(product(sma_set, lma_set, tsp_set))
        head = tune_parameters(df, df['Close'].pct_change(args.target).shift(-args.target).fillna(0), 'MACD', macd_params_list)
        head.to_csv(f'./{args.Symbol}_MACD_tune_results.csv')
    