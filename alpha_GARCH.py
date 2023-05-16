import pandas as pd
import numpy as np
import argparse

def add_GARCH(df, **kwargs):
    if len(kwargs.items()) == 0:
        length = 60
    else:
        length = kwargs['length']
        
    laggedReturn = np.log(df['Close'].shift(1)/df['Close'].shift(2))
    laggedVariance = laggedReturn.rolling(length, min_periods=1).std().shift(1) ** 2
    squaredLaggedReturn = laggedReturn ** 2
    realizedVar = squaredLaggedReturn.rolling(length, min_periods=1).mean()
    
    # SSE of ema price and realized variance
    sseArrayLambda = []
    for i in range(1, 100):
        sseLambda = np.nansum(pow(((((i/100) * laggedVariance) + ((1-(i/100))	* squaredLaggedReturn))	- realizedVar), 2))
        sseArrayLambda.append(sseLambda)
        
    optimalSSELambda = min(sseArrayLambda)
    Lambda_Index = (sseArrayLambda.index(optimalSSELambda))
    # The lambda of the EWMA and the Beta of Garch are equivalent
    Beta_Index = Lambda_Index + 1
    Beta_Index

    longrunVariance = laggedVariance.rolling(length, min_periods=1).mean()

    # Use SSE to fine optimal Gamma weight
    sseArrayGamma = []
    for j in range(100 - Beta_Index):
        sseGamma = np.nansum(pow((((j/100)*longrunVariance)+((1-Beta_Index-j)/100)*squaredLaggedReturn)+((Beta_Index/100)*laggedVariance),2))
        sseArrayGamma.append(sseGamma)

    optimalSSEGamma = min(sseArrayGamma)
    Gamma_Index = (sseArrayGamma.index(optimalSSEGamma)) + 1

    Beta = Beta_Index/100
    Gamma = Gamma_Index/100
    Alpha = 1 - Beta - Gamma

    # GARCH(1, 1) prediction
    GARCH = (Gamma * longrunVariance) + (Alpha * squaredLaggedReturn) + (Beta * laggedVariance)
    
    df['projectedDeviation'] = np.sqrt(GARCH) * 100 # in percent
    
    return df
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--length', type=int, default=60)
    args = parser.parse_args()
    
    path = f'../data/single_csv/binance_{args.Symbol}_2018_2023.csv'
    df = pd.read_csv(path, index_col=0)
    param = {'length': args.length}
    df = add_GARCH(df, **param)
    df.to_csv(f'./{args.Symbol}_alpha_GARCH.csv')
    