import pandas as pd
import numpy as np
import os
import requests, zipfile, io
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


def download_1h_klines(symbol, start_mon, end_mon):
    assert symbol in ['ETHUSDT', 'ADAUSDT'], f"Error: invalid symbol: {symbol}"
    mon_index = pd.date_range(start_mon, end_mon, freq='M').map(lambda x: str(x).split(' ')[0][:-3]).to_list()
    df_agg = pd.DataFrame(columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', \
                                    'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    
    pbar = tqdm(mon_index, colour='blue', desc='Downloading')
    unavailable_data = []
    for mon in pbar:
        pbar.set_postfix({'file': f'{symbol}-1h-{mon}.csv'})
        
        if mon.split('-')[0] not in os.listdir('data'):
            os.mkdir(f"data/{mon.split('-')[0]}")

        zip_file_url = f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{mon}.zip'
        r = requests.get(zip_file_url, stream=True)
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(f'{symbol}-1h-{mon}.csv') as f:
                    df = pd.read_csv(f, names=df_agg.columns)
                    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
                    df.to_csv(f"data/{mon.split('-')[0]}/{symbol}-1h-{mon}.csv")
                    df_agg = pd.concat([df_agg, df]).reset_index(drop=True)
        except:
            unavailable_data.append(f"{symbol}-1h-{mon}.zip")
            continue
    
    if 'agg_data' not in os.listdir('data'):
        os.mkdir('data/agg_data')
    
    df_agg.to_csv(f'data/agg_data/{symbol}-1h-{start_mon}-{end_mon}.csv')
    
    if len(unavailable_data):
        print("Unavailable Data:")
        for zip in unavailable_data:
            print(zip)
            
            

def download_agg_data(symbol, start_mon, end_mon):
    assert symbol in ['ETHUSDT', 'ADAUSDT'], f"Error: invalid symbol: {symbol}"
    
    mon_index = pd.date_range(start_mon, end_mon, freq='M').map(lambda x: str(x).split(' ')[0][:-3]).to_list()
    df_agg = pd.DataFrame(columns=['time', 'trade_at_current_ts', 'buy_sell_ratio_at_current_ts'])
    
    pbar = tqdm(mon_index, colour='blue', desc='Downloading')
    unavailable_data = []
    
    for mon in pbar:
        pbar.set_postfix({'file': f'{symbol}-aggTrades-{mon}.csv'})
        
        if mon.split('-')[0] not in os.listdir('data'):
            os.mkdir(f"data/{mon.split('-')[0]}")

        zip_file_url = f'https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{mon}.zip'

        r = requests.get(zip_file_url, stream=True)
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(f'{symbol}-aggTrades-{mon}.csv') as f:
                    pbar.set_postfix({'process': f'aggregating'})
                    df = pd.read_csv(f, names = ['Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Timestamp', 'Was the buyer the maker', 'Was the trade the best price match'])
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                    agg_data = []
                    for g in df.groupby(df['Timestamp'].apply(lambda x: str(x).split(':')[0])):
                        time_idx = g[0]
                        pbar.set_postfix({'hour': g[0]})
                        trade_at_current_ts = g[1].shape[0]
                        buy_sell_ratio_at_current_ts = sum(g[1]['Was the buyer the maker'] * 1.0) / trade_at_current_ts
                        agg_data.append([time_idx+':00:00', trade_at_current_ts, buy_sell_ratio_at_current_ts])
                    agg_data = pd.DataFrame(agg_data, columns=df_agg.columns)
                    agg_data.to_csv(f"data/{mon.split('-')[0]}/{symbol}-aggTrades-{mon}.csv")
                    df_agg = pd.concat([df_agg, agg_data]).reset_index(drop=True)
                    
        except:
            unavailable_data.append(f"{symbol}-aggTrades-{mon}.zip")
            continue

    df_agg.to_csv(f'data/agg_data/{symbol}-aggTrades-{start_mon}-{end_mon}.csv')
    
    if len(unavailable_data):
        print("Unavailable Data:")
        for zip in unavailable_data:
            print(zip)
            
            
            
def merge_df(symbol):
    df_1 = pd.read_csv(f'data/agg_data/{symbol}-1h-2018-01-2023-01.csv', index_col=0)
    df_2 = pd.read_csv(f'data/agg_data/{symbol}-aggTrades-2018-01-2023-01.csv', index_col=0)
    
    merge_df = pd.merge(df_1, df_2, left_on='Open time', right_on='time', how='left')
    merge_df['trade_at_current_ts'].loc[np.isnan(merge_df['trade_at_current_ts'])] = merge_df['Number of trades'].loc[np.isnan(merge_df['trade_at_current_ts'])]
    merge_df = merge_df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'trade_at_current_ts', 'buy_sell_ratio_at_current_ts']]
    
    merge_df.to_csv(f'data/single_csv/binance_{symbol}_2018_2023.csv')



def data_loader(start_mon, end_mon):
    download_1h_klines('ETHUSDT', start_mon, end_mon)
    download_1h_klines('ADAUSDT', start_mon, end_mon)
    download_agg_data('ETHUSDT', start_mon, end_mon)
    download_agg_data('ADAUSDT', start_mon, end_mon)
    merge_df('ETHUSDT')
    merge_df('ADAUSDT')
    


if __name__ == '__main__':
    if 'data' not in os.listdir('./'):
        os.mkdir('data')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mon', type=str)
    parser.add_argument('--end_mon', type=str)
    args = parser.parse_args()
    start_mon, end_mon = args.start_mon, args.end_mon
    
    data_loader(start_mon, end_mon)