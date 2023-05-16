import pandas as pd
import numpy as np
import tqdm
import os
import torch
from torch.utils.data import DataLoader

from model.dataset import SymbolDataset
from model.model_GRU import GRU
from model.model_LSTM import LSTM
from model.model_transformer import Transformer
from model.model_CNN import CNN

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

import argparse
import warnings
warnings.filterwarnings('ignore')


checkpointdir = './checkpoints/'

def l2_loss(pred, label):
	loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
	return loss


def cls_loss(pred, label):
	pred = (torch.sigmoid(pred) > 0.5) * 1.0
	label = torch.tensor((label.reshape(-1, 1) > 0) * 1.0, requires_grad=True)
	criterion = torch.nn.BCELoss()
	loss = criterion(pred, label)
	return loss


def train(model, dataloader, optimizer, mode):
	model.train()
	loader = tqdm.tqdm(dataloader)
	loss_epoch = 0
	for idx, (data, label) in enumerate(loader):
		data, label = data.float(), label.float()
		output = model(data)
		optimizer.zero_grad()
		if mode == 'reg':
			loss = l2_loss(output, label)
		else:
			loss = cls_loss(output, label)
		loss.backward()
		optimizer.step()
		loss_epoch += loss.item()
		# print('loss',loss)

	loss_epoch /= len(loader)
	return loss_epoch


def eval(model, dataloader, mode):
	model.eval()
	loader = tqdm.tqdm(dataloader)
	loss_epoch = 0
 
	for idx, (data, label) in enumerate(loader):
		data, label = data.float(), label.float()
		output = model(data)
		if mode == 'reg':
			loss = l2_loss(output, label)
		else:
			loss = cls_loss(output, label)
   
		loss_epoch += loss.detach().item()
	loss_epoch /= len(loader)
	return loss_epoch


nn_model_set = {
	'GRU': GRU,
	'LSTM': LSTM, 
	'transformer': Transformer,
	'CNN': CNN
}


def backtest_models(symbol, _model, mode, bar_gap=1):
	assert _model in ['Lasso', 'Random Forest', 'GRU', 'LSTM', 'transformer', 'CNN'], "Invalid Model"
	if '_checkpoints' not in os.listdir('model/'):
		os.mkdir('model/_checkpoints')
    
	if _model in ['Lasso', 'Random Forest']:
		df = pd.read_csv(f'data/single_csv/alpha_{symbol}.csv')

		X = df[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'SFP','MaMA', 'projectedDeviation']].fillna(0)
		y = df['Close'].pct_change(1).shift(-1).fillna(0)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

		scaler = StandardScaler().fit(X_train[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'MaMA', 'projectedDeviation']]) 

		X_train[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'MaMA', 'projectedDeviation']] = scaler.transform(X_train[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'MaMA', 'projectedDeviation']])

		X_test[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'MaMA', 'projectedDeviation']] = scaler.transform(X_test[['trade_at_current_ts', 'buy_sell_ratio_at_current_ts', 'macd', 'MaMA', 'projectedDeviation']])
		
		if _model == 'Lasso':
			model = Lasso(alpha=1)
			model.fit(X_train, y_train)

		else:
			model = RandomForestRegressor(n_estimators = 100, random_state = 0)
			model.fit(X_train, y_train)

		joblib_file = f"model/checkpoints/{symbol}_{_model}_{mode}_{bar_gap}.pkl"
		joblib.dump(model, joblib_file)
		
	else:
		# dataset
		dataset_train = SymbolDataset(symbol, bar_gap) # bar_gap: diff bar
		dataset_test = SymbolDataset(symbol, bar_gap, train_flag=False)
		###if MLP,CNN,batch_size = 1
		train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
		test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
		model = nn_model_set[_model]()

		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		total_epoch = 50
	
		for epoch_idx in range(total_epoch):
			if mode == 'reg':
				train_loss = train(model, train_loader, optimizer, 'reg')
			else:
				train_loss = train(model, train_loader, optimizer, 'cls')
			print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))
	
			if epoch_idx % 5==0:
				optimal_loss = np.Inf
				if mode == 'reg':
					eval_loss = eval(model, test_loader, 'reg')
				else:
					eval_loss = eval(model, test_loader, 'cls')
				print("stage: test, epoch:{:5d}, loss:{}".format(epoch_idx, eval_loss))
				if optimal_loss > eval_loss:
					optimal_loss = eval_loss
					save_path = f"model/_checkpoints/{symbol}_{_model}_{mode}_{bar_gap}.ckpt"
					torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Symbol', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--bar_gap', type=int, help='gap between current bar and previous bar')
    args = parser.parse_args()
    
    backtest_models(args.Symbol, args.model, args.mode, args.bar_gap)