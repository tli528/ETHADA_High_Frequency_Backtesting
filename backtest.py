import backtrader as bt
import pandas as pd
from model.model_CNN import CNN
from model.model_GRU import GRU
from model.model_LSTM import LSTM 
from model.model_transformer import Transformer
import backtrader.analyzers as btanalyzers
import backtrader as bt
from model.model_train import backtest_models


def add_analyzers(cerebro):
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='draw_down')
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(btanalyzers.Transactions, _name='transactions')
    return cerebro


def analyze_strat(strat):
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    annual_r = strat.analyzers.annual_return.get_analysis()
    max_drawdown = str(round(strat.analyzers.draw_down.get_analysis()['max']['drawdown'], 2)) + '%'
    number_of_trades = len(strat.analyzers.transactions.get_analysis())
    profit = []
    for x in strat.analyzers.transactions.get_analysis():
        profit.append(strat.analyzers.transactions.get_analysis()[x][0][4]) 
    total_pnl = sum(profit)

    return pd.DataFrame([sharpe_ratio, annual_r, max_drawdown, number_of_trades, total_pnl], index=['sharpe_ratio', 'annual_r', 'max_drawdown', 'number_of_trades', 'total_pnl']).T



class PandasData_ml(bt.feeds.PandasData):
    lines = ('predicted',)
    params = (('predicted', 6),)
    

class BaseStrategy(bt.Strategy):
    params = dict(
    )

    def __init__(self):
        self.data_predicted = self.datas[0].predicted 
        self.data_open = self.datas[0].Open
        self.data_close = self.datas[0].Close

        self.order = None
        self.price = None
        self.comm = None
    # logging function

    def log(self, txt):
        '''Logging function'''
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return
        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - -- Price: {order.executed.price: .2f}, \
                         Cost: {order.executed.value: .2f}, Commission: {order.executed.comm: .2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED - -- Price: {order.executed.price: .2f}, \
                         Cost: {order.executed.value: .2f}, Commission: {order.executed.comm: .2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
        # Wrong Order
        elif order.status in [order.Canceled, order.Margin,
                              order.Rejected]:
            self.log('Order Failed')
        # No hanging order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(
            f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
        
    def next(self):
        if self.order:
            return
        
        if self.position.size == 0:
            if self.data_predicted == 1: 
                self.order = self.buy(size = int(min(self.broker.getcash(), 5000) / self.data_open))
            
            if self.data_predicted == -1:
                self.order = self.sell(size = int(min(self.broker.getcash(), 5000) / self.data_open))  
                    
        elif self.position.size > 0:
            if self.data_predicted == 1: # add more long position                
                # long
                self.order = self.buy(size = int(min(self.broker.getcash(), 5000) / self.data_open))
                
            elif self.data_predicted == -1:
                self.order = self.sell(size = int(min(self.broker.getcash(), 5000) / self.data_open) + self.position.size)
                
            else:
                # signal vanish
                self.order = self.sell(size = self.position.size)
                
        else: # self.position.size < 0
            if self.data_predicted == -1:
                self.order = self.sell(size = int(min(self.broker.getcash(), 5000) / self.data_open))
                
            elif self.data_predicted == 1:  # signal vanish
                self.order = self.sell(size = int(min(self.broker.getcash(), 5000) / self.data_open) - self.position.size)
            else:
                self.order = self.buy(size = -1*self.position.size)


def backtest(strategy, kline_data, predictions, start_date=pd.to_datetime('2018-01-01'), end_date=pd.to_datetime('2022-12-30')):
    cerebro = bt.Cerebro() 

    df = kline_data

    # model prediction
    df['predicted'] = predictions

    # add data
    feeddata_df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'predicted']].set_index(pd.to_datetime(df['Open time']), drop=True)[start_date:end_date]

    data = PandasData_ml(dataname=feeddata_df, fromdate=start_date, todate=end_date)

    cerebro.adddata(data)
    cerebro.addstrategy(strategy)
    cerebro = add_analyzers(cerebro)

    strats = cerebro.run()
    
    return analyze_strat(strats[0])