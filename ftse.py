import bs4 as bs
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import statistics

style.use('ggplot')

def save_ftse100_tickers():

    resp = requests.get('https://en.m.wikipedia.org/wiki/FTSE_100_Index')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.findAll('table', {'class': 'wikitable sortable'})
    #because of multiple table in wikipedia i am taking table number [2]
    table= table[1]
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        # use the ticker.replace here because to remove dot from wikipedia input
        ticker = (ticker.replace('.',''))
        # adding .L here because wikipedia tickers dont have .L (london stock exchange)
        ticker = str(ticker) + '.L'
        tickers.append(ticker)

    with open("ftse100tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers

#  read the wikipedia stock list and save ftse 100 tickers to pickle file
#save_ftse100_tickers()

def get_data_from_yahoo(reload_ftse100=False):

    if reload_ftse100:
        tickers = save_ftse100_tickers()
    else:
        with open("ftse100tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2020, 1, 12)
    end = dt.datetime(2020, 2, 12)

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

#get_data_from_yahoo()

def compile_data():

    with open("ftse100tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
            print(main_df.head())
    main_df.to_csv('ftse100_joined_closes.csv')

#compile_data()

def process_data_for_labels(ticker):
    hm_days = 7
    #df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    df = pd.read_csv('ftse100_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker_name for ticker_name in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)

    #clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])


    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print(ticker)
    print()
    return confidence

# from statistics import mean
#
# with open("sp500tickers.pickle","rb") as f:
#     tickers = pickle.load(f)
#
# accuracies = []
# for count,ticker in enumerate(tickers):
#
#     if count%10==0:
#         print(count)
#
#     accuracy = do_ml(ticker)
#     accuracies.append(accuracy)
#     print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))
do_ml('ADM.L')
#save_nifty50_tickers()
#get_data_from_yahoo_nifty50()
#compile_data_nifty50()
