### Main module to preprocess dataset

import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import CHARAS_LIST

class CharData:
    def __init__(self, datapath = "../datasets/dataset.csv", returnpath = "../datasets/month_ret.pkl"):
        self.datapath = datapath
        self.retpath = returnpath
        self.loadData()
        self.dates = self.getDates()
        self.joinRets()
        self.rankNormalize()
        self.loaders()

    def loadData(self):
        self.df = pd.read_csv(self.datapath)
        with open(self.retpath, "rb") as fp:
            self.ret = pkl.load(fp)

    def getDates(self):
        return list(self.df.DATE.unique())

    def rankNormalize(self):
        dfs = []
        for date in tqdm(self.dates):
            cross_slice = self.df.loc[self.df.DATE == date].copy(deep = False)
            omitted_mask = 1.0 * np.isnan(cross_slice.loc[cross_slice['DATE'] == date])
            cross_slice.loc[cross_slice['DATE'] == date] = cross_slice.fillna(0) + omitted_mask * cross_slice.median()
            cross_slice.loc[cross_slice['DATE'] == date] = cross_slice.fillna(0)
            re_df = []
            for col in CHARAS_LIST:
                series = cross_slice[col]
                de_duplicate_slice = pd.DataFrame(series.drop_duplicates().to_list(), columns=['chara'])
                series = pd.DataFrame(series.to_list(), columns = ['chara'])
                de_duplicate_slice['sort_rank'] = de_duplicate_slice['chara'].argsort().argsort()
                rank = pd.merge(series, de_duplicate_slice, left_on='chara', right_on='chara', how='right')['sort_rank']
                rank_normal = ((rank - rank.min())/(rank.max() - rank.min())*2 - 1)
                re_df.append(rank_normal)
            re_df = pd.DataFrame(re_df, index = CHARAS_LIST).T.fillna(0)
            re_df['permno'] = list(cross_slice['permno'].astype(int))
            re_df['DATE'] = list(cross_slice['DATE'].astype(int))
            dfs.append(re_df)
        self.df = pd.concat(dfs)
    
    def joinRets(self):
        months = list(self.df.DATE)
        months = [int(str(m)[:6]) for m in months]
        self.df['month'] = months
        self.df = self.df.drop(['DATE'], axis = 1)
        self.df = pd.merge(self.df, self.ret, how= "inner", on = ['permno', 'month'])
        self.df.drop(['date'], axis = 1)
        # self.df[['ret-rf']] = StandardScaler().fit_transform(self.df[['ret-rf']])

    def cal_portfolio_ret(self, it):
        d, f = it[0], it[1]
        long_portfolio = self.df.loc[self.df.DATE == d][['permno', f]].sort_values(by=f, ascending = False)[:self.df.loc[self.df.DATE == d].shape[0] // 10]['permno'].to_list()
        short_portfolio = self.df.loc[self.df.DATE == d][['permno', f]].sort_values(by=f, ascending = False)[-self.df.loc[self.df.DATE == d].shape[0]//10:]['permno'].to_list()
        long_ret = self.ret.loc[self.df.DATE == d].drop_duplicates('permno').set_index('permno').reindex(long_portfolio)['ret-rf'].dropna().mean()
        short_ret = self.ret.loc[self.df.DATE == d].drop_duplicates('permno').set_index('permno').reindex(short_portfolio)['ret-rf'].dropna().mean()
        chara_ret = 0.5*(long_ret - short_ret)
        return chara_ret
    
    def _makedata(self, df):
        characs = {}
        dates = list(df.month.unique())

        for dt in tqdm(dates):
            df_date = df[df.month == dt]
            df_date = df_date.drop('month', axis = 1)
            ids = list(df_date['permno'])
            df_date = df_date.drop('permno', axis = 1)
            y = np.array(list(df_date['ret-rf']))
            df_date = df_date.drop('ret-rf', axis = 1)
            df_date.fillna(-1000)
            characs[dt] = {
                "ids": np.array(ids),
                "characs" : df_date.to_numpy(),
                "returns" : y
            }
        return characs
    
    def loaders(self):
        df_train = self.df[(self.df['month'] >= 195700) & (self.df['month'] <= 197412)]
        df_valid = self.df[(self.df['month'] >= 197500) & (self.df['month'] <= 198612)]
        df_test = self.df[self.df['month'] >= 198700]

        df_train = df_train.drop('date', axis = 1)
        df_valid = df_valid.drop('date', axis = 1)
        df_test = df_test.drop('date', axis = 1)

        x_train = self._makedata(df_train)
        x_valid = self._makedata(df_valid)
        x_test = self._makedata(df_test)

        df_train.to_csv("train.csv", index = False)
        df_valid.to_csv("valid.csv", index = False)
        df_test.to_csv("test.csv", index = False)

        with open("x_train.pkl", "wb") as fp:
            pkl.dump(x_train, fp)

        with open("x_test.pkl", "wb") as fp:
            pkl.dump(x_test, fp)

        with open("x_valid.pkl", "wb") as fp:
            pkl.dump(x_valid, fp)



    

    




        




        