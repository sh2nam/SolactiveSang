import datetime as dt
import os
import pandas as pd
import numpy as np


class IndexModel:
    def __init__(self) -> None:
        # Initialize Variables:

        # Folder Paath
        self.fld = os.path.dirname(os.getcwd())

        # Stock Price Data
        self.stock_price_df = pd.read_csv(os.path.join(self.fld, 'data_sources', 'stock_prices.csv'))
        self.stock_price_df['Date'] = pd.to_datetime(self.stock_price_df['Date'], format='%d/%m/%Y')
        self.stock_price_df.set_index('Date', inplace=True)

        # Weight Data Frame
        self.weight_df = self.get_weights()
        pass

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:

        # Calculate portfolio val
        index_df = self.stock_price_df * self.weight_df
        index_df.dropna(inplace=True)
        index_df['Portfolio Value'] = index_df.sum(axis=1)
        index_df = index_df[index_df.index >= pd.to_datetime(start_date)]
        index_df = index_df[index_df.index <= pd.to_datetime(end_date)]

        # Calculate Portfolio Return
        index_df['Portfolio Return'] = (index_df['Portfolio Value'] - index_df['Portfolio Value'].shift(1)) / index_df['Portfolio Value'].shift(1)
        index_df['Portfolio Return'].fillna(0, inplace=True)

        # Calculate Index
        index_df['Index Level'] = 100*(1 + index_df['Portfolio Return']).cumprod()

        return index_df

    def get_weights(self) -> pd.DataFrame():
        """

        :return: dataframe containing weights
        """
        last_month_df = self.stock_price_df.resample('M').last()

        # Select Top 3 Stock Prices. Change the rest to nan
        weight_df_a = last_month_df.where(last_month_df.apply(lambda x: x.eq(x.nlargest(3)), axis=1), np.nan)
        weight_df_a[weight_df_a.notnull()] = 0.25
        weight_df_a.fillna(0, inplace=True)
        weight_df_b = last_month_df.where(last_month_df.apply(lambda x: x.eq(x.nlargest(1)), axis=1), np.nan)
        weight_df_b[weight_df_b.notnull()] = 0.25
        weight_df_b.fillna(0, inplace=True)
        weight_df_lastmonth = weight_df_a+weight_df_b

        # shift weights to the first day of month
        weight_df = self.stock_price_df * weight_df_lastmonth / self.stock_price_df
        weight_df = weight_df.shift(1)
        weight_df.fillna(method='ffill', inplace=True)
        weight_df.dropna(inplace=True)

        return weight_df

    def export_values(self, file_name: str) -> None:
        # To be implemented
        pass

if __name__ == "__main__":
    a = IndexModel()
    df = a.calc_index_level(dt.date(year=2020, month=1, day=1), dt.date(year=2020, month=12, day=31))
    df.to_csv(r'C:\temp\test.csv')