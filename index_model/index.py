import datetime as dt
import os

from pathlib import Path

from pathlib import Path
import pandas as pd
import numpy as np


class IndexModel:
    def __init__(self) -> None:
        # Initialize Variables:

        # Folder Paath
        self.fld = Path(__file__).parent.parent

        # Stock Price Data
        self.stock_price_df = pd.read_csv(os.path.join(self.fld, 'data_sources', 'stock_prices.csv'))
        self.stock_price_df['Date'] = pd.to_datetime(self.stock_price_df['Date'], format='%d/%m/%Y')
        self.stock_price_df.set_index('Date', inplace=True)

        # Weight Data Frame
        self.weight_df = self.get_weights()

        # Index Data
        self.index_df = None
        pass

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:
        """
        Calculates index level based on provided date ranges.
        :param start_date: dt.date
        :param end_date: dt.date
        """

        # Calculate portfolio val (Weight T)
        index_df_a = self.stock_price_df * self.weight_df
        index_df_a.dropna(inplace=True)
        index_df_a['Portfolio Value'] = index_df_a.sum(axis=1)
        index_df_a = index_df_a[index_df_a.index >= pd.to_datetime(start_date)]
        index_df_a = index_df_a[index_df_a.index <= pd.to_datetime(end_date)]

        # Calculate portfolio val (Weight T-1) - To calculate weight for rebalance date
        index_df_b = self.stock_price_df * self.weight_df.shift(1)
        index_df_b.dropna(inplace=True)
        index_df_b['Portfolio Value T-1 Weight'] = index_df_b.sum(axis=1)
        index_df_b = index_df_b[index_df_b.index >= pd.to_datetime(start_date)]
        index_df_b = index_df_b[index_df_b.index <= pd.to_datetime(end_date)]
        rebal_dates = index_df_b.resample('BMS').first().index
        index_df_b = index_df_b[index_df_b.index.isin(rebal_dates)]
        index_df = index_df_a.join(index_df_b[['Portfolio Value T-1 Weight']])

        # Calculate rebalance weight
        index_df['Rebalance Weight'] = index_df['Portfolio Value T-1 Weight']/index_df['Portfolio Value']
        index_df['Rebalance Weight'] = index_df['Rebalance Weight'].fillna(method='ffill')
        index_df['Rebalance Weight'].fillna(1, inplace=True)

        # Calculate Portfolio Return
        index_df['Portfolio Return'] = np.where(index_df['Portfolio Value T-1 Weight'].notnull(),
                                                (index_df['Portfolio Value']*index_df['Rebalance Weight'] - index_df['Portfolio Value'].shift(1)) /
                                                index_df['Portfolio Value'].shift(1),
                                                (index_df['Portfolio Value'] - index_df['Portfolio Value'].shift(1)) / index_df['Portfolio Value'].shift(1))
        index_df['Portfolio Return'].fillna(0, inplace=True)

        # Calculate Index Level
        index_df['Index Level'] = 100*(1 + index_df['Portfolio Return']).cumprod()
        self.index_df = index_df[['Index Level']]

        pass

    def get_weights(self) -> pd.DataFrame():
        """
        returns stock weights for every month based on the end of previous month stock prices
        :return: dataframe containing weights
        """
        # get dataframe containing last business day of month
        last_month_df = self.stock_price_df.resample('BM').last()

        # Select Top 3 Stock Prices. Change the rest to nan. Assign 50% to the greatest 1, 25% for second and third
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
        """
        export index values to the chosen file
        :param file_name: str
        """
        # file destination
        file_dest = os.path.join(self.fld, file_name)

        # create file
        self.index_df.to_csv(file_dest)
        pass


if __name__ == "__main__":
    a = IndexModel()
    #print(a.calc_index_level(dt.date(year=2020, month=1, day=1), dt.date(year=2020, month=12, day=31)))