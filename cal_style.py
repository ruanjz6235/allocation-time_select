import pandas as pd
import numpy as np
from app.repository.stock_style import *
from app.utils import DbUtil, DbHandleUtil
from app.utils.RedisUtil import RedisUtil
from app.repository.SecuMain import get_codes

redis_conn = RedisUtil.get_redis_client()
jy_conn = DbUtil.get_conn('jy')
zj_conn = DbUtil.get_conn('zj')
fd_conn = DbUtil.get_conn('fd')
ms = [('大盘-价值型', 'BH'), ('大盘-成长型', 'BL'), ('小盘-价值型', 'SH'), ('小盘-成长型', 'SL'), ('大盘-平衡型', 'BM'),
      ('小盘-平衡型', 'SM'), ('中盘-价值型', 'MH'), ('中盘-成长型', 'ML'), ('中盘-平衡型', 'MM')]
ms_vs = [x[::-1] for x in ms]


def get_barra(date, codes=None):
    barra_query = """select SecuCode code, beta, momentum, size, earnings_yield, residual_vol, growth, book_to_price,
    leverage, liquidity, non_linear_size from FM_FactorExposure where TradingDay = '{date}'"""
    if not codes:
        sub_query = ''
    else:
        sub_query = """ and SecuCode in ({codes})""".format(codes=str(codes)[1:-1])
    barra_data = pd.read_sql(barra_query.format(date=date) + sub_query, zj_conn)
    return barra_data


def get_ind(date, codes=None):
    inds = ', '.join([f'ind_{x}' for x in range(28)])
    ind_query = """select SecuCode code, {} from FM_FactorExposure""".format(inds) + """ where TradingDay = '{date}'"""
    if not codes:
        sub_query = ''
    else:
        sub_query = """ and SecuCode in ({codes})""".format(codes=str(codes)[1:-1])
    ind_data = pd.read_sql(ind_query.format(date=date) + sub_query, zj_conn)
    return ind_data


def get_ms(date, codes=None):
    date = pd.to_datetime(date) - pd.offsets.QuarterEnd()
    ms_query = """select stock_code code, style from stock_style where end_date = '{date}'"""
    if not codes:
        sub_query = ''
    else:
        sub_query = """ and stock_code in ({codes})""".format(codes=str(codes)[1:-1])
    ms_data = pd.read_sql(ms_query.format(date=date) + sub_query, fd_conn)
    data = DataTransform(ms_data)
    return data.clear_data('code.str.len() == 6').get_dummy(ms)


class DataTransform:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getattr__(self, item):
        if item not in self.__dict__.keys():
            return getattr(self.df, item)
        else:
            return self.__dict__[item]

    def __getitem__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        return self.df.__repr__()

    def get_dummy(self, columns=None):
        if not columns:
            columns = self.df['style'].drop_duplicates().tolist()
            columns = dict(zip(columns, columns))
        if self.df.empty:
            return pd.DataFrame(columns=['code'] + list(dict(columns).values()))
        self.df['style'] = self.df['style'].apply(lambda x: dict(columns)[x])
        df_ = pd.get_dummies(self.df['style'])
        self.df = pd.concat([self.df['code'], df_], axis=1)
        return self

    def rename(self, columns):
        self.df.rename(columns=columns)

    def save(self, table_name):
        bind, table = table_name.split('.')
        DbHandleUtil.save(table_name, self.df, bind)

    def clear_data(self, *conds):
        for cond in conds:
            self.df = self.df.query(cond)
        return self


class StockStyle:
    def __init__(self):
        self.barra_style = None
        self.ind_style = None
        self.ms_style = None
        self._codes = None

    @property
    def codes(self):
        if self._codes:
            return self._codes
        self._codes = redis_conn.get('codes')
        if not self._codes:
            self._codes = get_codes()['code'].tolist()
        return self._codes

    @codes.setter
    def codes(self, value):
        self._codes = value

    def get_barra_style(self, date, codes=None):
        self.barra_style = redis_conn.get(f'barra_{date}')
        if not self.barra_style:
            self.barra_style = get_barra(date, codes)
        return self.barra_style

    def get_ind_style(self, date, codes=None):
        self.ind_style = redis_conn.get(f'ind_{date}')
        if not self.ind_style:
            self.ind_style = get_ind(date, codes)
        return self.ind_style

    def get_ms_style(self, date, codes=None):
        self.ms_style = redis_conn.get(f'ms_{date}')
        if not self.ms_style:
            self.ms_style = get_ms(date, codes)
        return self.ms_style


class FundStyle(StockStyle):
    """
    入参：
    holding: columns = ['fund', 'date', 'code', 'ratio_nv'] + kwargs.get('style_list')
    """
    def __init__(self, holding):
        super().__init__()
        self.holding = holding

    def get_fund_style(self, style_type, date):
        style = getattr(self, f'get_{style_type}_style')(date)
        if style.empty:
            return pd.DataFrame()
        style_list = style.columns[~style.columns.isin(['code'])].tolist()
        self.holding = self.holding.merge(style, on='code', how='inner')
        holding_ratio = self.holding.groupby('fund')['ratio_nv'].sum()
        self.holding = pd.concat([self.holding.set_index('fund'), holding_ratio], axis=1)
        self.holding['ratio'] = self.holding['ratio_nv_x'] / self.holding['ratio_nv_y']
        def get_w_sum(x, col):
            return pd.Series(data=(x['ratio'].values * x[col].values).sum(axis=0), index=col)
        holding = self.holding.reset_index().groupby(['fund', 'date']).apply(get_w_sum, col=style_list).reset_index()
        return holding.unstack()


def get_fund_stocks(date):
    fund_stocks_query = """select ZSCode fund, EndDate date, SecuCode code, MarketInTA ratio_nv from ZS_FUNDVALUATION
    where EndDate = to_date('{}', 'yyyy-mm-dd') and FirstClassCode = '%s' and SecuCode is not null""".format(date)
    return pd.read_sql(fund_stocks_query, fd_conn)


def cal_fund_style(request_id, type, **kwargs):
    input_in = kwargs.get('input')
    output_out = kwargs.get('output')
    input_out = ['fund', 'date', 'code', 'ratio_nv']
    output_in = ['fund', 'date', 'style', 'ratio_nv']
    dates = pd.date_range('2010-01-01', pd.to_datetime('today').normalize())
    for date in dates:
        fund_stocks = get_fund_stocks(date.strftime('%Y-%m-%d')).rename(columns=dict(zip(input_in, input_out)))
        if fund_stocks.empty:
            continue
        fs = FundStyle(fund_stocks)
        holding = fs.get_fund_style('ms', date)
        holding = holding.rename(columns=dict(zip(output_in, output_out)))
        DbHandleUtil.save('fund_style', holding, 'funddata')








