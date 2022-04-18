import pandas as pd
import numpy as np
from app.utils import *
import feather
import warnings
warnings.filterwarnings('ignore')


class DataTransform:
    def __init__(self, df):
        self.df = df

    def __getattr__(self, item):
        if item not in self.__dict__.keys():
            return getattr(self.df, item)
        else:
            return self.__dict__[item]


class BaseStockCost:
    """
    基础函数集合
    """
    def __init__(self):
        self.date_name = 'date'
        self.time_name = 'time'
        self.code_name = 'code'
        self.close_name = 'close'
        self.price_name = 'price'
        self.volume_name = 'volume'
        self.flag_name = 'flag'
        self.hold_name = 'holding'
        self.days_name = 'days'
        self.dur_name = 'duration'
        self.hold_mv_name = 'holding_mv'

    @staticmethod
    def assert_columns(df, name):
        df_name = [x for x in df.columns if (x.lower().find(name) + 1)]
        if len(df_name) == 0:
            return df
        else:
            return df.rename(columns={df_name[0]: name})

    def agg_func(self, df):
        volume = df[self.volume_name].sum()
        amount = (df[self.volume_name] * df[self.price_name]).sum()
        price = amount / volume
        return pd.Series([volume, price, amount], index=['volume', 'amount', 'price'])

    def volume_plus(self, df):
        return df[self.volume_name] > 0

    def volume_minus(self, df):
        return df[self.volume_name] < 0

    def volume_zero(self, df):
        return df[self.volume_name] == 0

    def volume_all(self, df):
        return df[self.volume_name] > - 1e12

    @staticmethod
    def hold_type(df, deal):
        return df['hold_type'] == deal

    @staticmethod
    def deal_type(df):
        return df['hold_type'].isna()

    @staticmethod
    def type_one(df):
        return df['type'] == 1

    @staticmethod
    def type_two(df):
        return df['type'] == 2

    def deal_factor(self, day_amount, day_trading, cond1, cond2, cond3):
        deals = day_amount[cond1 & cond2]
        if len(deals) == 0:
            deals_new = pd.DataFrame(columns=day_trading.columns.tolist() + ['type'])
        else:
            deal_codes = deals.index.to_list()
            deals_new = day_trading[day_trading[self.code_name].isin(deal_codes)]
            deals_new = deals_new.groupby(self.code_name).apply(self.get_day_in, cond=cond3).reset_index(drop=True)
        return deals_new

    def get_day_in(self, deal_buy, cond):
        # columns=['time', 'volume', 'price']
        if cond == 'one':
            deal_buy['type'] = 2
            return deal_buy
        elif cond in ['rn', 'dx', 'pt', 'pg', 'qz', 'hg']:
            deal_buy['type'] = 1
            return deal_buy
        if cond == 'buy':
            b_i = 1
        else:
            b_i = - 1
        # 划分日内回转交易和留底仓交易
        sell_deal = deal_buy[deal_buy[self.volume_name] * b_i < 0]
        if len(sell_deal) == 0:
            deal_buy['type'] = 2
        else:
            s_num = - b_i * sell_deal[self.volume_name].sum()
            buy_deal = deal_buy[deal_buy[self.volume_name] * b_i > 0]
            buy_deal['cum'] = buy_deal[self.volume_name].cumsum()
            one = buy_deal[buy_deal['cum'] * b_i <= s_num]
            if one.empty:
                b_num = 0
            else:
                b_num = (b_i * one['cum']).max()
            two = buy_deal[buy_deal['cum'] * b_i > s_num]
            if s_num > b_num:
                deal = two.iloc[:1]
                one = pd.concat([one, deal])
                one.loc[one.index == deal.index[0], self.volume_name] = b_i * (s_num - b_num)
                two.loc[two.index == deal.index[0], self.volume_name] = two[self.volume_name] - b_i * (s_num - b_num)
            # 通过type字段标记
            sell_deal['type'] = 1
            one['type'] = 1
            two['type'] = 2
            deal_buy = pd.concat([sell_deal, one, two]).sort_values('time').reset_index(drop=True)
        return deal_buy


# 默认优先级顺序为股息，日内，打新，非日内，非日内分为已实现和未实现，已实现和未实现均为新增已实现和新增未实现
class StockCost(BaseStockCost):
    """
    分解各策略收益，但不包括风格收益分解，可以计算内容：打新收益，已实现和未实现收益，平均持仓周期，持仓成本，持仓平均收益
    事先需要定义好各dataframe的columns
    1. self.trading: columns = ['date', 'time', 'flag', 'code', 'price', 'volume']
    2. self.columns = self.trading.columns
    3. day_trading: columns = ['date', 'time', 'flag', 'code', 'price', 'volume', 'close']
    4. self.trading_data / self.trading_data_all, 记录普通交易: columns = ['date', 'time', 'code', 'price', 'volume']
    5. self.holding_data, 区分打新、配股、红股、和普通: columns = ['date', 'hold_type', 'code', 'price', 'close', 'volume']
    6. self.trading_assert: self.holding_data的详细版, 同样区分打新、配股、红股、和普通:
       columns = ['date', 'time', 'hold_type', 'code', 'price', 'close', 'volume']
    7. self.stock_close: columns=['date', 'code', 'close']
    另外需要注意：
    有date入参的，一般是self.trading_assert和xxx_ret这种截面更新的数据，无date入参的则是累计数据更新，如self.holding_data
    """
    def __init__(self, stock_close: pd.DataFrame, trading: pd.DataFrame):
        super().__init__()
        self.trading = trading
        self.stock_close = stock_close
        self.trading_assert = pd.DataFrame(columns=[self.date_name, self.time_name, 'hold_type', self.code_name,
                                                    self.volume_name, self.price_name, self.close_name])
        names = [x for x in self.__dict__.keys() if x.find('name') >= 0]
        for data_name in ['trading', 'trading_assert', 'stock_close']:
            for name in names:
                self.__dict__[data_name] = self.assert_columns(self.__dict__[data_name], self.__dict__[name])
        self.columns = self.trading.columns.tolist() + [self.close_name]

    # 1*******每日更新交易数据
    def update_trading_data(self, day_trading):
        # assert set(day_trading.columns) == set(columns)
        DbHandleUtil.save('trading_data_all', day_trading, 'rjz')
        agg_trading = day_trading.groupby([self.code_name, self.date_name]).apply(self.agg_func)
        DbHandleUtil.save('trading_data', agg_trading, 'rjz')

    # 2*******每日更新股息数据
    def update_gx_data(self, day_trading):
        # 股息
        gx = day_trading[day_trading[self.flag_name].isin(['4018'])]
        gx_ret = pd.DataFrame(gx.groupby(self.code_name)[self.price_name].sum().rename('gx'))
        new = day_trading[~day_trading[self.flag_name].isin(['4018'])]
        return gx_ret, new

    # 3*******每日更新日内交易
    def update_rn_data(self, day_trading):
        bs_data = day_trading[day_trading[self.flag_name].isin(['4001', '4002'])]
        bs_data = bs_data.sort_values([self.date_name, self.time_name]).reset_index(drop=True)
        bs_data = self.extract_rn(bs_data)
        type_one = self.type_one(bs_data)
        type_two = self.type_two(bs_data)
        rn_data = bs_data[type_one]
        frn_data = bs_data[type_two]
        if len(rn_data) == 0:
            rn_ret = pd.DataFrame(columns=[self.code_name, 'rn']).set_index(self.code_name)
        else:
            rn_ret = pd.DataFrame(rn_data.groupby(self.code_name).apply(
                lambda x: - sum(x[self.volume_name] * x[self.price_name])).rename('rn'))
        del frn_data['type']
        return rn_ret, frn_data

    def extract_rn(self, day_trading):
        # columns=['time', 'code', 'volume', 'price']
        day_trading['count'] = 1
        day_amount = day_trading.groupby(self.code_name)[[self.volume_name, 'count']].agg(
            {self.volume_name: 'sum', 'count': 'count'})
        count1 = day_amount['count'] == 1
        volume_plus = self.volume_plus(day_amount)
        volume_minus = self.volume_minus(day_amount)
        volume_zero = self.volume_zero(day_amount)
        volume_all = self.volume_all(day_amount)
        one_deal = self.deal_factor(day_amount, day_trading, count1, volume_all, 'one')
        two_deal_buy = self.deal_factor(day_amount, day_trading, ~count1, volume_plus, 'buy')
        two_deal_sell = self.deal_factor(day_amount, day_trading, ~count1, volume_minus, 'sell')
        two_deal_rn = self.deal_factor(day_amount, day_trading, ~count1, volume_zero, 'rn')
        day_data = pd.concat([one_deal, two_deal_buy, two_deal_sell, two_deal_rn]).reset_index(drop=True)
        return day_data

    # 4*******每日更新打新交易
    def update_dx_data(self, day_trading, deal):
        # if len(self.trading_assert[self.trading_assert['hold_type'] == deal]) == 0:
        #     return pd.DataFrame(columns=[self.code_name, self.date_name, 'dx']), day_trading
        day_trading = day_trading.sort_values([self.date_name, self.time_name]).reset_index(drop=True)
        dx_data, fdx_data1 = self.extract_dx(day_trading, deal)

        type_one = self.type_one(dx_data)
        type_two = self.type_two(dx_data)
        hold_type = self.hold_type(dx_data, deal)
        deal_type = self.deal_type(dx_data)
        dx_data1 = dx_data[type_one]
        dx_holding = dx_data[type_two & hold_type]
        fdx_data2 = dx_data[type_two & deal_type]

        if len(dx_data1) == 0:
            dx_ret = pd.DataFrame(columns=[self.code_name, f'{deal}']).set_index(self.code_name)
        else:
            dx_ret = pd.DataFrame(dx_data1.groupby(self.code_name).apply(
                lambda x: - sum(x[self.volume_name] * x[self.price_name])).rename(f'{deal}'))
        self.trading_assert = pd.concat([self.trading_assert[~self.hold_type(self.trading_assert, deal)],
                                         dx_holding[self.trading_assert.columns]])
        fdx_data = pd.concat([fdx_data1, fdx_data2])[day_trading.columns]
        return dx_ret, fdx_data

    def extract_dx(self, day_trading, deal):
        dx_cond1 = self.hold_type(self.trading_assert, deal)
        last_dx = self.trading_assert[dx_cond1]

        dx_codes = last_dx[self.code_name].drop_duplicates().tolist()
        flag_cond = day_trading[self.flag_name] == '4001'
        dx_cond2 = day_trading[self.code_name].isin(dx_codes)
        today_dx = day_trading[flag_cond & dx_cond2]

        dx_data = pd.concat([last_dx, today_dx])
        dx_agg = pd.DataFrame(dx_data.groupby(self.code_name)[self.volume_name].sum())
        volume_plus = self.volume_plus(dx_agg)
        volume_minus = self.volume_minus(dx_agg)
        volume_zero = self.volume_zero(dx_agg)
        volume_all = self.volume_all(dx_agg)

        two_deal_buy = self.deal_factor(dx_agg, dx_data, volume_plus, volume_all, 'buy')
        two_deal_sell = self.deal_factor(dx_agg, dx_data, volume_minus, volume_all, 'sell')
        two_deal_dx = self.deal_factor(dx_agg, dx_data, volume_zero, volume_all, deal)
        dx_data = pd.concat([two_deal_buy, two_deal_sell, two_deal_dx]).reset_index(drop=True)

        flag_cond2 = day_trading[self.flag_name] == '4002'
        dx_cond3 = ~day_trading[self.code_name].isin(dx_codes)
        fdx_data = day_trading[flag_cond2 | dx_cond3]
        # dx_data.columns = self.columns + ['hold_type', 'flag', 'type'], fdx_data.columns = self.columns + ['flag']
        return dx_data, fdx_data

    # 5*******每日更新持仓（截断交易持仓），非普通交易
    def update_trading_assert1(self, day_trading):
        # 打新
        dx = day_trading[day_trading[self.flag_name].isin(['4016'])]
        dx['hold_type'] = 'dx'
        dx[self.close_name] = dx[self.price_name]
        # 配股
        pg = day_trading[day_trading[self.flag_name].isin(['4013'])]
        pg['hold_type'] = 'pg'
        pg[self.close_name] = pg[self.price_name]
        # 红股
        hg = day_trading[day_trading[self.flag_name].isin(['4015'])]
        hg['hold_type'] = 'hg'
        hg[self.price_name] = 0
        # 权证
        qz = day_trading[day_trading[self.flag_name].isin(['4017'])]
        qz['hold_type'] = 'qz'
        qz[self.price_name] = 0
        trading_assert = pd.concat([dx, pg, hg, qz])[self.trading_assert.columns]
        trading_assert[self.close_name] = trading_assert[self.close_name].fillna(trading_assert[self.price_name])
        return trading_assert

    # 6*******每日更新持仓，普通交易
    def update_trading_assert2(self, day_trading, date):
        pt_cond = self.trading_assert['hold_type'] == 'pt'
        last_holding = self.trading_assert[pt_cond]
        today_holding = pd.concat([last_holding, day_trading])
        hold_agg = pd.DataFrame(today_holding.groupby(self.code_name)[self.volume_name].sum())
        volume_plus = self.volume_plus(hold_agg)
        volume_minus = self.volume_minus(hold_agg)
        volume_zero = self.volume_zero(hold_agg)
        volume_all = self.volume_all(hold_agg)

        two_deal_buy = self.deal_factor(hold_agg, today_holding, volume_plus, volume_all, 'buy')
        two_deal_sell = self.deal_factor(hold_agg, today_holding, volume_minus, volume_all, 'sell')
        two_deal_pt = self.deal_factor(hold_agg, today_holding, volume_zero, volume_all, 'pt')
        trading_assert = pd.concat([two_deal_buy, two_deal_sell, two_deal_pt]).reset_index(drop=True)
        pt_holding = trading_assert[trading_assert['type'] == 2]
        self.trading_assert = pd.concat([self.trading_assert[~self.hold_type(self.trading_assert, 'pt')],
                                         pt_holding[self.trading_assert.columns]])
        self.trading_assert['hold_type'] = self.trading_assert['hold_type'].fillna('pt')
        pt = trading_assert[trading_assert['type'] == 1]
        if len(pt) == 0:
            pt_ret = pd.DataFrame(columns=[self.code_name, 'pt']).set_index(self.code_name)
        else:
            pt_ret = pd.DataFrame(pt.groupby(self.code_name).apply(
                lambda x: - sum(x[self.volume_name] * x[self.price_name])).rename('pt'))
        pt_ret[self.date_name] = date
        return pt_ret

    # 7*******每日更新未实现收益，持仓周期
    def update_realize_return(self, trading_assert, day_close, date):
        del self.trading_assert[self.close_name]
        self.trading_assert = self.trading_assert.merge(day_close[[self.code_name, self.close_name]], on=self.code_name, how='left')
        self.trading_assert[self.close_name] = self.trading_assert[self.close_name].fillna(self.trading_assert[self.price_name])
        self.trading_assert = pd.concat([self.trading_assert, trading_assert])
        if self.trading_assert.empty:
            realize_return = pd.DataFrame(columns=[self.code_name, 'hold_type', 'ret'])
            holding = pd.DataFrame(columns=[self.code_name, self.hold_name, self.dur_name])
        else:
            realize_return = self.trading_assert.groupby([self.code_name, 'hold_type']).apply(
                lambda x: sum(x[self.volume_name] * (x[self.close_name] - x[self.price_name]))).rename('ret').reset_index()
            holding = self.get_hold_days(day_close, date)
        realize_return = realize_return.pivot(columns='hold_type', index=self.code_name, values='ret')
        realize_return.columns = realize_return.columns + '_realize'
        return realize_return, holding

    def get_hold_days(self, day_close, date):
        self.trading_assert[self.days_name] = (date - self.trading_assert[self.date_name]).apply(lambda x: x.days)
        holding = self.trading_assert.groupby(self.code_name)[self.volume_name].sum().rename(self.hold_name)
        def get_dur_data(x):
            try:
                return sum(x[self.volume_name] * x[self.days_name]) / sum(x[self.volume_name])
            except:
                return 0
        duration = self.trading_assert.groupby(self.code_name).apply(get_dur_data).rename('duration')
        holding = pd.concat([holding, duration], axis=1)
        close = day_close[[self.code_name, self.close_name]].set_index(self.code_name)
        holding = pd.concat([holding, close], axis=1)
        holding[self.hold_mv_name] = holding[self.hold_name] * holding[self.close_name]
        del self.trading_assert[self.days_name]
        return holding

    # 每日更新收益分解
    def deal_decompose(self):
        dates = self.trading[self.date_name].drop_duplicates().tolist()
        res = []
        for date in dates:
            print(date)
            day_deal = self.trading[self.trading[self.date_name] == date]
            day_close = self.stock_close[self.stock_close[self.date_name] == date]
            day_deal = day_deal.merge(day_close, on=[self.code_name, self.date_name], how='outer')
            gx_ret, day_trading = self.update_gx_data(day_deal)
            rn_ret, day_trading = self.update_rn_data(day_trading)
            dx_ret, day_trading = self.update_dx_data(day_trading, 'dx')
            hg_ret, day_trading = self.update_dx_data(day_trading, 'hg')
            pg_ret, day_trading = self.update_dx_data(day_trading, 'pg')
            qz_ret, day_trading = self.update_dx_data(day_trading, 'qz')
            trading_assert1 = self.update_trading_assert1(day_deal)
            pt_ret = self.update_trading_assert2(day_trading, date)
            realize_return, holding = self.update_realize_return(trading_assert1, day_close, date)
            realized_return = pd.concat([gx_ret, rn_ret, dx_ret, pg_ret, qz_ret, pt_ret, hg_ret], axis=1)
            ret = pd.concat([realize_return, realized_return, holding], axis=1).reset_index()
            ret[self.date_name] = date
            res.append(ret)
        res = pd.concat(res)
        del res['index']
        return res


def test_deal_decompose():
    zs_codes = get_zs_codes()
    stock_close = get_stock_close()
    for zs_code in zs_codes:
        trading = get_trading(zs_code)
        sc = StockCost(stock_close, trading)
        res = sc.deal_decompose()
        res['fund'] = zs_code


def test_hflhzx21():
    trading = get_trading(1)
    stock_close = feather.read_dataframe(r'/Users/kai/Desktop/1800040558/stock_close.feather')
    sc = StockCost(stock_close, trading)
    res = sc.deal_decompose()
    res['fund'] = '幻方量化专享21号1期'
    codes = sc.trading['code'].drop_duplicates().tolist()
    stock_ind = get_stock_ind(codes)
    res = res.merge(stock_ind, on='code', how='outer')
    realized_name = ['gx', 'rn', 'dx', 'hg', 'pg', 'qz', 'pt']
    realize_name = ['pt_realize', 'hg_realize', 'dx_realize', 'qz_realize', 'pg_realize']
    hold_name = ['holding', 'duration']
    res[[realized_name + realize_name + hold_name]] = res[[realized_name + realize_name + hold_name]].fillna(0)

    DbHandleUtil.save('ra_sm', res, 'rjz')


def get_trading(zs_code):
    # pd.read_csv(r'/Users/kai/Desktop/1800040558/1800040558.csv', encoding='gbk')
    return feather.read_dataframe(r'/Users/kai/Desktop/1800040558/1800040558.feather')


def get_stock_close():
    try:
        return feather.read_dataframe(r'/Users/kai/Desktop/1800040558/stock_close.feather')
    except:
        query = """select qt.TradingDay date, sm.SecuCode code, qt.ClosePrice close from JYDB.QT_Performance qt
        inner join JYDB.SecuMain sm on qt.InnerCode = sm.InnerCode where qt.TradingDay >= '2019-10-10'"""
        return pd.read_sql(query, DbUtil.get_conn())


def get_stock_ind(codes):
    query = f"""select secucode code, name_abbr name,  first_industry ind, info_date from funddata.stock_industry
    where secucode in ({str(codes)[1:-1]}) and standard = 38 and secumarket in (83, 90)"""
    ind = pd.read_sql(query, DbUtil.get_conn())
    ind_new = ind.groupby('code')['info_date'].max().rename('info').reset_index()
    ind = ind.merge(ind_new, on='code', how='outer')
    ind = ind[ind['info_date'] == ind['info']]
    return ind[['code', 'name', 'ind']]


# if __name__ == '__main__':
#     trading = get_trading(1)
#     stock_close = feather.read_dataframe(r'/Users/kai/Desktop/1800040558/stock_close.feather')
#     tradingdays = stock_close['date'].drop_duplicates().to_list()
#     sc = StockCost(stock_close, trading)
#     res = []
#     for date in dates:
#         print(date)
#         day_deal = sc.trading[sc.trading[sc.date_name] == date]
#         day_close = sc.stock_close[sc.stock_close[sc.date_name] == date]
#         day_deal = day_deal.merge(day_close, on=[sc.code_name, sc.date_name], how='outer')
#         # self.update_trading_data(day_trading, date)
#         gx_ret, day_trading = sc.update_gx_data(day_deal)
#         rn_ret, day_trading = sc.update_rn_data(day_trading)
#         dx_ret, day_trading = sc.update_dx_data(day_trading, 'dx')
#         pg_ret, day_trading = sc.update_dx_data(day_trading, 'pg')
#         hg_ret, day_trading = sc.update_dx_data(day_trading, 'hg')
#         qz_ret, day_trading = sc.update_dx_data(day_trading, 'qz')
#         trading_assert1 = sc.update_trading_assert1(day_deal)
#         pt_ret = sc.update_trading_assert2(day_trading, date)
#         realize_return = sc.update_realize_return(trading_assert1, day_close)
#         realized_return = pd.concat([gx_ret, rn_ret, dx_ret, hg_ret, pg_ret, qz_ret, pt_ret], axis=1)
#         ret = pd.concat([realize_return, realized_return], axis=1).reset_index()
#         ret[sc.date_name] = date
#         res.append(ret)
#     res = pd.concat(res)
#     res['fund'] = '幻方量化专享21号1期'




