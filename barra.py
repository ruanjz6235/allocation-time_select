# 和基金评级一样，整套模型包括研究和生产，这里主要展示生产环境。
# 研究有一些常规框架（前7点），加上一些开放内容（后3点），包括：
# ①单因子对股票定价解释度的检验，信息比率，检验到哪些因子对于定价解释度高就选哪些因子；
# ②对于股票量价如何处理，缺失值是因为数据错误，还是停牌，涨跌停如何处理，ST股票如何处理；
# ③对于股票财务数据、一致预期数据等其他数据如何处理，缺失值用插值、回归、季度处理等何种方式；
# ④极值如何处理，极值由于比例失调产生，还是由于本身数据有错误，还是数据本身就是如此；
# ⑤哪些因子需要中性化，中性化的目的就是不要让某些因子在市值偏大，或者固定行业上有明显有偏差的暴露；
# ⑥哪些因子需要正交化，一旦AIC满足一定的阈值标准，即认定该因子需要正交化，滚动模型需要考虑到模型的鲁棒性；
# ⑦有哪些方法可以充分考虑到模型的鲁棒性，以及模型的不稳定性可以有哪些修正方法；
# ⑧新因子：A股的北向资金、抱团现象值得研究，同时要加大对极端负面的市场的研究；
# ⑨单一因子组合问题，Barra原模型给定固定比例，当然可以根据一些算法重新定义因子的组合；
# ⑩权重配比，拥挤度、景气度的收益风险因子挖掘。
#
# 数据来自于恒生聚源
# 关于数据预处理方面，主要有：
# 数据的爬取
# 股票代码和名称的处理
# 缺失日期的补全
# 缺失值填充（回归、插值、直接去掉）
# 极值处理（不处理、3σ原则、残差的3σ原则）
# 累计、单期、ttm的互相转换；日、月、季、年的频率转换
# 标准化
# 中性化（市值中性化、行业中性化）
# 正交化（高度相关的因子有盈利因子、杠杆因子，正交后这两个因子解释性一般，可以直接去掉；以及流动性、波动和贝塔之间存在较高的相关性）


import numpy as np
import pandas as pd
import warnings
import datetime as dt
import statsmodels.api as sm
import logging as logger
from functools import lru_cache, reduce, wraps
from pandarallel import pandarallel
from scipy.interpolate import interp1d
import pymysql
from configparser import ConfigParser
# %%
warnings.filterwarnings('ignore')
pandarallel.initialize(nb_workers=4)
BASIC_DATA = ['stocks_price', 'mv_trs', 'industry', 'fundamentals']
# fundamentals中包括EPS, NetAssetPS, BasicEPSYOY, DebtAssetsRatio
# 'Close', 'TurnoverRate', 'NegotiableMV', 'fundamentals', 'FirstIndustryName', 'SecondIndustryName'
MARKET_DATA = []
FINANCE_DATA = []
FACTOR_DATA = []
NAME_DATA = {'stocks_price': ['close', 'log_ret'], 'mv_trs': ['turnover_rate', 'negotiable_mv'],
             'industry': ['industry'], 'fundamentals': ['eps', 'naps', 'growth', 'leverage']}
NAME_DATA2 = {}
QUERY_DICT = {}
SENTINEL = 1e10
START_YEAR = 2009
END_YEAR = 2019
BENCHMARK = '000300'
CONFIG_NAME = ''
config = ConfigParser()
config.read('config.ini', encoding='gbk')
# CONF = dict(config.items(CONFIG_NAME))
CONF = {'host': 'dev.zhijuninvest.com', 'user': 'rjz', 'password': 'ruanjiazheng2021molcud1', 'port': 3306}


# %%
def time_decorator(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = dt.datetime.now()
        result = func(*args, **kwargs)
        end = dt.datetime.now()
        logger.info(f'“{func.__name__}” run time: {end - start}.')
        return result

    return timer


def config_decorator(func):
    @wraps(func)
    def configer(*args, **kwargs):
        conn = pymysql.connect(**CONF)
        result = func(*args, **kwargs, conn=conn)
        return result
    return configer


# %%
# 数据获取
# 并未获取所有所需数据
class BaseDataSelect:
    @staticmethod
    @config_decorator
    def get_fundamental(code, start, end, conn):
        query = """SELECT SM.SecuCode code, LCMIN.EndDate date, YEAR(LCMIN.EndDate) year, QUARTER(LCMIN.EndDate) quarter,
        LCMIN.EPS EPS_Raw, LCMIN.NetAssetPS naps, LCMIN.BasicEPSYOY growth, LCMIN.DebtAssetsRatio leverage
        FROM JYDB.LC_MainIndexNew LCMIN inner JOIN JYDB.SecuMain SM ON LCMIN.CompanyCode = SM.CompanyCode
        WHERE SM.SecuCode = '%s' AND SM.SecuCategory = 1 AND LCMIN.EndDate >= '%s' and LCMIN.EndDate <= '%s'
        ORDER BY EndDate asc""" % (code, start, end)
        fundamental = pd.read_sql(query, conn)
        return fundamental

    @staticmethod
    @config_decorator
    def get_industry(codes, conn):
        ind_list = ['银行', '房地产', '医药生物', '公用事业', '综合', '机械设备', '建筑装饰', '建筑材料', '家用电器', '汽车',
                    '食品饮料', '电子', '计算机', '交通运输', '轻工制造', '通信', '休闲服务', '传媒', '农林牧渔', '商业贸易',
                    '化工', '有色金属', '非银金融', '电气设备', '国防军工', '采掘', '纺织服装', '钢铁']
        query = """SELECT SM.SecuCode, LCEI.FirstIndustryName industry FROM JYDB.SecuMain SM inner JOIN JYDB.LC_ExgIndustry LCEI
        ON SM.CompanyCode = LCEI.CompanyCode WHERE SM.SecuCode in (%s) AND SM.SecuCategory = 1 AND LCEI.Standard = 24
        AND LCEI.CancelDate is NULL""" % (str(codes)[1:-1])
        industry = pd.read_sql(query, conn)
        industry_code = pd.DataFrame({'industry': ind_list, 'industry_code': np.arange(28)})
        industry = industry.merge(industry_code, on=['industry'], how='inner')
        del industry['industry_code']
        return industry

    @staticmethod
    @config_decorator
    def get_sub_stock_price(code, date, conn):
        query = """SELECT code, date, YEAR(date) year, QUARTER(date) quarter, close Price FROM funddata.stock_daily_quote
        WHERE code = '%s' AND date >= '%s' ORDER BY date asc""" % (code, date)
        stock_price = pd.read_sql(query, conn)
        return stock_price

    @staticmethod
    @config_decorator
    def get_mv_tr(codes, date, conn):
        query = """SELECT sm.SecuCode code, qt.TradingDay date, qt.TurnoverRate turnover_rate,
        qt.NegotiableMV negotiable_mv FROM JYDB.QT_Performance qt inner join JYDB.SecuMain sm WHERE sm.SecuCode in (%s)
        AND TurnoverRate != 0 AND TradingDay >= '%s'""" % (str(codes)[1:-1], date)
        mv_tr = pd.read_sql(query, conn)
        return mv_tr


# 数据预处理
class DataSelect(BaseDataSelect):
    def __init__(self):
        super().__init__()
        self.basic_data = BASIC_DATA
        self.market_data = MARKET_DATA
        self.finance_data = FINANCE_DATA
        self.factor_data = FACTOR_DATA
        self.name_data = NAME_DATA
        self.all_list = list(NAME_DATA.keys()) + list(np.hstack(NAME_DATA.values()))

    def __getattr__(self, name):
        if name not in self.__dict__:
            if name == 'tradingdays':
                self.__dict__[name] = getattr(self, 'get_' + name)()
            else:
                name = self.identify_name(name)
                upper_name = [key for key in self.name_data.keys() if name in self.name_data[key]][0]
                if upper_name not in self.__dict__:
                    codes = self.get_codes()
                    if upper_name == 'industry':
                        self.__dict__[upper_name] = getattr(self, 'get_' + upper_name)(codes)
                    elif upper_name == 'fundamentals':
                        self.__dict__[upper_name] = getattr(self, 'get_' + upper_name)(
                            codes, pd.to_datetime('today').normalize())
                    else:  # upper_name == 'mv_trs' or 'stock_price'
                        start, _ = self.get_last_run_date(), self.get_available_date()
                        if upper_name == 'mv_trs':
                            days = 251
                        else:
                            days = 525
                        dates = self.get_tradingdays()
                        date = dates[dates <= start].iloc[-days]
                        self.__dict__[upper_name] = getattr(self, 'get_' + upper_name)(date, codes)
                if 'date' in self.__dict__[upper_name].columns:
                    name_df = self.__dict__[upper_name][['code', 'date', name]]
                    self.__dict__[name] = name_df.pivot(columns='code', index='date', values=name)
                else:
                    self.__dict__[name] = self.__dict__[upper_name][['code', name]]
        return self.__dict__[name]

    def identify_name(self, name):
        if name not in np.hstack(self.all_list):
            all_list = pd.Series(np.hstack(self.all_list)).apply(lambda x: x.lower())
            name = name.lower()
            validation = all_list.str.contains(name)
            if len(all_list[validation]) == 0:
                msg = f"请确认因子名称{name}是否正确"
                raise IndexError(msg)
            else:
                name = all_list[validation].iloc[0]
        return name

    @lru_cache(maxsize=999)
    @config_decorator
    def get_codes(self, conn):
        query = """select SecuCode, ListedState, ListedSector from JYDB.SecuMain where SecuMarket in (83, 90)
        and SecuCategory = 1 and listedsector in (1, 2, 6, 7)"""
        return pd.read_sql(query, conn).SecuCode.tolist()

    @lru_cache(maxsize=999)
    @config_decorator
    def get_tradingdays(self, conn):
        query = """SELECT TradingDate as EndDate FROM JYDB.QT_TradingDayNew WHERE IfTradingDay = 1 AND SecuMarket = 83
        AND TradingDate >= '2015-01-01' AND TradingDate <= '{ed}' order by TradingDate asc""".format(
            ed=pd.to_datetime('today').normalize())
        return pd.read_sql(query, conn).EndDate

    @config_decorator
    def get_last_run_date(self, conn):
        query = """SELECT max(TradingDay) TradingDay FROM zj_data.FM_WLS_Beta"""
        return pd.read_sql(query, conn).TradingDay.iloc[0]

    @config_decorator
    def get_available_date(self, conn):
        query = """SELECT max(date) TradingDay FROM funddata.stock_daily_quote"""
        available_date = pd.read_sql(query, conn).TradingDay.iloc[0]
        if available_date < pd.to_datetime('today').normalize():
            logger.info('the newest stock price is not today')
        return available_date

    @staticmethod
    def get_quarter_end(start, end):
        return pd.date_range(start, end, freq='q')

    @staticmethod
    def codes_dates_df(x, y):
        xx = pd.DataFrame(x)
        xx['count'] = 1
        yy = pd.DataFrame(y)
        yy['count'] = 1
        xy = xx.merge(yy, on='count', how='outer')
        del xy['count']
        return xy

    @staticmethod
    def transfer_freq(df, freq_before='q', freq_after='q', *names):
        """收益率转换"""
        dt_name, bname, aname = names
        if freq_before != freq_after:
            if freq_before == 'd':
                if freq_after == 'm':
                    df['calendar'] = df[dt_name].apply(lambda x: x.strftime('%Y-%m'))
                elif freq_after == 'q':
                    df['calendar'] = df[dt_name].apply(lambda x: str(x.year)) + '-' + df[dt_name].apply(
                        lambda x: str(x.quarter))
                else:  # freq_after == 'y'
                    df['calendar'] = df[dt_name].apply(lambda x: str(x.year))
            elif freq_before == 'm':
                if freq_after == 'q':
                    df['calendar'] = df[dt_name].apply(lambda x: x[:-2]) + df[dt_name].apply(
                        lambda x: str((int(x) - 1) // 3 + 1))
                else:  # freq_after == 'y'
                    df['calendar'] = df[dt_name].apply(lambda x: x[:4])
            else:  # freq_before == 'q' & freq_after == 'y'
                df['calendar'] = df[dt_name].apply(lambda x: x[:4])
            df[aname] = df.groupby(dt_name)[bname].sum()
            return df

    @staticmethod
    def transfer_data(df, freq='q', start_state='cum', end_state='single', *names):
        dt_name, bname, aname = names
        if freq == 'q':
            if start_state == 'cum' and end_state == 'single':
                df[aname] = df[bname] - df[bname].shift(1)
                df['calendar'] = df[dt_name].apply(lambda x: str(x.year) + '-' + str(x.quarter))
                df.loc[df['calendar'].str.endswith('1'), aname] = df[bname]
            elif start_state == 'single' and end_state == 'cum':
                df['calendar'] = df[dt_name].apply(lambda x: x[:4])
                df[aname] = df.groupby('calendar')[bname].cumsum()
                del df['calendar']
            elif start_state == 'single' and end_state == 'ttm':
                df[aname] = df.rolling(4)[bname].sum()
            else:  # start_state == 'cum' and end_state == 'ttm'
                df[aname] = df[bname] - df[bname].shift(1)
                df['calendar'] = df[dt_name].apply(lambda x: str(x.year) + '-' + str(x.quarter))
                df.loc[df['calendar'].str.endswith('1'), aname] = df[bname]
                df[aname] = df.rolling(4)[aname].sum()
            return df
        else:
            return df

    @staticmethod
    def interpolation(series, kind=1, *args):
        """
        kind = 1: 线性插值
        kind = 2: 二次插值
        kind = 3: 三次插值
        kind = 0: 指数模拟
        """
        series_new = series.loc[(~(series.isna()) & (series.shift(-1).isna()))
                                | (~(series.isna()) & (series.shift(1).isna()))].iloc[1:-1]
        for i in range(len(series_new) // 2):
            sub_series = series_new.iloc[2 * i, 2 * (i + 1)]
            x = sub_series.index.tolist()
            y = series[x]
            x_new = list(range(x[0], x[1]))
            if kind == 1:
                func = interp1d(x, y, kind="linear")
                y_new = func(x_new)
            elif kind == 2:
                func = interp1d(x, y, kind="quadratic")
                y_new = func(x_new)
            elif kind == 3:
                func = interp1d(x, y, kind="cubic")
                y_new = func(x_new)
            else:  # 指数模拟
                func = interp1d(x, y, kind="linear")
                series_index = args[0][x_new]
                y_new = func(series_index)
            series.loc[(series.index > x[0]) & (series.index < x[1])] = y_new[1:-1]
        return series

    @staticmethod
    def fill_vals(series, val=None, method=None):
        valid_idx = np.argwhere(series.notna().values).flatten()
        try:
            series_valid = series.iloc[valid_idx[0]:]
        except IndexError:
            return series
        if val:
            series_valid = series_valid.fillna(val)
        elif method:
            series_valid = series_valid.fillna(method=method)
        else:
            median = np.nanmedian(series_valid)
            series_valid = series_valid.fillna(median)
        series = series.iloc[:valid_idx[0]].append(series_valid)
        return series

    @staticmethod
    def three_sigma(series, if_regress=False, *condition):
        if not if_regress:
            sigma, mean = series.std(), series.mean()
            series_valid = series.loc[(series <= mean + 3 * sigma) & (series >= mean - 3 * sigma)]
            sigma, mean = series_valid.std(), series_valid.mean()
            series.loc[series > mean + 3 * sigma] = mean + 3 * sigma
            series.loc[series < mean - 3 * sigma] = mean - 3 * sigma
            return series
        else:
            cond = condition[0]
            if len(series) != len(cond):
                raise ValueError('cannot regress because len(series) != len(cond)')
            cond = sm.add_constant(cond)
            result = sm.OLS(series, cond).fit()
            params, resid, r2 = result.params, result.resid, result.rsquared
            sigma = resid.std()
            valid_idx = np.argwhere(resid.loc[(resid <= 3 * sigma) & (resid >= - 3 * sigma)].values)
            invalid_high = np.argwhere(resid.loc[(resid > 3 * sigma)].values)
            invalid_low = np.argwhere(resid.loc[(resid < - 3 * sigma)].values)
            series_valid, cond_valid = series[valid_idx], cond[valid_idx]
            cond_valid = sm.add_constant(cond_valid)
            result = sm.OLS(series_valid, cond_valid).fit()
            params, resid, r2 = result.params, result.resid, result.rsquared
            sigma = resid.std()
            series.loc[series.index.isin(invalid_high)] = params[0] + params[1] * cond.loc[
                cond.index.isin(invalid_high)] + 3 * sigma
            series.loc[series.index.isin(invalid_low)] = params[0] + params[1] * cond.loc[
                cond.index.isin(invalid_low)] - 3 * sigma
        return series

    @staticmethod
    def standardize(series):
        sigma, mean = series.std(), series.mean()
        series = (series - mean) / sigma
        return series

    @staticmethod
    def neutralize(series, type_='ind', *condition):
        """type_: 行业中性化 or 市值中性化。分别对应ind和scale"""
        date = series.name
        cond = condition[0]
        df = pd.concat([pd.DataFrame(series), pd.DataFrame(cond[0])], axis=1).reset_index()
        if type_ == 'ind':
            df_ind = df.groupby(cond.name).mean().rename('mean').reset_index()
            df = df.merge(df_ind, on=cond.name, how='outer')
            df[date] = df[date] - df['mean']
            series = df.set_index('index')[date]
        else:  # type_ = 'scale'
            result = sm.OLS(df[date], df[cond.name]).fit()
            series = result.resid
        return series

    @staticmethod
    def align(df1, df2, *dfs, if_column=True):
        dfs_all = [df1, df2] + list(dfs)
        mut_date_range = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
        dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        if if_column:
            mut_codes = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
            dfs_all = [df.loc[:, mut_codes] for df in dfs_all]
        return dfs_all

    @staticmethod
    def for_print(i):
        if i % 500 == 0:
            logger.info(i)

    def complete_dates(self, df, freq='q', *names):
        """freq取'd', 'm', 'q', 'y', 't'"""
        sec_name, dt_name = names
        name = list(set(df.columns) - set(names))
        start, end = df[dt_name].min(), df[dt_name].max()
        if freq == 't':
            dates = self.tradingdays
        else:
            dates = pd.date_range(start, end, freq=freq)
        codes = df[sec_name].unique()
        codes_dates = self.codes_dates_df(codes, dates)
        codes_dates.columns = [sec_name, dt_name]
        df = codes_dates.merge(df, on=[sec_name, dt_name], how='outer').sort_values(
            [sec_name, dt_name]).reset_index(drop=True)
        if freq != 't':
            df[name] = df.groupby(sec_name)[name].apply(self.interpolation)
        return df

    def get_fundamentals(self, codes, end):
        end = pd.to_datetime(end)
        start = end - pd.offsets.QuarterEnd() - pd.DateOffset(months=39)
        fundamentals = list()
        for i, code in enumerate(codes):
            self.for_print(i)
            fundamental = self.get_fundamental(code, start, end)
            fundamentals.append(fundamental)
        fundamentals = pd.concat(fundamentals, ignore_index=True, sort=False)
        names = ('code', 'date')
        fundamentals = self.complete_dates(fundamentals, *names)
        args = ('q', 'cum', 'single', 'date', 'EPS_Raw', 'eps')
        fundamentals = fundamentals.groupby('code').apply(self.transfer_data, *args)
        return fundamentals

    def get_stocks_price(self, date, codes):
        stocks_price = list()
        for i, code in enumerate(codes):
            self.for_print(i)
            stock_price = self.get_sub_stock_price(code, date)
            stocks_price.append(stock_price)
        stocks_price = pd.concat(stocks_price)
        names = ('code', 'date')
        stocks_price = self.complete_dates(stocks_price, 't', *names)
        stocks_price['log_ret'] = stocks_price.groupby('code')['close'].apply(lambda x: np.log(x / x.shift(1)))
        return stocks_price
        # return stocks_price[~stocks_price['SecuCode'].str.startswith('688')]

    def get_mv_trs(self, date, codes):
        mv_trs = list()
        sub_codes = []
        for i, code in enumerate(codes):
            self.for_print(i)
            sub_codes.append(code)
            if i % 500 != 0 and i < len(codes) - 1:
                continue
            mv_tr = self.get_mv_tr(sub_codes, date)
            mv_trs.append(mv_tr)
            sub_codes = []
        mv_trs = pd.concat(mv_trs)
        names = ('code', 'date')
        mv_trs = self.complete_dates(mv_trs, 't', *names)
        return mv_trs


# 基本数据计算
class BasicDataProcess:
    @staticmethod
    def nanfunc(series, func, weights=None):
        series = series.values
        valid_idx = np.argwhere((series <= SENTINEL) & (series >= - SENTINEL) & (series.notna()))
        if weights is None:
            return func(series[valid_idx])
        else:
            weights = weights.values
            weights /= np.sum(weights)
            return func(series[valid_idx] * weights[valid_idx])

    @staticmethod
    def regress(y, x, thrshld=0.7, intercept=True, weight=1, if_interpolation=False):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if if_interpolation:
            y = y.interpolate()
        len_y, len_x = len(y), len(x)
        if len_x != len_y:
            raise ValueError('len(x) != len(y), please try again')
        # count_y等于1，count_y小于0.7，count_y介于1和0.7之间， 默认count_x等于1
        cnt_y, cnt_x = y.count() / len_y, x.count() / len_x
        def unit_regress(yy, xx, intcpt=True, wt=1):
            xxx = xx.copy()
            if intcpt:
                xxx = sm.add_constant(xxx)
            res = sm.WLS(yy, xxx, weights=wt).fit()
            return res.params, res.resid, res.rsquared
        fst, scd = cnt_y[(cnt_y == 1) & (cnt_y < thrshld)].index, cnt_y[(cnt_y < 1) & (cnt_y >= thrshld)].index
        params, resid, r2 = pd.DataFrame(), pd.DataFrame(), pd.Series()
        if len(fst) > 0:
            params, resid, r2 = unit_regress(y[fst], x, intercept, weight)
            params.columns = fst
        if len(scd) > 0:
            for code in scd:
                y2, x2 = DataSelect.align(y[code].dropna(), x, False)
                scd_params, scd_resid, scd_r2 = unit_regress(y2, x2, intercept, weight)
                scd_params = scd_params.rename(code)
                params = pd.concat([params, scd_params.params.to_frame()], axis=1)
                resid = pd.concat([resid, scd_resid], axis=1)
                r2 = pd.concat([r2, pd.Series({code: scd_r2})])
        return params, resid, r2

    @staticmethod
    def cal_cmra(series, months=12, period=21):
        z = sorted(series[-i * period:].sum() for i in range(1, months + 1))
        return z[-1] - z[0]

    def cal_midcap(self, series):
        x = series.dropna().values
        params = self.regress(x ** 3, x)
        beta, alpha = params.values
        return series ** 3 - (alpha + beta[0] * series)

    @staticmethod
    def cal_liquidity(series, period=21):
        freq = len(series) // period
        res = np.log(series.sum() / freq)
        return res

    def cal_growth_rate(self, series, period=21, remain=0):
        series_new = series.loc[series.index % period == remain]
        y, x = series_new.values, np.array(series_new.index)
        beta, _, _ = self.regress(y, x)
        return beta.iloc[0] / y.mean()

    def cal_gr_avg(self, series, period=21):
        growth_rate = [self.cal_growth_rate(series, remain=x) for x in range(period)]
        return np.mean(growth_rate)


# 复杂数据计算
class DataProcess(DataSelect):
    def __init__(self):
        super().__init__()
        self.basic_process = BasicDataProcess()

    def __getattr__(self, item):
        try:
            return getattr(self.basic_process, item)
        except AttributeError:
            return super().__getattr__(item)

    @staticmethod
    @time_decorator
    def pandarallel_cal(df, func, args=None, window=None):
        if window:
            df = df.rolling(window=window)

        if args is None:
            res = df.parallel_apply(func)
        else:
            res = df.parallel_apply(func, args=args)
        return res

    @staticmethod
    @time_decorator
    def rolling_apply(df, func, args=None, window=None):
        if window:
            res = df.rolling(window=window).apply(func, args=args)
        else:
            res = df.apply(func, args=args)
        return res

    @staticmethod
    @time_decorator
    def agg_cal(df, func, args=None, window=None):
        df = df.groupby('code')
        if window:
            res = df.rolling(window).agg(lambda x: func(x, args))
        else:
            res = df.agg(lambda x: func(x, args))
        return res

    # def cal_stock_return(self, stock_price, *names):
    #     """复牌第一天的收益率不纳入回归计算"""
    #     price_name, return_name = names
    #     return stock_price[['code', return_name]]
    #
    # @property
    # def stock_ret(self):
    #     if 'stock_ret' not in self.__dict__.keys():
    #         self.stock_return = self.cal_stock_return(self.stock_price, 'close', 'log_ret')
    #     return self.stock_return

    def cal_month_return(self, series, if_next=False):
        date, stocks = series.name, series.index
        if not if_next:
            lstdate = max(self.tradingdays[self.tradingdays <= date - pd.DateOffset(months=1)])
        else:
            lstdate = date
            date = max(self.tradingdays[self.tradingdays <= date + pd.DateOffset(months=1)])
        try:
            res = self.stock_price.loc[stocks, date] / self.stock_price.loc[stocks, lstdate] - 1
        except KeyError:
            res = series.where(pd.isnull(series), np.nan)
        return res

    def cal_month_returns(self, if_next=False):
        days = self.tradingdays[self.tradingdays >= pd.to_datetime('2005-01-01')]
        res = self.stock_price[list(days)]
        res = res.apply(self.cal_month_return).T
        if if_next:
            res = res.shift(21)
        """
        res = res.apply(self.cal_month_return, if_next=if_next).T
        """
        return res

    def get_growth_rate(self, df, freq='y'):
        rptdates = self.rptdates
        df = df.groupby(pd.Grouper(freq=freq)).apply(lambda x: x.iloc[-1]).reset_index()
        df = self.pandarallel_cal(df, self.cal_growth_rate, window=5).reset_index()
        df = pd.melt(df, id_vars='code', value_vars='rptdate', value_name='value')
        rptdates = pd.melt(rptdates, id_vars='code', value_vars='date', value_name='rptdate')
        res = df.merge(rptdates, on=['code', 'rptdate'], how='inner')
        res = res.pivot(values='value', index='code', columns='date')
        return res

    def rolling(self, df, window, func_name='sum', weights=None):
        df = df.where(pd.notnull(df), SENTINEL)
        func = getattr(np, func_name, )
        if func is None:
            msg = f"""Search func:{func_name} from numpy failed, 
                   only numpy ufunc is supported currently, please retry."""
            raise AttributeError(msg)
        if weights:
            args = func, weights
        else:
            args = func
        try:
            res = self.pandarallel_cal(df, self.nanfunc, args=args, window=window)
        except Exception:
            logger.info('Calculating under single core mode...')
            res = self.rolling_apply(df, self.nanfunc, args=args, window=window)
        return res

    def rolling_regress(self, y, x, window=5, intercept=True):
        x, y = self.align(x, y, False)
        start_idx = x.loc[pd.notnull(x).values.flatten()].index[0]
        x, y = x.loc[start_idx:], y.loc[start_idx:, :]

        beta, alpha, sigma = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in range(len(y) - window + 1):
            rolling_y = pd.DataFrame(y.iloc[i:i + window])
            rolling_x = pd.DataFrame(x.iloc[i:i + window])
            params, resid, r2 = self.regress(rolling_y, rolling_x, intercept=intercept, weight=1)
            vol = pd.DataFrame(resid.std()).T
            beta = pd.concat([beta, params.iloc[-1, :]])
            alpha = pd.concat([alpha, params.iloc[0, :]])
            sigma = pd.concat([sigma, vol])
        beta = beta.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        alpha = alpha.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        sigma = sigma.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        return beta, alpha, sigma

    @lru_cache(maxsize=999)
    def capm_regress(self, index_code, window=504):
        y = self.close
        x = self.index_price[index_code]
        beta, alpha, sigma = self.rolling_regress(y, x, window=window)
        return beta, alpha, sigma


# %%
# 成品
# 1******Size
class Size(DataProcess):
    @property
    def LNCAP(self):
        lncap = np.log(self.negotiable_mv)
        return lncap

    @property
    def MIDCAP(self):
        lncap = self.LNCAP.copy()
        midcap = self.pandarallel_cal(lncap, self.cal_midcap)
        return midcap.T

    def get_value(self):
        pass


# 2******Volatility
class Volatility(DataProcess):
    @property
    def BETA(self):
        if 'BETA' in self.__dict__:
            return self.__dict__['BETA']
        beta, alpha, hsigma = self.capm_regress(window=504, index_code='000300')
        self.__dict__['HSIGMA'] = hsigma
        self.__dict__['HALPHA'] = alpha
        return beta

    @property
    def HSIGMA(self):
        if 'HSIGMA' in self.__dict__:
            return self.__dict__['HSIGMA']
        beta, alpha, hsigma = self.capm_regress(window=504, index_code='000300')
        self.__dict__['BETA'] = beta
        self.__dict__['HALPHA'] = alpha
        return hsigma

    @property
    def HALPHA(self):
        if 'HALPHA' in self.__dict__:
            return self.__dict__['HALPHA']
        beta, alpha, hsigma = self.capm_regress(window=504, index_code='000300')
        self.__dict__['BETA'] = beta
        self.__dict__['HSIGMA'] = hsigma
        return alpha

    @property
    def DASTD(self):
        dastd = self.rolling(self.log_ret, window=252, func_name='std')
        return dastd

    @property
    def CMRA(self):
        stock_ret = self.log_ret.apply(lambda x: self.fill_vals(x, 0))
        cmra = self.pandarallel_cal(stock_ret, self.cal_cmra, args=(12, 21), window=252, axis=0).T
        return cmra

    def get_value(self):
        pass


# 3******Liquidity
class Liquidity(DataProcess):
    @property
    def STOM(self):
        amt, mkt_cap_float = self.align(self.turnover_rate, self.negotiable_mv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stom = self.pandarallel_cal(share_turnover, self.cal_liquidity, axis=0, window=21)
        return stom.T

    @property
    def STOQ(self):
        amt, mkt_cap_float = self.align(self.turnover_rate, self.negotiable_mv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoq = self.pandarallel_cal(share_turnover, self.cal_liquidity, axis=0, window=63)
        return stoq.T

    @property
    def STOA(self):
        amt, mkt_cap_float = self.align(self.turnover_rate, self.negotiable_mv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoa = self.pandarallel_cal(share_turnover, self.cal_liquidity, axis=0, window=252)
        return stoa.T

    @property
    def ATVR(self):
        atvr = self.rolling(self.turnover_rate, window=252, func_name='sum')
        return atvr

    def get_value(self):
        pass


# 4******Momentum
class Momentum(DataProcess):
    @property
    def STREV(self):
        stock_ret = self.log_ret.apply(lambda x: self.fill_vals(x, 0))
        strev = self.rolling(stock_ret, window=21, func_name='sum')
        return strev

    @property
    def SEASON(self):
        nyears = 5
        month_returns = self.cal_month_returns()
        month_returns_shift = [month_returns.shift(i * 21 * 12 - 21) for i in range(1, nyears + 1)]
        seasonality = sum(month_returns_shift) / nyears
        seasonality = seasonality.loc[f'{START_YEAR}':f'{END_YEAR}']
        return seasonality.T

    @property
    def INDMOM(self):
        window = 6 * 21
        stock_ret = self.log_ret.apply(lambda x: self.fill_vals(x, 0))

        rs = self.rolling(stock_ret, window, 'sum')
        cap_sqrt = np.sqrt(self.negotiable_mv)
        ind_citic_lv1 = self.industry
        rs, cap_sqrt, ind_citic_lv1 = self.align(rs, cap_sqrt, ind_citic_lv1)

        dat = pd.DataFrame()
        for df in [rs, cap_sqrt, ind_citic_lv1]:
            df.index.name = 'time'
            df.columns.name = 'code'
            dat = pd.concat([dat, df.unstack()], axis=1)
        dat.columns = ['rs', 'weight', 'ind']
        dat = dat.reset_index()

        rs_ind = {(time, ind): (df['weight'] * df['rs']).sum() / df['weight'].sum()
                  for time, df_gp in dat.groupby(['time'])
                  for ind, df in df_gp.groupby(['ind'])}

        def _get(key):
            nonlocal rs_ind
            try:
                return rs_ind[key]
            except:
                return np.nan

        dat['rs_ind'] = [_get((date, ind)) for date, ind in zip(dat['time'], dat['ind'])]
        dat['indmom'] = dat['rs_ind'] - dat['rs'] * dat['weight'] / dat['weight'].sum()
        indmom = pd.pivot_table(dat, values='indmom', index=['code'], columns=['time'])
        return indmom

    @property
    def RSTR(self, index_code='000300'):
        benchmark_ret, stock_ret = self.align(self.index_price[index_code], self.log_ret, False)
        excess_ret = stock_ret - benchmark_ret
        rstr = self.rolling(excess_ret, window=252, func_name='sum')
        rstr = rstr.rolling(window=11, min_periods=1).mean()
        return rstr

    def get_value(self):
        pass


# 5******Quality
class Quality(DataProcess):
    pass


class Leverage(Quality):
    @property
    def MLEV(self):
        ld = self.transfer_freq(self.total_noncurrent_liability, freq_before='q', freq_after='d')
        pe = self.transfer_freq(self.prefered_equity, freq_before='q', freq_after='d')
        me = self.totalmv.shift(1)
        me, pe, ld = self.align(me, pe, ld)
        mlev = (me + pe + ld) / me
        return mlev.T

    @property
    def BLEV(self):
        ld = self.transfer_freq(self.total_noncurrent_liability, freq_before='q', freq_after='d')
        pe = self.transfer_freq(self.prefered_equity, freq_before='q', freq_after='d')
        be = self.transfer_freq(self.total_shareholder_equity, freq_before='q', freq_after='d')
        be, pe, ld = self.align(be, pe, ld)
        blev = (be + pe + ld) / be
        return blev.T

    @property
    def DTOA(self):
        tl = self.transfer_freq(self.total_liability, freq_before='q', freq_after='d')
        ta = self.transfer_freq(self.total_assets, freq_before='q', freq_after='d')
        tl, ta = self.align(tl, ta)
        dtoa = tl / ta
        return dtoa.T

    def get_value(self):
        pass


class EarningsVariablity(Quality):
    window = 5

    @property
    def VSAL(self):
        sales_y = self.transfer_freq(self.operating_reenue, freq_before='q', freq_after='d')
        std = sales_y.rolling(window=self.window).std()
        avg = sales_y.rolling(window=self.window).mean()
        vsal = std / avg
        vsal = self.transfer_freq(vsal, freq_before='q', freq_after='d')
        return vsal.T

    @property
    def VERN(self):
        earnings_y = self.transfer_freq(self.netprofit, freq_before='q', freq_after='d')
        std = earnings_y.rolling(window=self.window).std()
        avg = earnings_y.rolling(window=self.window).mean()
        vern = std / avg

        vern = self.transfer_freq(vern, freq_before='q', freq_after='d')
        return vern.T

    @property
    def VFLO(self):
        cashflows_y = self.transfer_freq(self.cashequialentincrease, freq_before='q', freq_after='d')
        std = cashflows_y.rolling(window=self.window).std()
        avg = cashflows_y.rolling(window=self.window).mean()
        vflo = std / avg
        vflo = self.transfer_freq(vflo, freq_before='q', freq_after='d')
        return vflo.T

    @property
    def ETOPF_STD(self):
        etopf = self.west_eps_ftm.T
        etopf_std = etopf.rolling(window=240).std()
        etopf_std, close = self.align(etopf_std, self.close)
        etopf_std /= close
        return etopf_std.T

    def get_value(self):
        pass


class EarningsQuality(Quality):
    @property
    def ABS(self):
        cetoda, ce = self.align(self.capital_expenditure_todm, self.capital_expenditure)
        cetoda = cetoda.interpolate()
        da = ce / cetoda
        # sewmi_to_interestbeardebt, sewithoutmitotl, total_liability
        sewmi_to_ibd, sewmi_to_tl, tl = self.align(self.sewmi_to_ibd, self.sewmi_to_tl, self.tl)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)

        ta, cash, tl, td = self.align(self.totalassets, self.cashequialents, self.totalliability, ibd)
        noa = (ta - cash) - (tl - td)

        noa, da = self.align(noa, da)
        accr_bs = noa - noa.shift(1) - da

        accr_bs, ta = self.align(accr_bs, ta)
        abs_ = - accr_bs / ta
        abs_ = self.transfer_freq(abs_, freq_before='q', freq_after='d')
        return abs_.T

    @property
    def ACF(self):
        cetoda, ce = self.align(self.capital_expenditure_todm, self.capital_expenditure)
        cetoda = cetoda.interpolate()
        da = ce / cetoda  # 此处需对cetoda插值填充处理
        ni, cfo, cfi, da = self.align(self.net_profit, self.net_operate_cashflow, self.net_investcashflow, da)
        accr_cf = ni - (cfo + cfi) + da

        accr_cf, ta = self.align(accr_cf, self.totalassets)
        acf = - accr_cf / ta
        acf = self.transfer_freq(acf, freq_before='q', freq_after='d')
        return acf.T

    def get_value(self):
        pass


class Profitability(Quality):
    @property
    def ATO(self):
        sales = self.transfer_freq(
            self.transfer_data(self.operating_reenue, 'q', 'single', 'ttm', 'calendar', 'value', 'value_new'),
            freq_before='q', freq_after='d')
        ta = self.transfer_freq(self.total_assets, freq_before='q', freq_after='d')
        sales, ta = self.align(sales, ta)
        ato = sales / ta
        return ato.T

    @property
    def GP(self):
        sales = self.transfer_freq(self.operating_reenue, freq_before='q', freq_after='d')
        cogs = self.transfer_freq(self.cogs_q, freq_before='q', freq_after='d')
        ta = self.transfer_freq(self.total_assets, freq_before='q', freq_after='d')
        sales, cogs, ta = self.align(sales, cogs, ta)
        gp = (sales - cogs) / ta
        return gp.T

    @property
    def GPM(self):
        sales = self.transfer_freq(self.operating_reenue, freq_before='q', freq_after='d')
        cogs = self.transfer_freq(self.cogs_q, freq_before='q', freq_after='d')
        sales, cogs = self.align(sales, cogs)
        gpm = (sales - cogs) / sales
        return gpm.T

    @property
    def ROA(self):
        earnings = self._transfer_freq(
            self.transfer_data(self.net_profit, 'q', 'single', 'ttm', 'calendar', 'value', 'value_new'),
            freq_before='q', freq_after='d')
        ta = self.transfer_freq(self.total_assets, freq_before='q', freq_after='d')
        earnings, ta = self.align(earnings, ta)
        roa = earnings / ta
        return roa.T

    def get_value(self):
        pass


class InvestmentQuality(Quality):
    window = 5

    @property
    def AGRO(self):
        agro = self.get_growth_rate(self.totalassets)
        return agro

    @property
    def IGRO(self):
        igro = self.get_growth_rate(self.totalshares)
        return igro

    @property
    def CXGRO(self):
        cxgro = self.get_growth_rate(self.capital_expenditure)
        return cxgro

    def get_value(self):
        pass


# 6*******Value
class Value(DataProcess):
    @property
    def BTOP(self):
        bv = self.transfer_freq(self.sewithoutmi, freq_before='q', freq_after='d')
        bv, mkv = self.align(bv, self.totalmv)
        btop = bv / (mkv * 10000)
        return btop.T


class EarningsYield(Value):
    @property
    def ETOP(self):
        earings_ttm = self.transfer_freq(
            self.transfer_data(self.netprofit, 'q', 'single', 'ttm', 'calendar', 'value', 'value_new'),
            freq_before='q', freq_after='d')
        e_ttm, mkv = self.align(earings_ttm, self.totalmv)
        etop = e_ttm / (mkv * 10000)
        return etop.T

    @property
    def ETOPF(self):
        return pd.DataFrame()

    @property
    def CETOP(self):
        cash_earnings = self.transfer_freq(
            self.transfer_data(self.netoperatecashflow, 'q', 'single', 'ttm', 'calendar', 'value', 'value_new'),
            freq_before='q', freq_after='d')
        ce, mkv = self.align(cash_earnings, self.totalmv)
        cetop = ce / (mkv * 10000)
        return cetop.T

    @property
    def EM(self):
        ebit = self.transfer_freq(self.ebit, freq_before='q', freq_after='d')
        sewmi_to_ibd, sewmit_to_tl, tl = self.align(
            self.sewmitointerestbeardebt, self.sewithoutmitotl, self.totalliability)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
        ibd = self.transfer_freq(ibd, freq_before='q', freq_after='d')

        cash = self.transfer_freq(self.cashequialents, freq_before='q', freq_after='d')
        ebit, mkv, ibd, cash = self.align(ebit, self.totalmv, ibd, cash)
        ev = mkv * 10000 + ibd - cash
        em = ebit / ev
        return em.T

    def get_value(self):
        pass


class LongTermReversal(Value):
    @property
    def LTRSTR(self, index_code='000300'):
        benchmark_ret = self.index_price[index_code]
        stock_ret = self.log_ret
        benchmark_ret, stock_ret = self.align(benchmark_ret, stock_ret)
        excess_ret = stock_ret - benchmark_ret
        ltrstr = self.rolling(excess_ret, window=1040, func_name='sum').T
        ltrstr = (-1) * ltrstr.shift(273).rolling(window=11).mean()
        return ltrstr.T

    @property
    def LTHALPHA(self, index_code='000300'):
        _, alpha, _ = self.capm_regress(window=1040, index_code=index_code)
        lthalpha = (-1) * alpha.T.shift(273).rolling(window=11).mean()
        return lthalpha.T

    def get_value(self):
        pass


# 7*******Growth
class Growth(DataProcess):
    window = 5

    @property
    def EGRO(self):
        egro = self.get_growth_rate(self.eps)
        return egro

    @property
    def SGRO(self):
        total_shares, operatingrevenue = self.align(self.totalshares, self.operatingreenue)
        ops = operatingrevenue / total_shares
        sgro = self.get_growth_rate(ops)
        return sgro

    def get_value(self):
        pass


# 8*******Sentiment
class Sentiment(DataProcess):
    @property
    def RRIBS(self):
        return pd.DataFrame()

    @property
    def EPIBSC(self):
        return pd.DataFrame()

    @property
    def EARNC(self):
        return pd.DataFrame()

    def get_value(self):
        pass


# 9*******DividendYield
class DividendYield(DataProcess):
    @property
    def DTOP(self):
        dps = self.transfer_freq(
            self.transfer_data(self.fill_vals(self.dividendps, val=0),
                               'q', 'single', 'ttm', 'calendar', 'value', 'value_new'),
            freq_before='q', freq_after='d')
        price_lme = self._get_price_last_month_end('close')
        dps, price_lme = self.align(dps, price_lme)
        dtop = dps / price_lme
        return dtop.T

    def get_value(self):
        pass


# %%
# 数据再处理:去极值、标准化、中性化、正交化
class DataProcessNew(DataProcess):
    def __init__(self):
        super().__init__()
        self.size = Size()
        self.vol = Volatility()
        self.liquidity = Liquidity()
        self.momentum = Momentum()
        self.leverage = Leverage()
        self.earningsvariablity = EarningsVariablity()
        self.earningsquality = EarningsQuality()
        self.profitability = Profitability()
        self.investmentquality = InvestmentQuality()
        self.earningsyield = EarningsYield()
        self.longtermreversal = LongTermReversal()
        self.growth = Growth()
        self.sentiment = Sentiment()
        self.dividendyield = DividendYield()
        self.style_factor = None

    def __getattr__(self, item):
        try:
            return getattr(self, name).get_value()
        except Exception:
            raise ValueError(f'no value in {item}')

    def get_vif(self):
        data = {}
        for day in self.close.index:
            sub_data = {}
            for name in self.__dict__.keys():
                other = set(self.__dict__.keys()) - {name}
                y = self.__dict__[name].loc[day]
                x = pd.concat([self.__dict__[na].loc[day].to_frame(na) for na in other], axis=1)
                x = sm.add_constant(x)
                res = sm.WLS(y, x, weights=1).fit()
                sub_data.update({name: 1 - res.rsquared_adj})
            data.update({day: sub_data})
        return pd.DataFrame(data)

    def no_multi_linear(self, method='barra'):
        """method = 'barra' or 'vif'"""
        if method == 'barra':
            self.style_factor = self.barra_orthogonalize()
        else:          # method == 'vif':
            self.style_factor = self.vif_orthogonalize()

    def barra_orthogonalize(self):
        return pd.DataFrame()

    def vif_orthogonalize(self):
        vif = self.get_vif()
        return vif

    def remove_extreme(self):
        for name in self.__dict__.keys():
            upper_name = name.upper()
            self.__dict__[upper_name] = self.__dict__[name].parallel_apply(self.three_sigma, axis=1)

    def new_standardize(self):
        for name in self.__dict__.keys():
            upper_name = name.upper()
            self.__dict__[upper_name] = self.__dict__[name].parallel_apply(self.standardize, axis=1)

    def new_neutralize(self):
        industry = self.industry.set_index('code')['industry']
        for name in self.__dict__.keys():
            upper_name = name.upper()
            name_df = self.__dict__[name].parallel_apply(self.neutralize, args=('ind', industry), axis=1)
            self.__dict__[upper_name] = name_df.parallel_apply(self.neutralize, args=('scale', industry), axis=1)


# %%
# 回归
class RegressionModel(DataProcessNew):
    def __init__(self):
        super().__init__()

    def __getattr__(self, item):
        try:
            return getattr(self, name).get_value()
        except Exception:
            raise ValueError(f'no value in {item}')

    def regression(self):
        pass


# %%
if __name__ == '__main__':
    a1 = pd.DataFrame([1, 4, 2, 3, 4])
    a2 = pd.DataFrame([np.nan] * 3)
    a3 = pd.DataFrame([4, 7, 9, 6])
    a4 = pd.DataFrame([np.nan] * 2)
    a5 = pd.DataFrame([8, 8, 6])
    a = pd.concat([a1, a2, a3, a4, a5])
    a['b'] = np.random.randn(17)
    a.columns = ['a', 'b']
    b = pd.DataFrame(range(17), columns=['c'])
    a = a.reset_index()


    def align(df1, df2, if_column=True, *dfs):
        dfs_all = [df1, df2] + list(dfs)
        mut_date_range = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
        dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        if if_column:
            mut_codes = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
            dfs_all = [df.loc[:, mut_codes] for df in dfs_all]
        return dfs_all


    a, b = align(a.dropna(), b.dropna(), False)
    y, x = a.copy(), b.copy()
    x = sm.add_constant(x)
    result1 = sm.OLS(y['b'], x).fit()
    print(result1.params)
    result1 = sm.OLS(y['a'], x).fit()
    print(result1.params)
    result1 = sm.OLS(y['index'], x).fit()
    print(result1.params)
    result = sm.OLS(y, x).fit()
    print(result.params)
