import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pymysql
from functools import lru_cache
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from app.utils import *
import feather
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, nan_euclidean_distances
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import os
history_data = '/Users/kai/Desktop/基金评级数据/基金评级/统计/'
CACHE_PATH = '/Users/kai/Desktop/基金评级数据/基金评级/cache_rating/'
CACHE_PATH1 = '/Users/kai/Desktop/基金评级数据/基金评级/分类统计/'


def get_ret_new(series):
    valid = series[~series.isna()].sort_index().index[:150]
    series_new = series.copy()
    series_new.loc[series_new.index.isin(valid)] = np.nan
    return series_new


def get_new_cumprod(series):
    try:
        start_idx = series[series.notna()].index[0]
        return (1 + series[series.index >= start_idx]).cumprod()
    except:
        return pd.Series()


# 该函数检验get_new_cumprod
def get_nannum(series):
    start_idx = series[series.notna()].index[0]
    series_new = series[series.index >= start_idx]
    nannum = 1 - len(series_new.dropna()) / len(series_new)
    return nannum


def get_sml_bool(series, thrshld):
    if len(series[~series.isna()]) >= thrshld * len(series):
        return True
    return False


def get_sub_ret_cos(mf_return, thrshld, func, if_cumprod=False):
    ret_new = mf_return[mf_return.columns[mf_return.apply(get_sml_bool, args=(thrshld,))]]
    if len(ret_new.columns) <= 1:
        return pd.DataFrame()
    if if_cumprod:
        # 用get_nannum检验，但是目前代码无需用get_nannum检验了
        ret_new_notna = ret_new[ret_new.notna().all(axis=1)]
        na_dates = ret_new.index[~ret_new.index.isin(ret_new_notna.index)]
        df_na = pd.DataFrame(index=na_dates, columns=ret_new.columns)
        df_new = pd.concat([(1 + ret_new_notna).cumprod(), df_na])
        # df_new = ret_new.apply(get_new_cumprod)
    else:
        df_new = ret_new.dropna()
    if len(df_new.dropna()) <= 20:
        return pd.DataFrame()
    ret_cos = pd.DataFrame(func(df_new.T), columns=ret_new.columns, index=ret_new.columns)
    return ret_cos


def get_ret_cos(ret, func, if_cumprod=False):
    ret_cos = pd.DataFrame()
    for thrshld in np.arange(0.2, 1.01, 0.1):
        ret_cos_new = get_sub_ret_cos(ret, thrshld, func, if_cumprod)
        if len(ret_cos_new) <= 1:
            continue
        if ret_cos.empty:
            ret_cos = ret_cos_new.copy()
        else:
            ret_cos.loc[ret_cos_new.index, ret_cos_new.columns] = ret_cos_new
    return ret_cos


def FundHierarchicalClustering(mf_ret, similar, threshold, linkage, if_distance):
    acl = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree=True,
                                  distance_threshold=threshold, linkage=linkage)
    if not if_distance:
        similar = 1 - similar
    lab = acl.fit(similar).labels_
    labels = pd.DataFrame(lab, columns=[str(threshold)], index=mf_ret.columns).sort_values(str(threshold))
    return labels


def get_basic_info():
    # 评级与基本信息
    query_r1 = """select secucode code, end_date date, stars, period, first_class_code, second_class_code, if_quarter_end
    from fund_rating where stars in (3,4,5) order by end_date asc"""
    rating = pd.read_sql(query_r1, DbUtil.get_conn('funddata'))
    codes = rating.code.tolist()
    q_info = f"""select fi.maincode code, fi.name_abbr name, fi.company, fi.individual_ratio ratio, fi.bond_ratio bond,
    fi.stock_ratio stock from funddata.fund_information fi inner join funddata.fund_type ft on fi.secucode = ft.secucode
    where fi.secucode in ({str(codes)[1:-1]}) and fi.listed_state = 1
    and (ft.first_class_code, ft.second_class_code) in ((1, null), (2, 1), (2, 2), (2, 3), (3, 6))"""
    info = pd.read_sql(q_info, DbUtil.get_conn('funddata'))
    query_r2 = f"""select secucode code, end_date date, stars, period, first_class_code, second_class_code, if_quarter_end
    from fund_rating where secucode in ({str(codes)[1:-1]}) order by end_date asc"""
    rating = pd.read_sql(query_r2, DbUtil.get_conn('funddata'))
    return info, rating


def get_ret_first(company_all):
    for company in company_all:
        print(company)
        codes = company_code[company_code.values == company]
        k, v = len(codes) // 20, len(codes) % 20
        m = k if v == 0 else k + 1
        mf_ret = []
        for i in range(m):
            sub_codes = codes.iloc[i * 20: (i + 1) * 20].index.tolist()
            query_ret = f"""select secucode code, end_date date, complex_log_return ret from mf_return
            where secucode in ({str(sub_codes)[1:-1]})"""
            sub_mf_ret = pd.read_sql(query_ret, DbUtil.get_conn('funddata'))
            mf_ret.append(sub_mf_ret)
        mf_ret = pd.concat(mf_ret)
        feather.write_dataframe(mf_ret, CACHE_PATH + company + '.feather')


def get_ret():
    mf_ret = []
    company_all = [x[:-8] for x in os.listdir(CACHE_PATH)]
    for company in company_all:
        try:
            sub_mf_ret = feather.read_dataframe(CACHE_PATH + company + '.feather')
            mf_ret.append(sub_mf_ret)
        except:
            pass
    return pd.concat(mf_ret).pivot(columns='code', index='date', values='ret')


def get_cal_bool(series, year):
    start_idx = series[series.notna()].index[0]
    if start_idx.year > year:
        return False
    start_y1, start_y2 = f'{year}-01-01', f'{year + 1}-01-01'
    start_series = series[(series.index >= start_y1) & (series.index < start_y2)]
    nan1 = len(start_series[start_series.index < start_idx]) / len(start_series)
    if nan1 >= 0.5:
        return False
    series_new = series[series.index >= start_y1]
    nan2 = 1 - len(series_new.dropna()) / len(series_new)
    if nan2 >= 0.8:
        return False
    return True


def get_raw_similar(sub_ret):
    # 除去公布净值的交易日少于全部交易日的20%
    ret = sub_ret.T[sub_ret.apply(get_sml_bool, args=(0.2,))].T
    similar1 = get_ret_cos(ret, cosine_similarity)
    similar2 = ret.corr()
    similar3 = get_ret_cos(ret, nan_euclidean_distances, True)
    return similar1, similar2, similar3


# 检验距离的分布模型
def similar_distribution_test(similar1, similar3):
    a1 = 1 - np.sort(similar1.values.flatten())[:-len(similar1)]
    a3 = np.sort(similar3.values.flatten())[::-1][:-len(similar3)]
    # 两个分布基本相似
    plt.figure(figsize=(15, 15))
    plt.scatter(a1, a3)
    plt.plot(a1, a1 * np.mean(a3) / np.mean(a1))
    plt.show()
    # 分布QQ图，从逻辑上也能解释，similar本质上是平方性质的
    x = np.array(range(len(a1))) ** 2
    plt.figure(figsize=(15, 5))
    plt.plot(x, a1[::-1])
    plt.show()


# 至此，我们确定标准化的方法：即similar_new = np.sqrt(similar)，然后再标准化，这样的分布更接近均匀分布
def get_new_feature(*args):
    similar1, similar2, similar3 = args[:3]
    similar1_new = np.sqrt(1 - similar1)
    similar3_new = np.sqrt(similar3)
    # similar1_new = (similar1_new - np.mean(similar1_new)) / np.std(similar1_new)
    # similar3_new = (similar3_new - np.mean(similar3_new)) / np.std(similar3_new)
    new_feature = pd.concat([similar1_new, similar3_new]).fillna(0)
    return new_feature


def get_similar_new(*args):
    new_feature = get_new_feature(*args)
    codes = new_feature.columns
    similar_new = pd.DataFrame(euclidean_distances(new_feature.T), columns=codes, index=codes)
    return similar_new


def get_cat_fund(ret, similar_new):
    cat_fund = []
    for distance in np.arange(40, 0, -1):
        labels_new = FundHierarchicalClustering(ret, similar_new, distance, 'average', True)
        cat_fund.append(labels_new)
    cat_fund = pd.concat(cat_fund, axis=1)
    cat_fund = cat_fund.sort_values(list(cat_fund.columns))
    for i in cat_fund.columns:
        labs = cat_fund[i].drop_duplicates().to_list()
        dict_labs = dict(zip(labs, list(range(len(labs)))))
        cat_fund[i] = cat_fund[i].apply(lambda x: dict_labs[x])
    # similar_new[labels_new_all.index].loc[labels_new_all.index].to_csv(CACHE_PATH + 'similar_new.csv')
    return cat_fund


def get_labels(ret, similar_new):
    labels = pd.Series(name='cat')
    for distance in np.arange(40, 0, -1):
        labels = FundHierarchicalClustering(ret, similar_new, distance, 'average', True)[str(distance)]
        if labels.max() > 10:
            labels = labels.rename('cat')
            break
    return labels.reset_index()


def save_data(date, cat_fund):
    feather.write_dataframe(cat_fund, CACHE_PATH + date + '.csv')


def cal_dates_label(mf_ret):
    dates = pd.date_range('2005-12-31', '2020-03-18', freq='M')
    mf_ret = mf_ret.apply(get_ret_new)
    basic_info, _ = get_basic_info()
    labels_ = {}
    for date in dates:
        print(date)
        sub_ret = mf_ret.loc[date: date + pd.DateOffset(months=24)].dropna(how='all')
        similar = get_raw_similar(sub_ret)
        similar1, similar2, similar3 = similar[:3]
        similar_new = get_similar_new(similar1, similar2, similar3)
        cat_fund = get_cat_fund(similar1, similar_new)
        cat_fund['date'] = date + pd.DateOffset(months=24)
        info = basic_info.merge(labels, on='code', how='inner')


def cal_dates_cat(mf_ret):
    dates = pd.date_range('2009-12-31', '2022-03-18', freq='M')
    mf_ret_new = mf_ret.apply(get_ret_new)
    # basic_info, _ = get_basic_info()
    cat_funds = []
    for date in dates:
        print(date)
        sub_ret = mf_ret_new.loc[date: date + pd.DateOffset(months=24)].dropna(how='all')
        cat_fund = cal_date_cat(sub_ret)
        cat_fund['date'] = date + pd.DateOffset(months=24)
        label = cal_cat_label(cat_fund)
        benchmark = cal_benchmark(sub_ret, label)
        feather.write_dataframe(benchmark, history_data + 'benchmark/' + date.strftime('%Y-%m-%d') + '.feather')
    return cat_funds


def plt_strategy_funds(mf_ret):
    dates = pd.date_range('2009-12-31', '2022-03-18', freq='M')
    mf_ret_new = mf_ret.apply(get_ret_new)
    # basic_info, _ = get_basic_info()
    cat_funds = []
    for date in dates:
        print(date)
        sub_ret = mf_ret_new.loc[date: date + pd.DateOffset(months=24)].dropna(how='all')
        cat_fund = cal_date_cat(sub_ret)
        cat_fund['date'] = date + pd.DateOffset(months=24)
        label = cal_cat_label(cat_fund)
        for j in label.unique():
            codes = ii[ii == j].index.tolist()
            plt.figure(figsize=(15, 5))
            plt.plot(ret[codes].apply(get_new_cumprod))
            plt.title(i + '--' + str(j))
            plt.savefig(
                '/Users/kai/WeDrive/杭州智君信息科技有限公司/公司资料/Books & Papers/券商研报/多因子量化/未命名文件夹/2011-11-30/'
                + i + '--' + str(j) + '.png')
    return cat_funds


def cal_date_cat(sub_ret):
    similar = get_raw_similar(sub_ret)
    similar1, similar2, similar3 = similar[:3]
    similar_new = get_similar_new(similar1, similar2, similar3)
    cat_fund = get_cat_fund(similar1, similar_new)
    return cat_fund


def cal_cat_label(cat_fund):
    cat = cat_fund[cat_fund.columns[:-1]]
    indexes = [int(x) for x in cat.columns[cat.iloc[-1] > 10]]
    cat_index = str(max(indexes))
    label = cat_fund[['date', cat_index]].rename(columns={cat_index: 'cat'}).reset_index()
    return label


def cal_benchmark(sub_ret, label):
    ret_now = sub_ret.loc[sub_ret.index.strftime('%y-%m') == max(sub_ret.index).strftime('%y-%m')]
    ret_now = ret_now.unstack().rename('ret').dropna().reset_index()
    codes_ret = label[['code', 'cat']].merge(ret_now, on='code', how='left')
    labels_ret = codes_ret.groupby(['cat', 'date'])['ret'].mean().rename('cat_ret').reset_index()
    benchmark = codes_ret.merge(labels_ret, on=['cat', 'date'], how='outer')
    return benchmark


def cal_cat_labels(cat_funds):
    labels = []
    for cat_fund in cat_funds:
        cat = cat_fund[cat_fund.columns[:-1]]
        indexes = [int(x) for x in cat.columns[cat.iloc[-1] > 10]]
        cat_index = str(max(indexes))
        label = cat_fund[['date', cat_index]].rename(columns={cat_index: 'cat'}).reset_index()
        labels.append(label)
    labels = pd.concat(labels)
    return labels


def get_cal_fund():
    mf_ret = get_ret()
    cal_dates_cat(mf_ret)


# %%
def get_benchmark():
    benchmark = []
    dates = [x[:-8] for x in os.listdir(history_data + 'benchmark/') if x[:-8] != '.']
    for date in dates:
        if date >= '2020-03-31':
            continue
        sub_bench = feather.read_dataframe(history_data + 'benchmark/' + date + '.feather')
        benchmark.append(sub_bench)
    benchmark = pd.concat(benchmark).sort_values(['date', 'cat', 'code']).reset_index(drop=True)
    return benchmark


def tm_model(benchmark):
    benchmark['exc_ret'] = benchmark['ret'] - benchmark['cat_ret']














