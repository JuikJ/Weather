# !/usr/bin/env python
# @author : hyegyu
# @Date : 2024.05.29
# @coding: utf-8


import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.float_format', '{:.3f}'.format)


train = pd.read_csv('./electric_train.csv', encoding='cp949', index_col=0)


train.columns = [col.split('electric_train.', 1)[-1] for col in train.columns]


# 아예 NA인 값은 없음
train.isnull().sum().to_frame('na_count')


train.describe()


# Q&A 에서 -99인 값은 결측치라고 되어있기때문에 삭제
# 7593355 => 7593181, 174개 제거
train = train[(train['nph_ws_10m'] != -99) & (train['elec'] != -99)]
train.reset_index(inplace=True, drop=True)


# ## AWS 데이터 전처리
# meta 출처 : 기상청 기상자료개방포털 - 데이터 - 메타데이터 - 관측지점정보 - 방재기상관측
aws = pd.read_csv('./방재기상관측_AWS_지점.csv', encoding='cp949', index_col=0)

aws_meta = pd.read_csv('./META_관측지점정보_20240529164351.csv', encoding='cp949')
aws_meta.drop_duplicates(subset='지점', keep='first', inplace=True, ignore_index=True)


aws = aws[['지점번호', 	'지점명(한글)', '경도(degree)', '위도(degree)']]
aws_meta = aws_meta[['지점', '지점명', '위도', '경도']]


# 결측치 없음
aws.isnull().sum().to_frame('na_count')

# 결측치 없음
aws_meta.isnull().sum().to_frame('na_count')


# 컬럼명 영문 변환
aws = aws.rename(columns={'지점번호': 'stn','지점명(한글)': 'stn_nm','위도(degree)': 'lon','경도(degree)': 'lat'})
aws_meta = aws_meta.rename(columns={'지점': 'stn','지점명': 'stn_nm','위도': 'lon','경도': 'lat'})


# aws 데이터는 692개
# aws_meta 데이터는 549개
aws_df = pd.concat([aws, aws_meta], axis=0).drop_duplicates(subset='stn', keep='first', ignore_index=True)


# GeoDataFrame 변환
aws_df = gpd.GeoDataFrame( \
            aws_df, \
            geometry=gpd.points_from_xy( \
                                        aws_df["lat"], \
                                        aws_df["lon"]\
                                       ),\
            crs="EPSG:4326"\
        ).drop(columns=["lat", 'lon'])



# ## 데이터 결합
df = pd.merge(train, aws_df, how='left', on='stn')
del df["stn_nm"]

# 결측치 없음
df.isnull().sum().to_frame('na_count')


# 연도, 월, 일, 시간으로 나누기
df['tm'] = pd.to_datetime(df['tm'])

df['year'] = df['tm'].dt.year
df['month'] = df['tm'].dt.month
df['day'] = df['tm'].dt.day
df['hour'] = df['tm'].dt.hour

del df['tm']


df.groupby('stn')['elec'].agg(['sum', 'count']).sort_values('sum')
'''
sum	count
stn		
283	875739.120	8758
884	875999.740	8760
252	875999.880	8760
605	875999.970	8760
177	876000.200	8760
...	...	...
550	10521599.940	105216
572	10521600.960	105216
541	12276001.950	122760
133	13151519.440	131515
846	14904810.850	149052
'''


# ## EDA
def df_corr(df):
    '''
    상관계수 변환
    
    Parameters : 
    df : 상관계수를 보고하자는 데이터프레임
    
    Returns :
    result : 상관계수로 변환된 데이터프레임
    '''
    df_cor = df.corr()
    result = df_cor[(df_cor >= 0.5) | (df_cor <= -0.5)]
    result = result.dropna(how='all', axis=0).dropna(how='all', axis=1)
    return result



# nph_ta - nph_ta_chi : 0.94
# sum_qctr - sum_load : 0.91
# n - sum_qctr : 0.91

plt.figure(figsize=(15, 10))
sns.heatmap(df_corr(df), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


df.describe()
