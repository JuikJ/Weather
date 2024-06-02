#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from datetime import datetime, timedelta

train = pd.read_csv('electric_train.csv', index_col = 0)
test = pd.read_csv('electric_test.csv', index_col = 0)

############################################# 주제 : 공동주택 전력수요 예측 개선
# 기상변수 및 공공데이터 등 활용 공동주택 전력수요 증감 영향 요인 분석
# 계절, 지역에 따른 모델 세분화를 통한 공동주택 전력 수요 예측 (전력기상지수) 최적모델 개발

# 전력기상지수 : 기상변화에 따른 지역별 공동주택의 예상되는 전력부하 변화를 기상예보처럼 국민들이 쉽게 인지할 수 있도록 수치화하여 예측해주는 서비스

############################################# 컬럼 정보
# num : 격자 넘버 - 기상청 동네예보 격자넘버
# tm : 날짜 - 공동주택 전력부하 측정 날짜 시간포함, 단위(0 ~ 23시)
# hh24 : 시간 - 공동주택 전력부하 측정 시간(1 ~ 24), 5시는 4시 1분 ~ 5시 00분 까지 전력 부하를 의미
# n : 공동주택 수 - 해당격자의 전력통계 산출에 포함된 공동주택의 수, 단위(단지)
# stn : 지점 번호 - AWS 지점 번호, AWS(Automated Weather Station, 자동기상관측장비)는 과거에 사람이 직접 관측하던 것을 자동으로 관측할 수 있도록 설계한 기상관측장비
# sum_qctr : 계약전력합계 - 해당격자의 전력통계 산출에 포함된 공동주택의 계약전력 합계
# sum_load : 전력수요합계 - 해당격자/시각에 측정된 공동주택의 전력수요 합계
# n_mean_load : 전력부하량평균 - 격자내 총 전력부하량을 아파트 수로 나누어 격자의 평균 부하량을 산출
# nph_ta : 기온 - 단위(C)
# nph_hm : 상대습도 - 단위(%)
# nph_ws_10m : 풍속 - 객관분석 10분 평균 풍속, 단위(m/s)
# nph_rn_60m : 강수량 - 객관분석 1시간 누적 강수량, 단위(mm)
# nph_ta_chi : 체감온도 - 단위(C)
# weekday : 요일 - 요일을 숫자형식으로 표시 월요일(0) ~ 일요일(6)
# week_name : 주중 주말 - 주중 주말을 숫자형식으로 표시, 주중(0) ~ 주말(1)
# elec : 전력기상지수(TARGET) - 해당 격자의 공동주택의 연평균 부하량을 100으로 했을 때, 해당 시작에 예상되는 부하량을 상대적인 수치로 표현

# test 데이터는 전력기상지수를 산출할 수 있는 변수인
# sum_qctr, n, sum_load, n_mean_load를 제외하고 제공됨

# 컬럼명 변경
train.rename(columns = {'electric_train.num' : 'num', 'electric_train.tm' : 'tm',
                        'electric_train.hh24' : 'hh24', 'electric_train.n' : 'n',
                        'electric_train.stn' : 'stn', 'electric_train.sum_qctr' : 'sum_qctr',
                        'electric_train.sum_load' : 'sum_load', 'electric_train.n_mean_load' : 'n_mean_load',
                        'electric_train.nph_ta' : 'nph_ta', 'electric_train.nph_hm' : 'nph_hm',
                        'electric_train.nph_ws_10m' : 'nph_ws_10m', 'electric_train.nph_rn_60m' : 'nph_rn_60m',
                        'electric_train.nph_ta_chi' : 'nph_ta_chi', 'electric_train.weekday' : 'weekday',
                        'electric_train.week_name' : 'week_name', 'electric_train.elec' : 'elec'}, inplace = True)

test.rename(columns = {'electric_test.num' : 'num', 'electric_test.tm' : 'tm',
                        'electric_test.hh24' : 'hh24', 'electric_test.n' : 'n',
                        'electric_test.stn' : 'stn', 'electric_test.sum_qctr' : 'sum_qctr',
                        'electric_test.sum_load' : 'sum_load', 'electric_test.n_mean_load' : 'n_mean_load',
                        'electric_test.nph_ta' : 'nph_ta', 'electric_test.nph_hm' : 'nph_hm',
                        'electric_test.nph_ws_10m' : 'nph_ws_10m', 'electric_test.nph_rn_60m' : 'nph_rn_60m',
                        'electric_test.nph_ta_chi' : 'nph_ta_chi', 'electric_test.weekday' : 'weekday',
                        'electric_test.week_name' : 'week_name', 'electric_test.elec' : 'elec'}, inplace = True)
##### 데이터셋의 기본 정보
train.info()

train.describe().T

train.shape

############################################# 데이터 전처리

# 결측치 찾기

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count() * 100)
    tt = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    types = []

    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)

    tt['Types'] = types
    return(np.transpose(tt))

missing_data(train)

missing_data(test)

# -99, -99.9, -999가 결측치 -> -99 존재 -> 결측치 처리해야함
rows_with_missing_values = train[(train == -99) | (train == -99.9) | (train == -999.0)].any(axis = 1)
train[rows_with_missing_values]
# 174개의 missing_values

rows_with_missing_values = test[(test == -99) | (test == -99.9) | (test == -999.0)].any(axis = 1)
test[rows_with_missing_values]
# 96개의 missing_values


# 년/월/일 추출 후 tm 변수 Drop
train['tm'] = pd.to_datetime(train['tm'], format = '%Y-%m-%d %H:%M:%S')

train['Year'] = train['tm'].dt.year
train['Month'] = train['tm'].dt.month
train['day'] = train['tm'].dt.day

test['tm'] = pd.to_datetime(test['tm'], format = '%Y-%m-%d %H:%M:%S')

test['Year'] = test['tm'].dt.year
test['Month'] = test['tm'].dt.month
test['day'] = test['tm'].dt.day

train = train.drop(['tm'], axis = 1)
test = test.drop(['tm'], axis = 1)

# 기온 범주화

def categorize_temperature(temperature):
    # 기온을 범주화 하는 함수

    if temperature < -10:
        return 1 # 매우 추움
    elif -10 <= temperature < 0:
        return 2 # 추움
    elif 0 <= temperature < 25:
        return 3 # 적당함
    elif 25 <= temperature < 30:
        return 4 # 더움
    else:
        return 5 # 매우 더움

train['Temperature_Category'] = train['nph_ta'].apply(categorize_temperature)
test['Temperature_Category'] = test['nph_ta'].apply(categorize_temperature)

# 상대습도 범주화

def categorize_rh(rh): # Relative Humidity
    # 상대습도를 범주화하는 함수

    if rh < 10.0:
        return 1 # 매우 건조
    elif 10.0 <= rh < 30.0:
        return 2 # 건조
    elif 30 <= rh < 60.0:
        return 3 # 보통
    elif 60 <= rh < 80.0:
        return 4 # 다습
    else:
        return 5 # 매우 다습

train['Rh_Category'] = train['nph_hm'].apply(categorize_rh)
test['Rh_Category'] = test['nph_hm'].apply(categorize_rh)

# 풍속 범주화

def categorize_ws(ws): # Wind Speed
    # 상대습도를 범주화하는 함수

    if ws <= 0.0:
        return 1 # 매우 약함
    elif 0.0 < ws <= 3.0:
        return 2 # 약함
    elif 3.0 < ws <= 6.0:
        return 3 # 보통
    elif 6.0 < ws <= 10.0:
        return 4 # 강함
    else:
        return 5 # 매우 강함

train['Ws_Category'] = train['nph_ws_10m'].apply(categorize_ws)
test['Ws_Category'] = test['nph_ws_10m'].apply(categorize_ws)

# 강수량 범주화

def categorize_precipitation(precipitation):
    # 강수량을 범주화하는 함수

    if precipitation <= 0.0:
        return 1 # 매우 적음
    elif 0.0 < precipitation <= 3.0:
        return 2 # 적음
    elif 3.0 < precipitation <= 15.0: 
        return 3 # 보통 
    elif 15.0 < precipitation <= 30.0: 
        return 4 # 많음
    else:
        return 5 # 매우 많음

train['Precipitation_Category'] = train['nph_rn_60m'].apply(categorize_precipitation)
test['Precipitation_Category'] = test['nph_rn_60m'].apply(categorize_precipitation)


############################################# 샘플 추출 (데이터 셋이 너무 큼)
sample_train = train.sample(n = 10000)
col = train.columns.tolist()

num_cols = sample_train.select_dtypes(include = ['int64', 'float64'])


############################################# 데이터 시각화

colors = ['magenta', 'green', 'blue']

# 변수 간 상관관계 알아보기
plt.figure(figsize = (15, 10))
sns.heatmap(num_cols.corr(), 
            annot = True, 
            cmap = 'YlOrRd',
            linewidths = 0.4,
            linecolor = 'white')

plt.title('Correlation of numeric values')
plt.show()

# 월별 전력기상지수(sum_load)를 평균내어 그래프로 시각화

monthly_elec = train.groupby(['Year', 'Month'])['elec'].mean()[:-1].reset_index()  # 2023년 1월 제외
grouped_data_elec = monthly_elec.groupby('Year')

plt.figure(figsize = (12, 8))

for i, (year, group) in enumerate(grouped_data_elec):
    plt.plot(group['Month'], group['elec'], label = f'{year}', color = colors[i])

plt.title('Eelc by Month')
plt.xlabel('Month')
plt.ylabel('elec')
plt.xticks(range(1, 13), [str(month) for month in range(1, 13)])
plt.legend(title = 'Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# 월별 전력수요합계(sum_load)를 평균내어 그래프로 시각화

monthly_sum_load = train.groupby(['Year', 'Month'])['sum_load'].mean()[:-1].reset_index()  # 2023년 1월 1일 제외
grouped_data_sum_load = monthly_sum_load.groupby('Year')

plt.figure(figsize = (12, 8))

for i, (year, group) in enumerate(grouped_data_sum_load):
    plt.plot(group['Month'], group['sum_load'], label = f'{year}', color = colors[i])

plt.title('Sum load by Month')
plt.xlabel('Month')
plt.ylabel('Sum load')
plt.xticks(range(1, 13), [str(month) for month in range(1, 13)])
plt.legend(title = 'Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# 월별 체감온도(nph_ta_chi)를 평균내어 그래프로 시각화

monthly_nph_ta_chi = train.groupby(['Year', 'Month'])['nph_ta_chi'].mean()[:-1].reset_index()  # 2023년 1월 1일 제외
grouped_data_nph_ta_chi = monthly_nph_ta_chi.groupby('Year')

plt.figure(figsize = (12, 8))

for i, (year, group) in enumerate(grouped_data_nph_ta_chi):
    plt.plot(group['Month'], group['nph_ta_chi'], label = f'{year}', color = colors[i])

plt.title('Sensible Temperature by Month')
plt.xlabel('Month')
plt.ylabel('nph_ta_chi')
plt.xticks(range(1, 13), [str(month) for month in range(1, 13)])
plt.legend(title = 'Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# 월별 강수량(nph_rn_60m)을 평균내어 그래프로 시각화

monthly_nph_rn_60m = train.groupby(['Year', 'Month'])['nph_rn_60m'].mean()[:-1].reset_index()  # 2023년 1월 1일 제외
grouped_data_nph_rn_60m = monthly_nph_rn_60m.groupby('Year')

plt.figure(figsize = (12, 8))

for i, (year, group) in enumerate(grouped_data_nph_rn_60m):
    plt.plot(group['Month'], group['nph_rn_60m'], label = f'{year}', color = colors[i])

plt.title('Precipitation by Month')
plt.xlabel('Month')
plt.ylabel('nph_rn_60m')
plt.xticks(range(1, 13), [str(month) for month in range(1, 13)])
plt.legend(title = 'Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# 전력수요합계(sum_load)를 주중과 주말로 나눠서 시각화

monthly_wn = train.groupby(['Year','Month','week_name'])['sum_load'].mean()[:-1].reset_index()  # 2023년 1월 1일 제외

weekday_data = monthly_wn[monthly_wn['week_name'] == 0]
weekend_data = monthly_wn[monthly_wn['week_name'] == 1]

weekday_grouped = weekday_data.groupby(['Year', 'Month']).first()
weekend_grouped = weekend_data.groupby(['Year', 'Month']).first()

fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15 ,12))

# 주중 그래프
for year, group in weekday_grouped.groupby(level = 0):
    axes[0].plot(group.index.get_level_values('Month'), group['sum_load'], marker = 'o', label = f'{year}')
axes[0].set_title('Weekday sum_load')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('sum_load')
axes[0].legend(title = 'Year')
axes[0].grid(True)

# 주말 그래프
for year, group in weekend_grouped.groupby(level = 0):
    axes[1].plot(group.index.get_level_values('Month'), group['sum_load'], marker = 'o', label = f'{year}')
axes[1].set_title('Weekend sum_load')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('sum_load')
axes[1].legend(title = 'Year')
axes[1].grid(True)

# 전력수요합계(sum_load)를 요일별로 나눠서 시각화
day_data = train.groupby(['Year','weekday'])['sum_load'].mean()[:-1].reset_index() # 2023년 1월 1일 제외

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 12))

for i, year in enumerate(day_data['Year'].unique()):
    year_data = day_data[day_data['Year'] == year]
    axes[i].plot(year_data['weekday'], year_data['sum_load'], marker = 'o', color = 'green')
    axes[i].set_title(f'Sum Load by Weekday for year {year}')
    axes[i].set_xlabel('Weekday')
    axes[i].set_ylabel('Sum Load')
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 전력수요합계(sum_load)를 시간별로 나눠서 시각화
hour_data = train.groupby(['Year','hh24'])['sum_load'].mean()[:-1].reset_index() # 2023년 1월 1일 제외

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 12))

for i, year in enumerate(hour_data['Year'].unique()):
    year_data = hour_data[hour_data['Year'] == year]
    axes[i].plot(year_data['hh24'], year_data['sum_load'], marker = 'o', color = 'green')
    axes[i].set_title(f'Sum Load by hour for year {year}')
    axes[i].set_xlabel('Hour')
    axes[i].set_ylabel('Sum Load')
    axes[i].set_xticks(range(1, 25))
    axes[i].grid(True)

plt.tight_layout()
plt.show()

#%%