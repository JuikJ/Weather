#%%
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




############################################# 데이터 및 라이브러리 준비

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from datetime import datetime, timedelta

train = pd.read_csv('electric_train.csv', index_col = 0)
test = pd.read_csv('electric_test.csv', index_col = 0)

len_train = len(train)

############################################# 데이터 전처리

# 컬럼명 변경
train.columns = train.columns.str.replace('electric_train.', '')
test.columns = test.columns.str.replace('electric_test.', '')

# 시간 관련 변수 생성
train['tm'] = pd.to_datetime(train['tm'], format = '%Y-%m-%d %H:%M:%S')

train['Year'] = train['tm'].dt.year
train['Month'] = train['tm'].dt.month
train['day'] = train['tm'].dt.day

test['tm'] = pd.to_datetime(test['tm'], format = '%Y-%m-%d %H:%M:%S')

test['Year'] = test['tm'].dt.year
test['Month'] = test['tm'].dt.month
test['day'] = test['tm'].dt.day

########################################
# 결측치 처리
########################################

train.replace(-99, np.nan, inplace = True)
test.replace(-99, np.nan, inplace = True)

train = train.dropna()
########################################
# sin, cos 시간 변수 생성
########################################

train['sin_time'] = np.sin(2 * np.pi * train.hh24 / 24)
test['sin_time'] = np.sin(2 * np.pi * test.hh24 / 24)


train['cos_time'] = np.cos(2 * np.pi * train.hh24 / 24)
test['cos_time'] = np.cos(2 * np.pi * test.hh24 / 24)

########################################
# 불쾌지수 변수 생성
########################################

train['THI'] = 9/5 * train['nph_ta'] - 0.55 * (1 - train['nph_hm'] / 100) * (9/5 * train['nph_ta'] - 26) + 32
test['THI'] = 9/5 * test['nph_ta'] - 0.55 * (1 - test['nph_hm'] / 100) * (9/5 * test['nph_ta'] - 26) + 32

########################################
# 기온, 습도, 이동평균 변수 생성
########################################

# 24 * 7 = 168, 7일치의 관측치를 사용하여 이동평균 계산
train['temperature_7d_ma'] = train['nph_ta'].rolling(window = 168, min_periods= 1).mean()
test['temperature_7d_ma'] = test['nph_ta'].rolling(window = 168, min_periods= 1).mean()

train['humidity_7d_ma'] = train['nph_hm'].rolling(window = 168, min_periods= 1).mean()
test['humidity_7d_ma'] = test['nph_hm'].rolling(window = 168, min_periods= 1).mean()

########################################
# 폭염 여부
########################################

train['Heatwave'] = (train['nph_ta_chi'] >= 33).astype(int)
test['Heatwave'] = (test['nph_ta_chi'] >= 33).astype(int)


########################################
# 열 지수
########################################

def calculate_heat_index(celsius_temp, relative_humidity):
    c = [-42.379, 2.04901523, 10.14333127, -0.22475541,
         -6.83783e-3, -5.481717e-2, 1.22874e-3,
         8.5282e-4, -1.99e-6]

    hi = (
        c[0] + c[1] * celsius_temp + c[2] * relative_humidity +
        c[3] * celsius_temp * relative_humidity +
        c[4] * celsius_temp**2 + c[5] * relative_humidity**2 +
        c[6] * celsius_temp**2 * relative_humidity +
        c[7] * celsius_temp * relative_humidity**2 +
        c[8] * celsius_temp**2 * relative_humidity**2
    )

    return hi

train['heat_index'] = train.apply(lambda row : calculate_heat_index(train['nph_ta'], train['nph_hm']), axis =1)

# 변수 삭제
train.drop(['tm','hh24','num', 'n', 'stn', 'sum_qctr', 'sum_load', 'n_mean_load'],axis = 1, inplace = True)
test.drop(['tm','hh24','num', 'stn'], axis = 1, inplace = True)

############################################# EDA


sample_data = train.sample(n = 10000, random_state = 42)
corr_matrix = sample_data.corr()

plt.figure(figsize = (12, 10))
sns.heatmap(corr_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', linewidth = 0.5)


############################################# 모델링

Y = train['elec']
X = train.drop(['elec'], axis = 1)

from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from xgboost import XGBRegressor

xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

pred = xgb_reg.predict(X_val)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(pred, y_val))
correlation = np.corrcoef(pred, y_val)[0, 1]
print(correlation) # 상관계수


#%%