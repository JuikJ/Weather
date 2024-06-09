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

from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/Users/jangjuik/Desktop/electric/electric_train_mac.csv', index_col = 0)
test = pd.read_csv('/Users/jangjuik/Desktop/electric/electric_test_mac.csv', index_col = 0)

train.head()

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

# 변수 삭제
train.drop(['tm','num', 'n', 'stn', 'sum_qctr', 'sum_load', 'n_mean_load'],axis = 1, inplace = True)
test.drop(['tm','num', 'stn'], axis = 1, inplace = True)

############################################# EDA

# 1) 전력기상지수 분포
plt.figure(figsize = (10, 6))
sns.histplot(train['elec'], bins = 30, kde = True)
plt.title('Distribution of Elec')
plt.xlabel('Elec')

# 데이터가 왼쪽으로 치우쳐 있음

# 2) 전력기상지수 평균 분포 시각화

# 월별 전력기상지수(elec)를 평균내어 그래프로 시각화

colors = ['magenta', 'green', 'blue']

monthly_elec = train.groupby(['Year', 'Month'])['elec'].mean()[:-1].reset_index()  # 2023년 1월 제외
grouped_data_elec = monthly_elec.groupby('Year')

plt.figure(figsize = (12, 8))

for i, (year, group) in enumerate(grouped_data_elec):
    plt.plot(group['Month'], group['elec'], label = f'{year}', color = colors[i])

plt.title('Elec by Month')
plt.xlabel('Month')
plt.ylabel('elec')
plt.xticks(range(1, 13), [str(month) for month in range(1, 13)])
plt.legend(title = 'Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# 여름에 전력기상지수가 높게 나타나는 것을 알 수 있음

# 전력기상지수(elec)를 주중과 주말로 나눠서 시각화

monthly_wn = train.groupby(['Year','Month','week_name'])['elec'].mean()[:-1].reset_index()  # 2023년 1월 1일 제외

weekday_data = monthly_wn[monthly_wn['week_name'] == 0]
weekend_data = monthly_wn[monthly_wn['week_name'] == 1]

weekday_grouped = weekday_data.groupby(['Year', 'Month']).first()
weekend_grouped = weekend_data.groupby(['Year', 'Month']).first()

fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15 ,12))

# 주중 그래프
for year, group in weekday_grouped.groupby(level = 0):
    axes[0].plot(group.index.get_level_values('Month'), group['elec'], marker = 'o', label = f'{year}')
axes[0].set_title('Weekday elec')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('elec')
axes[0].legend(title = 'Year')
axes[0].grid(True)

# 주말 그래프
for year, group in weekend_grouped.groupby(level = 0):
    axes[1].plot(group.index.get_level_values('Month'), group['elec'], marker = 'o', label = f'{year}')
axes[1].set_title('Weekend sum_load')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('elec')
axes[1].legend(title = 'Year')
axes[1].grid(True)

# 전력기상지수는 주중과 주말 비슷한 양상을 보임

# 전력기상지수(elec)를 요일별로 나눠서 시각화

day_data = train.groupby(['Year','weekday'])['elec'].mean()[:-1].reset_index() # 2023년 1월 1일 제외

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 12))

for i, year in enumerate(day_data['Year'].unique()):
    year_data = day_data[day_data['Year'] == year]
    axes[i].plot(year_data['weekday'], year_data['elec'], marker = 'o', color = 'green')
    axes[i].set_title(f'Elec by Weekday for year {year}')
    axes[i].set_xlabel('Weekday')
    axes[i].set_ylabel('Elec')
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 전력기상지수는 주말에 더 높게 나타나는 모습을 보임. 아마도 주말에는 출근하지 않는 사람이 많다보니 공동주택 전력수요가 증가해서 그런듯함

# 계절별 시간대별 전력기상지수 시각화

conditions = [
    train['Month'].isin([3, 4, 5]),
    train['Month'].isin([6, 7, 8]),
    train['Month'].isin([9, 10, 11]),
    train['Month'].isin([12, 1, 2])
]

choices = [1, 2, 3, 4]

train['Season'] = np.select(conditions, choices, default=0)

plt.figure(figsize=(14, 8))
plot = sns.lineplot(data=train, x='hh24', y='elec', hue='Season', palette='tab10', marker='o')

# 범례 수정
handles, labels = plot.get_legend_handles_labels()
new_labels = ['Spring (1)', 'Summer (2)', 'Fall (3)', 'Winter (4)']
plot.legend(handles=handles, labels=new_labels, title='Season')

plt.title('Electric Weather Index (elec) by Hour and Season')
plt.xlabel('Hour of the Day')
plt.ylabel('Electric Weather Index (elec)')
plt.grid(True)
plt.show()

# 여름에 전력기상지수가 가장 높게 나타나고, 그 다음으로 겨울이 높다.


# 월별 시간대에 따른 전력기상지수(elec)를 시각화

# 서브플롯 생성
fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
axes = axes.flatten()

# 각 월별로 서브플롯에 그래프 그리기
for i in range(1, 13):
    ax = axes[i - 1]
    month_data =train[train['Month'] == i]
    if not month_data.empty:
        sns.lineplot(data=month_data, x='hh24', y='elec', ax=ax, marker='o')
        ax.set_title(f'Month {i}')
        ax.set_xlabel('Hour of the Day')
        ax.set_ylabel('Electric Weather Index (elec)')
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.show()

# 전력부하가 많은 7,8월에 전력기상지수가 가장 높게 나타난다.

# 아파트 매매 가격과 전력 수요량의 관계

# 아파트 ㎡당 매매평균가격(최근 5년)
apartment = pd.read_excel('아파트 ㎡당 매매평균가격_20240603.xlsx')
apartment = apartment.transpose()
apartment = apartment.drop(apartment.index[0]).reset_index()
new_columns = ['DateTime', '강북14개구', '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', 
               '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구', '강남11개구', '양천구', '강서구', '구로구', 
               '금천구', '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구']
apartment.columns = new_columns

apartment['DateTime'] = pd.to_datetime(apartment['DateTime'])

# 2022년 데이터만 필터링
apartment_2022 = apartment[apartment['DateTime'].dt.year == 2022]

# 서울시 모든 구의 평균 매매가격 구하기
seoul_avg_price = apartment_2022.drop(columns=['DateTime', '강북14개구', '강남11개구']).mean().sort_values(ascending = False)

korean_to_english = {
    '종로구': 'jongno-gu',
    '중구': 'jung-gu',
    '용산구': 'yongsan-gu',
    '성동구': 'seongdong-gu',
    '광진구': 'gwangjin-gu',
    '동대문구': 'dongdaemun-gu',
    '중랑구': 'jungrang-gu',
    '성북구': 'seongbuk-gu',
    '강북구': 'gangbuk-gu',
    '도봉구': 'dobong-gu',
    '노원구': 'nowon-gu',
    '은평구': 'eunpyeong-gu',
    '서대문구': 'seodaemun-gu',
    '마포구': 'mapo-gu',
    '양천구': 'yangcheon-gu',
    '강서구': 'gangseo-gu',
    '구로구': 'guro-gu',
    '금천구': 'geumcheon-gu',
    '영등포구': 'yeongdeungpo-gu',
    '동작구': 'dongjak-gu',
    '관악구': 'gwanak-gu',
    '서초구': 'seocho-gu',
    '강남구': 'gangnam-gu',
    '송파구': 'songpa-gu',
    '강동구': 'gangdong-gu'
}
english_district_prices = {korean_to_english[key]: value for key, value in seoul_avg_price.items()}

plt.figure(figsize=(12, 8))
plt.bar(english_district_prices.keys(), english_district_prices.values(), color='lightgreen')
plt.title('Average Housing Prices in Seoul by District')
plt.xlabel('District')
plt.ylabel('Average Price (in ten thousand won)')
plt.xticks(rotation=45, ha='right')
plt.legend(['Average Price'], loc='upper right')
plt.tight_layout()
plt.show()

# 가구 평균 전력 사용량 데이터(서울시 2022-01 ~ 2022-12)
house1 = pd.read_excel("/Users/jangjuik/Desktop/electric/code/가구 평균 월별 전력사용량_210106.xls")
house2 = pd.read_excel("/Users/jangjuik/Desktop/electric/code/가구 평균 월별 전력사용량_210712.xls")
house1 = house1.drop(house1.index[:3])
house1.index = range(len(house1))
new_columns = ['Datetime', '시도', '시군구', '대상가구수(호)', '가구당 평균 전력 사용량(kWh)', '가구당 평균 전기요금(원)']
house1.columns = new_columns

house2 = house2.drop(house2.index[:3])
house2.index = range(len(house2))
new_columns = ['Datetime', '시도', '시군구', '대상가구수(호)', '가구당 평균 전력 사용량(kWh)', '가구당 평균 전기요금(원)']
house2.columns = new_columns

house = pd.concat([house1, house2])

house = house[house['Datetime'] != '년월']
house = house[house['시군구'] != '전체']
house = house.dropna()
house.index = range(len(house))

house['Datetime'] = pd.to_datetime(house['Datetime'], format='%Y%m')

# '대상가구수(호)' 열의 쉼표 제거 후 정수형으로 변환
house['대상가구수(호)'] = house['대상가구수(호)'].str.replace(',', '').astype(int)

# '가구당 평균 전기요금(원)' 열의 쉼표 제거 후 정수형으로 변환
house['가구당 평균 전기요금(원)'] = house['가구당 평균 전기요금(원)'].str.replace(',', '').astype(int)

house['대상가구수(호)'] = house['대상가구수(호)'].astype(int)
house['가구당 평균 전력 사용량(kWh)'] = house['가구당 평균 전력 사용량(kWh)'].astype(int)
house['가구당 평균 전기요금(원)'] = house['가구당 평균 전기요금(원)'].astype(int)

all_gu_df = pd.DataFrame(columns=house.columns)

for gu in house['시군구'].unique():
    gu_df = house[house['시군구'] == gu]
    all_gu_df = pd.concat([all_gu_df, gu_df])

# 시각화

plt.figure(figsize=(14, 8))

# 각 선에 대한 영어 이름 설정 및 색상 팔레트 변경
colors = sns.color_palette("husl", len(korean_to_english))
for i, (gu_korean, gu_english) in enumerate(korean_to_english.items()):
    gu_df = all_gu_df[all_gu_df['시군구'] == gu_korean]
    sns.lineplot(data=gu_df, x='Datetime', y='가구당 평균 전력 사용량(kWh)', label=gu_english, marker='o', color=colors[i])

plt.title('Average electricity usage per household in Seoul by district from January to December')
plt.xlabel('Month')
plt.ylabel('Kwh')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 범례 위치 조정
plt.legend(bbox_to_anchor=(1, 1), ncol=1)

plt.show()

# 2022년의 서울시 구별 아파트 평균 매매가격을 시각화해보니,
# 강남구가 2582만원으로 가장 높고, 금천구가 898만원으로 가장 낮게 나타났다.

# 2022년 1월 ~ 2022년 12월 서울시 구별 가구당 평균 전력 사용량(kWh)을 시각화해보니,
# 8월에는 서초구가 416으로 가장 높고, 관악구가 259로 가장 낮게 나타났다.
# 관악구는 2022년에 1인가구의 수가 약 17만으로, 전체 가구 수의 60%에 해당한다.

# 아파트 평균 매매가격과 전력 사용량에는 상관이 다소 있는 것으로 보인다.
# 매매가격 상위 구역과 전력 사용량 상위 구역은 비슷하게 나타난다.

avg_power_usage = house.groupby('시군구')['가구당 평균 전력 사용량(kWh)'].mean().sort_values(ascending = False)
seoul_avg_price = apartment_2022.drop(columns=['DateTime', '강북14개구', '강남11개구']).mean().sort_values(ascending = False)

df_avg_power_usage = pd.DataFrame(list(avg_power_usage.items()), columns=['시군구', '가구당 평균 전력 사용량(kWh)'])
df_seoul_avg_price = pd.DataFrame(list(seoul_avg_price.items()), columns=['시군구', '평균매매가격'])

merged_df = pd.merge(df_avg_power_usage, df_seoul_avg_price, on='시군구')

# 시군구를 기준으로 데이터프레임 정렬
merged_df.sort_values(by='시군구', inplace=True)

# '시군구' 열을 인덱스로 설정
merged_df.set_index('시군구', inplace=True)

# 상관관계 계산
correlation = merged_df.corr(method='pearson')

# 결과 출력
print(correlation)

# 아파트 평균매매가격과 가구당 평균 전력 사용량에는 약 0.666의 양의 상관이 있다.

############################################# 모델링

#### 2020 ~ 2021년을 학습시켜 2022년 예측해보자

Y = train['elec']
X = train.drop(['elec'], axis = 1)

train_index = X[(X['Year'] >= 2020) & (X['Year'] <= 2021)].index
test_index = X[(X['Year'] >= 2022) & (X['Year'] <= 2023)].index

train_X = X.loc[train_index]
train_Y = Y.loc[train_index]

test_X = X.loc[test_index]
test_Y = Y.loc[test_index]

### 1) XGBoost
xgb_reg = XGBRegressor(n_estimators = 3000, learning_rate = 0.02, max_depth = 7, min_child_weight = 3, colsample_bytree = 0.75, tree_method = 'hist')
xgb_reg.fit(
    train_X, 
    train_Y, 
    early_stopping_rounds=200, 
    eval_metric='rmse', 
    eval_set=[(test_X, test_Y)], 
    verbose=True
)

pred = xgb_reg.predict(test_X)

rmse_xgb = mean_squared_error(pred, test_Y) ** 0.5 
print(rmse_xgb)

r2_xgb = r2_score(pred, test_Y)
print(r2_xgb)

correlation_xgb = np.corrcoef(pred, test_Y)[0, 1]
print(correlation_xgb) # 상관계수는 0.967



### 선형회귀
reg = LinearRegression()
reg.fit(train_X, train_Y)
pred_reg = reg.predict(test_X)

rmse_reg = mean_squared_error(pred_reg, test_Y) ** 0.5 
print(rmse_reg)

r2_reg = r2_score(pred_reg, test_Y)
print(r2_reg)

correlation_reg = np.corrcoef(pred_reg, test_Y)[0, 1]
print(correlation_reg) # 상관계수는 0.75 정도



### Lightgbm
lgb_train = lgb.Dataset(train_X, train_Y)
lgb_test = lgb.Dataset(test_X, test_Y, reference = lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
print('Starting training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_test],
                callbacks=[lgb.early_stopping(stopping_rounds=10)])
print('Making predictions...')
Y_pred = gbm.predict(test_X, num_iteration = gbm.best_iteration)

# 평가
rmse_gbm = mean_squared_error(Y_pred, test_Y) ** 0.5
print(rmse_gbm)

r2_gbm = r2_score(Y_pred, test_Y)
print(r2_gbm)

correlation_gbm = np.corrcoef(Y_pred, test_Y)[0, 1]
print(correlation_gbm) # 상관계수는 0.966 정도


### 모델 성능 비교
correlations = [correlation_xgb, correlation_reg, correlation_gbm]
models = ['XGBoost', 'Linear Regression', 'LightGBM']

plt.figure(figsize=(12, 8))
bars = plt.barh(models, correlations, color=['blue', 'green', 'red'],height=0.6)
plt.xlabel('Correlation Coefficient')
plt.title('Correlation between Predictions and Actual Values')
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 각 바의 값 표시
for bar in bars:
    width = bar.get_width()
    plt.text(width - 0.1, bar.get_y() + bar.get_height() / 2, 
    f'{width : .3f}', va = 'center', ha = 'right', fontsize = 20, fontweight = 'bold', color = 'white')

# 범례에 점수를 추가
handles = [plt.Rectangle((0,0),1,1, color=bar.get_facecolor(), edgecolor=bar.get_edgecolor(), linewidth=bar.get_linewidth()) for bar in bars]
labels = [f'{model}: {corr:.3f}' for model, corr in zip(models, correlations)]
plt.legend(handles, labels, title="Model Scores", loc = 'center right', bbox_to_anchor=(1.3, 1), ncol=1)

plt.show()
#%%