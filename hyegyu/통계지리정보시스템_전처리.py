# !/usr/bin/env python
# @author : hyegyu
# @Date : 2024.06.17
# @coding: utf-8

"""
모델 성능을 높이기 위한 추가 데이터

원천정보 데이터?
https://sgis.kostat.go.kr/에서 배포
2023년 격자단위 거주인구 데이터 및 주택 데이터 받음
    - 왜 거주인구?
    - 거주인구 이동 변화는 생각보다 크지 않음. 생활권이 바뀌는 생활인구, 이동에 대한 유동인구에 비해 잘 변화하지 않는 정보
    - 그리고 분석내용이 "주택을 대상으로 한 전력 예측"이기때문에, 주택 == 거주 데이터라고 볼 수 있음.
1km X 1km 단위의 국가격자
    - 또한, train데이터에는 있어도, test 데이터에는 없음. 물론.. train데이터를 가공하는 방안도 있지만 한번 테스트 용으로 진행.
"""

#############

import pandas as pd
import glob

def read_txt(path):
    data = {}
    for file_path in path:
        file_name = file_path.split('/')[-1]
        prefix = '_'.join(file_name.split('_')[:2])
        
        df = pd.read_csv(file_path, delimiter='^', header=None)
        if prefix not in data:
            data[prefix] = []
        data[prefix].append(df)
    merged_data = {}
    for prefix, dfs in data.items():
        combined_df = pd.concat(dfs)
        combined_df.columns = ['Year', 'Region', 'Code', 'Value']
    aggregated_df = combined_df.groupby('Region', as_index=False)['Value'].sum()
    return aggregated_df


# 파일 경로 설정
pop_paths = glob.glob('./_census_reqdoc_1718784603273/2023년_인구*.txt')
house_paths = glob.glob('./_census_reqdoc_1718784603273/2023년_주택*.txt')

pop_df = read_txt(pop_paths)
house_df = read_txt(house_paths)

pop_df.rename(columns={'Value':'pop_sum'}, inplace=True)
house_df.rename(columns={'Value':'house_sum'}, inplace=True)

sgis_stat = pd.merge(pop_df, house_df)

##################

import geopandas as gpd
import pandas as pd
import glob
import os

# Base path for the files
base_path = './sgis_grid/'

dirs = glob.glob(os.path.join(base_path, '_grid_border_grid_2023_grid_*'))

# Collect all shapefile paths that end with '_1k.shp'
shapefile_paths = []
for dir_path in dirs:
    shapefile_paths.extend(glob.glob(os.path.join(dir_path, '*_1k.shp')))

# Read all shapefiles into dataframes
data_frames = []
for shp_path in shapefile_paths:
    gdf = gpd.read_file(shp_path)
    data_frames.append(gdf)

# Combine all dataframes into one
grid = pd.concat(data_frames, ignore_index=True)

grid = grid.to_crs('EPSG:4326')



stat = pd.merge(grid, sgis_stat, left_on='GRID_1K_CD', right_on='Region', how='right').drop(columns='GRID_1K_CD')


stat.to_csv('pop_house_grid_1k.csv', encoding='CP949', index=False)
