#!/usr/bin/env python
# coding: utf-8

import os
import sys

import uuid
import pickle

from datetime import datetime
import pandas as pd
import numpy as np



def read_data(filename, categorical, year, month):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
        
    df_result.to_parquet(
        output_file,
        index=False
    )


def get_paths(taxi_type, year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_dir = f'output/{taxi_type}'
    output_file = f'{output_dir}/{year:04d}-{month:02d}.parquet'

    os.makedirs(output_dir, exist_ok=True)

    return input_file, output_file



def run():
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    categorical = ['PULocationID', 'DOLocationID']
    input_file, output_file = get_paths(taxi_type, year, month)

    dv, model = load_model()
    df = read_data(input_file, categorical, year, month)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    print(np.mean(y_pred))

    save_results(df, y_pred, output_file)

if __name__ == '__main__':
    run()




