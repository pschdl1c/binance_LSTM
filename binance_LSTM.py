import requests
import datetime
import time
import os 

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_SPLIT = 40000
PAST_HISTORY = 24*5 # last 5 days
FUTURE_TARGET = 1   # next hour
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 39880

def download_save_klines(history_range: int=50, limit_param: int=1000, save: bool=True) -> pd:
    base = 'https://api.binance.com'    # 'https://fapi.binance.com'
    req = '/api/v3/klines'  # '/fapi/v1/klines'
    url = base + req

    # current time
    date_time = datetime.datetime.now()
    unix_date_end = 1000*int(time.mktime(date_time.timetuple()))

    end_time = unix_date_end
    df = pd.DataFrame([])
    for _ in tqdm(range(history_range), desc='Total'):
        params = {'symbol': 'ETHUSDT',
                    'interval': '1h',
                    'endTime': end_time,
                    'limit': limit_param}

        r = requests.get(url, params=params)
        if r.status_code == 200:
            # w take a data packet for the previous time period
            end_time = r.json()[0][0] - 3600 * 1000
        else:
            print(r.json())
        df = pd.concat([pd.DataFrame(r.json()), df])
        
    df.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume',
                    'closeTime', 'quoteAssetVolume', 'numTrade', 'takerBuyBaseAssetVolume',
                    'takerBuyQuoteAssetVolume', 'ignore']
    df = df.reset_index(drop=True)
    if save:
        df.to_csv('data/klines.csv', index=False)
    return df


def preprocess_data(df: pd) -> pd:
    # convert time data
    df['openTime'] = pd.to_datetime(df['openTime']/1000, unit='s')
    df['closeTime'] = pd.to_datetime(df['closeTime']/1000, unit='s')
    # convert to float
    obj_to_float_list = df.select_dtypes(object).columns
    df[obj_to_float_list] = df[obj_to_float_list].astype(np.float64)
    return df

def feature_engineering(df: pd) -> pd:
    # feature engineering for date
    for time_name in df.select_dtypes('datetime64').columns.values:
        df[f'{time_name}_year'] = df[time_name].dt.year
        df[f'{time_name}_month'] = df[time_name].dt.month
        df[f'{time_name}_day'] = df[time_name].dt.day
        df[f'{time_name}_hour'] = df[time_name].dt.hour
        df[f'{time_name}_minute'] = df[time_name].dt.minute
        df[f'{time_name}_second'] = df[time_name].dt.second
    # cyclic features
    def encode(data, col, max_val):
        data[f'{col}_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[f'{col}_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    df_cyclic = encode(df, 'openTime_month', 12)
    df_cyclic = encode(df_cyclic, 'closeTime_month', 12)
    df_cyclic = encode(df_cyclic, 'openTime_day', 31)
    df_cyclic = encode(df_cyclic, 'closeTime_day', 31)
    df_cyclic = encode(df_cyclic, 'openTime_hour', 23)
    df_cyclic = encode(df_cyclic, 'closeTime_hour', 23)
    df_cyclic = encode(df_cyclic, 'closeTime_minute', 59)
    df_cyclic = encode(df_cyclic, 'closeTime_second', 59)

    features_list = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numTrade', 'takerBuyBaseAssetVolume',
                     'takerBuyQuoteAssetVolume', 'openTime_year','openTime_day', 'closeTime_year', 'openTime_day_sin',
                     'openTime_day_cos', 'openTime_month_sin', 'openTime_month_cos', 'closeTime_month_sin', 'closeTime_month_cos',
                     'closeTime_day_sin', 'closeTime_day_cos', 'openTime_hour_sin', 'openTime_hour_cos','closeTime_hour_sin',
                     'closeTime_hour_cos', 'closeTime_minute_sin', 'closeTime_minute_cos', 'closeTime_second_sin', 'closeTime_second_cos']
    return df_cyclic[features_list]


def create_train_valid_dataset(data: pd):
    dataset = data.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        for i in tqdm(range(start_index, end_index)):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
        return np.array(data), np.array(labels)
    
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 3], 0,
                                                       TRAIN_SPLIT, PAST_HISTORY,
                                                       FUTURE_TARGET, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 3],
                                                   TRAIN_SPLIT, None, PAST_HISTORY,
                                                   FUTURE_TARGET, STEP,
                                                   single_step=True)
    return x_train_single, y_train_single, x_val_single, y_val_single, data_mean, data_std


def create_tf_dataset(x_train_single, y_train_single, x_val_single, y_val_single):
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    return train_data_single, val_data_single


def create_lstm_model(train_data_single: tf) -> tf.keras:
    inputs = tf.keras.Input(train_data_single.element_spec[0].shape[-2:])
    bidirectional1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, ))(inputs)
    lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)(bidirectional1)
    lstm2 = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)

    # merge layers
    merged_layers = tf.keras.layers.concatenate([lstm1, lstm2])

    bidirectional12 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, ))(merged_layers)
    lstm12 = tf.keras.layers.LSTM(128, )(bidirectional12)
    lstm22 = tf.keras.layers.LSTM(128, )(merged_layers)

    merged_layers2 = tf.keras.layers.add([lstm12, lstm22])

    outputs = tf.keras.layers.Dense(1)(merged_layers2) 
    lstm_double = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")
    # tf.keras.utils.plot_model(single_step_history, "model.png",show_shapes=True)
    lstm_double.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    # single_step_model.summary()
    return lstm_double


def show_plot(plot_data, delta, title):
    def create_time_steps(length):
        return list(range(-length, 0))
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    plt.ylabel('close')
    return plt


if __name__ == '__main__':
    # data preprocessing
    df = download_save_klines()
    df = preprocess_data(df)
    data = feature_engineering(df)
    x_train_single, y_train_single, x_val_single, y_val_single, data_mean, data_std  = create_train_valid_dataset(data)
    train_data_single, val_data_single = create_tf_dataset(x_train_single, y_train_single, x_val_single, y_val_single)
    lstm_double = create_lstm_model(train_data_single)
    
    # model learn
    EPOCHS = 1
    EVALUATION_INTERVAL = x_train_single.shape[0] // BATCH_SIZE
    VAL_STEPS = x_val_single.shape[0] // BATCH_SIZE

    single_step_history = lstm_double.fit(train_data_single, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_single,
                                          validation_steps=VAL_STEPS,
                                          shuffle=False)
    
    # visualization of test results
    for i in range(0, 20, 8):
        plot = show_plot([x_val_single[-(i+1)][:, 3], y_val_single[-(i+1)],
                        lstm_double.predict(np.array([x_val_single[-(i+1)]]))], 1,
                        f'Single Step Prediction for last 5 days (120h)\n[-{i} hours from the last observation]')
        plot.show()
    
    # prediction for the next hour from the latest data
    df_last = feature_engineering(preprocess_data(download_save_klines(history_range=1, limit_param=120, save=False))).values
    df_last = (df_last-data_mean)/data_std
    plt.plot(df_last[:, 3], label='close')
    plt.scatter(121, lstm_double.predict(np.array([df_last]))[0][0], marker='o', c='g', label='LSTM prediction')
    plt.legend()
    plt.xlabel('Time-Step')
    plt.ylabel('close')
    plt.title('Prediction for the next hour')
    plt.show()