# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import librosa


def get_features(df, segment_id, sensor_id, signal):
    """
    Computes various features from the sensor readings.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The feature set being built.
    segment_id : int
        Id of the data segment.
    sensor_id : str
        String indicating the given sensor.
    signal : pandas.Series
        The readings from the sensor.
    
    Returns
    -------
    df : pandas.DataFrame
        The feature set being built.
    """
    
    # Indicates whether all the sensor readings are missing
    df.loc[segment_id, f'{sensor_id}_tot_missed'] = 1*(signal.isnull().sum() == len(signal))
    
    # Fill missing values with 0
    signal = signal.fillna(0)
    
    # Basic statistics features in the time domain
    df.loc[segment_id, f'{sensor_id}_mean'] = signal.mean()
    df.loc[segment_id, f'{sensor_id}_std'] = signal.std()
    df.loc[segment_id, f'{sensor_id}_var'] = signal.var() 
    df.loc[segment_id, f'{sensor_id}_max'] = signal.max()
    df.loc[segment_id, f'{sensor_id}_min'] = signal.min()
    df.loc[segment_id, f'{sensor_id}_median'] = signal.median()
    df.loc[segment_id, f'{sensor_id}_skew'] = signal.skew()
    df.loc[segment_id, f'{sensor_id}_mad'] = signal.mad()
    df.loc[segment_id, f'{sensor_id}_kurtosis'] = signal.kurtosis()
    df.loc[segment_id, f'{sensor_id}_q90']= np.quantile(signal, 0.90)
    df.loc[segment_id, f'{sensor_id}_q80']= np.quantile(signal, 0.80)
    df.loc[segment_id, f'{sensor_id}_q60']= np.quantile(signal, 0.60) 
    df.loc[segment_id, f'{sensor_id}_q40']= np.quantile(signal, 0.40) 
    df.loc[segment_id, f'{sensor_id}_q20']= np.quantile(signal, 0.20) 
    df.loc[segment_id, f'{sensor_id}_q10']= np.quantile(signal, 0.10)
    
    # Rolling window features
    window_sizes = [50, 150]
    percentiles = np.arange(0,110,10)
    for w in window_sizes:
        for pctl in percentiles:
            df.loc[segment_id, f'{sensor_id}_roll{w}_std_p{pctl}'] = np.percentile(signal.rolling(w).std().dropna().values, pctl)   
    
    # MFCC features
    mfcc = librosa.feature.mfcc(signal.values, n_mfcc = 20)
    mfcc_mean = mfcc.mean(axis = 1)
    for i,j in enumerate(mfcc_mean):
        df.loc[segment_id, f'{sensor_id}_mfcc{i}'] = j
      
    # Basic statistics features in the frequency domain
    f = np.fft.fft(signal)
    f_real = np.real(f)
    f_real = pd.Series(f_real)
    df.loc[segment_id, f'{sensor_id}_fft_real_mean']= f_real.mean()
    df.loc[segment_id, f'{sensor_id}_fft_real_std'] = f_real.std()
    df.loc[segment_id, f'{sensor_id}_fft_real_var'] = f_real.var()
    df.loc[segment_id, f'{sensor_id}_fft_real_max'] = f_real.max()
    df.loc[segment_id, f'{sensor_id}_fft_real_min'] = f_real.min()
    df.loc[segment_id, f'{sensor_id}_fft_real_median'] = f_real.median()
    df.loc[segment_id, f'{sensor_id}_fft_real_skew'] = f_real.skew()
    df.loc[segment_id, f'{sensor_id}_fft_real_mad'] = f_real.mad()
    df.loc[segment_id, f'{sensor_id}_fft_real_kurtosis'] = f_real.kurtosis()
    return df


def sensor_feature_names(feature_names):
    sensor_features = []
    for f_name in feature_names: 
        sensors = ['sensor_' + str(i) for i in range(1,11)]
        sensor_features.extend([s + '_' + f_name for s in sensors])

    return sensor_features









