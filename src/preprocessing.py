# -*- coding: utf-8 -*-
import pandas as pd
from src.utilities import get_features

    
# Training set
train_meta = pd.read_csv('data/train.csv').iloc[:1,:]
train = pd.DataFrame()
train['segment_id'] = train_meta['segment_id']
train['time_to_eruption'] = train_meta['time_to_eruption']
train = train.set_index('segment_id')

for seg_id in train.index.values.tolist():
    train_segment = pd.read_csv(f'data/{seg_id}.csv')
    for i in range(1, 11):
        sensor_id = f'sensor_{i}'
        train = get_features(train, seg_id, sensor_id, train_segment[sensor_id])

train.to_csv('data/train_gen.csv')

# Test set
test_meta = pd.read_csv('data/sample_submission.csv').iloc[:1,:]
test = pd.DataFrame()
test['segment_id'] = test_meta['segment_id']
test['time_to_eruption'] = test_meta['time_to_eruption']
test = test.set_index('segment_id')

for seg_id in test.index.values.tolist():
    test_segment = pd.read_csv(f'data/{seg_id}.csv')
    for i in range(1, 11):
        sensor_id = f'sensor_{i}'
        test = get_features(test, seg_id, sensor_id, test_segment[sensor_id])

test.to_csv('data/test_gen.csv')