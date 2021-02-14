# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from src.utilities import sensor_feature_names
from sklearn import preprocessing
import os
import random
from sklearn.impute import SimpleImputer

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Input, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping


train_all = pd.read_csv('data/train_all.csv')
test_all = pd.read_csv('data/test_all.csv')


# Selected features
features = ['q40',
           'fft_real_kurtosis',
           '_number_peaks__n_3',
           '_number_cwt_peaks__n_1',
           '_number_peaks__n_10',
           '_spkt_welch_density__coeff_5',
           '_spkt_welch_density__coeff_2',
           '_absolute_sum_of_changes',
           '_change_quantiles__f_agg_var__isabs_False__qh_0.2__ql_0.0',
           '_augmented_dickey_fuller__attr_teststat__autolag_AIC',
           'mfcc4',
           'roll150_std_p10',
           'roll50_std_p40']


train_features = sensor_feature_names(features)
train_features.extend(['segment_id','time_to_eruption'])
train = train_all[train_features]


test_features = sensor_feature_names(features)
test_features.append('segment_id')
test = test_all[test_features]
del train_all, test_all


# Normalize the data for the neural network
def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    
    if normalize_opt != None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        elif normalize_opt == 'robust':
            scaler = preprocessing.RobustScaler()
        elif normalize_opt == 'standard':
            scaler = preprocessing.StandardScaler()
        elif normalize_opt == 'max_abs':
            scaler = preprocessing.MaxAbsScaler()
        
        # Fit the scaler to the training data only to avoid train-test contamination
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_valid, X_test


def create_nn_model(input_dim, lrate):
    inp = Input(shape=(input_dim, 1))
    x = BatchNormalization()(inp)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(128, (2), activation='relu', padding="same")(x)
    x = Conv1D(100, (2), activation='relu', padding="same")(x)
    x = Conv1D(64, (2), activation='relu', padding="same")(x)

    x = Flatten()(x)
    
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    tte = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=tte)    
    opt = Nadam(lr = lrate)
    model.compile(optimizer=opt, loss='mae', metrics='mae')
    return model


# Number of CV folds
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state = 1)


# Learning rate for Nadam optimizer
lrate = 0.008
# Batch size
batch_size = 64


# Test features
X_test = test.drop(['segment_id'], axis=1)


# Initialize arrays for the OOF and test predictions
oof_ids = list()
oof_preds = list()
oof_targets = list()
trn_preds = list()
trn_targets = list()
test_preds = np.zeros(len(test))


for trn_index, valid_index in kf.split(train):
    trn, valid = train.iloc[trn_index], train.iloc[valid_index]
    
    
    X_trn = trn.drop(['segment_id','time_to_eruption'], axis=1)
    y_trn = trn['time_to_eruption']
    
    
    X_valid = valid.drop(['segment_id','time_to_eruption'], axis=1)
    y_valid = valid['time_to_eruption']
    
    
    # Normalize the datasets
    X_trn, X_valid, X_test_scaled = normalize(X_trn.copy(), X_valid.copy(), X_test.copy(), 'min_max', [])
    # Impute missing values
    imputer = SimpleImputer().fit(X_trn)
    X_trn = imputer.transform(X_trn)
    X_valid = imputer.transform(X_valid)
    X_test_scaled = imputer.transform(X_test_scaled)
    
    
    # Initialize the NN model
    model = create_nn_model(input_dim=X_trn.shape[1], lrate=lrate)
    
    
    # Set up the callbacks
    cb_checkpoint = ModelCheckpoint("models/model.hdf5", monitor='val_mae', save_weights_only=True, save_best_only=True)
    cb_early_stopping = EarlyStopping(monitor='val_mae', patience=40)
    callbacks = [cb_checkpoint, cb_early_stopping]
    
    
    # Set up NN seeds
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(4)
    random.seed(5)
    tf.random.set_seed(6)
    
    
    model.fit(x=X_trn, y=y_trn,
              epochs=700, callbacks=callbacks,
              batch_size=batch_size, verbose=0,
              validation_data=(X_valid, y_valid))
    
    
    # Load the best model saved to disk
    model.load_weights("model.hdf5")
    
    
    # OOF segment_ids
    oof_ids.extend(valid['segment_id'])
    # OOF predictions
    oof_preds.extend(model.predict(X_valid).flatten())
    # OOF target values
    oof_targets.extend(y_valid)
    
    
    # Training predictions
    trn_preds.extend(model.predict(X_trn).flatten())
    # Training target values
    trn_targets.extend(y_trn)
    
    
    # Test predictions
    test_preds += model.predict(X_test_scaled).flatten()/num_folds

    
# Create a data frame with OOF predictions
oof_df = pd.DataFrame({'segment_id' : oof_ids,
                       'pred' : oof_preds,
                       'target' : oof_targets})
# Save the OOF file
#oof_df.to_csv('oof_and_sub/oof_3.csv', header=True, index=False)
    
    
cv_mae = mean_absolute_error(oof_targets, oof_preds)
print('CV MAE : {}'.format(cv_mae))
train_mae = mean_absolute_error(trn_targets, trn_preds)
print('Train MAE : {}'.format(train_mae))
print('Train validation gap (ideal 0) : {}'.format(cv_mae/train_mae - 1))


## Test predictions
submission = pd.DataFrame()
submission['segment_id'] = test['segment_id']
submission['time_to_eruption'] = np.maximum(0, test_preds)
#submission.to_csv('oof_and_sub/sub_3.csv', header=True, index=False)


