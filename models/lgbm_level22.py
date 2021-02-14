# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_error
from src.utilities import sensor_feature_names


train_all = pd.read_csv('data/train_all.csv')
test_all = pd.read_csv('data/test_all.csv')


# Selected features
features = ['var',
           'q40',
           'fft_real_mad',
           'fft_real_kurtosis',
           '_number_peaks__n_3',
           '_number_cwt_peaks__n_1',
           '_number_peaks__n_10',
           '_spkt_welch_density__coeff_5',
           '_spkt_welch_density__coeff_2',
           '_absolute_sum_of_changes',
           '_change_quantiles__f_agg_var__isabs_False__qh_0.2__ql_0.0',
           '_augmented_dickey_fuller__attr_teststat__autolag_AIC']


train_features = sensor_feature_names(features)
train_features.extend(['segment_id','time_to_eruption'])
train = train_all[train_features]


test_features = sensor_feature_names(features)
test_features.append('segment_id')
test = test_all[test_features]
del train_all, test_all


# LGBM level 1 OOF predictions
oof_1 = pd.read_csv("oof_and_sub/oof_1.csv")
# NN level 1 OOF predictions
oof_3 = pd.read_csv("oof_and_sub/oof_3.csv")
assert ~ (oof_1.segment_id != oof_3.segment_id).any()
train_meta = pd.DataFrame({'segment_id': oof_1.segment_id,
                           'pred_1': oof_1.pred,
                           'pred_2': oof_3.pred})
# Merge OOF predictions as meta-features into the training set
train = train.merge(train_meta, on='segment_id')


# LGBM level 1 SUB predictions
sub_1 = pd.read_csv("oof_and_sub/sub_1.csv")
# NN level 1 SUB predictions
sub_3 = pd.read_csv("oof_and_sub/sub_3.csv")
assert ~ (sub_1.segment_id != sub_3.segment_id).any()
test_meta = pd.DataFrame({'segment_id': sub_1.segment_id,
                          'pred_1': sub_1.time_to_eruption,
                          'pred_2': sub_3.time_to_eruption})
# Merge SUB predictions as meta-features into the test set
test = test.merge(test_meta, on='segment_id')


# Create the CV folds
kf = KFold(n_splits=5, shuffle=True, random_state = 1)


# L2-regularization
reg_lambda = 0.5


# Initialize columns for the data frames
oof_ids = list()
oof_preds = list()
oof_targets = list()
trn_preds = list()
trn_targets = list()


for trn_index, valid_index in kf.split(train):
    trn, valid = train.iloc[trn_index], train.iloc[valid_index]
    
    
    # Initialize the model
    lgb = lgbm.LGBMRegressor(subsample = 0.7,
                             reg_lambda = reg_lambda,
                             random_state = 2,
                             n_jobs = -1)
    
    
    X_trn = trn.drop(['segment_id','time_to_eruption'], axis=1)
    y_trn = trn['time_to_eruption']
    X_valid = valid.drop(['segment_id','time_to_eruption'], axis=1)
    y_valid = valid['time_to_eruption']
    
    # Train the model
    lgb.fit(X_trn, y_trn)
    
    # OOF segment_ids
    oof_ids.extend(valid['segment_id'])
    # OOF predictions
    oof_preds.extend(lgb.predict(X_valid))
    # OOF target values
    oof_targets.extend(y_valid)
    
    # Training predictions
    trn_preds.extend(lgb.predict(X_trn))
    # Training target values
    trn_targets.extend(y_trn)
    
oof_df = pd.DataFrame({'segment_id' : oof_ids,
                        'pred' : oof_preds,
                        'target' : oof_targets})

cv_mae = mean_absolute_error(oof_df['target'], oof_df['pred'])
print('CV MAE : {}'.format(cv_mae))
train_mae = mean_absolute_error(trn_targets, trn_preds)
print('Train MAE : {}'.format(train_mae))
print('Train validation gap (ideal 0) : {}'.format(cv_mae/train_mae - 1))
      

# Train and test sets
X_train = train.drop(['segment_id','time_to_eruption'], axis=1)
y_train = train['time_to_eruption']
X_test = test.drop(['segment_id'], axis=1)

# Train the model on the whole dataset
lgb = lgbm.LGBMRegressor(subsample = 0.7,
                         reg_lambda = reg_lambda,
                         random_state = 2,
                         n_jobs = -1)
lgb.fit(X_train, y_train)
# Make predictions on the test set
y_test = lgb.predict(X_test)


# Submit predictions
submission = pd.DataFrame()
submission['segment_id'] = test['segment_id']
submission['time_to_eruption'] = np.maximum(0, y_test)
# Save the SUB file
#submission.to_csv('sub_lgbm22.csv', header=True, index=False)


