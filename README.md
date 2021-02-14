# INGV - Volcanic Eruption Prediction, 20th place solution (620 participants)
Kaggle competition hosted by Italy's Istituto Nazionale di Geofisica e Vulcanologia (INGV).

## Overview
What if scientists could anticipate volcanic eruptions as they predict the weather? While determining rain or shine days in advance is more difficult, weather reports become more accurate on shorter time scales. A similar approach with volcanoes could make a big impact. Just one unforeseen eruption can result in tens of thousands of lives lost. If scientists could reliably predict when a volcano will next erupt, evacuations could be more timely and the damage mitigated.
Currently, scientists often identify "time to eruption" by surveying volcanic tremors from seismic signals. In some volcanoes, this intensifies as volcanoes awaken and prepare to erupt. Unfortunately, patterns of seismicity are difficult to interpret. In very active volcanoes, current approaches predict eruptions some minutes in advance, but they usually fail at longer-term predictions.
The INGV's main objective is to contribute to the understanding of the Earth's system while mitigating the associated risks. Tasked with the 24-hour monitoring of seismicity and active volcano activity across Italy, the INGV seeks to find the earliest detectable precursors that provide information about the timing of future volcanic eruptions.
In this competition, competitors' task was to predict when a volcano's next eruption will occur using a large geophysical dataset collected by sensors deployed on an active volcano.

## Data
The training and test datasets contain readings from several seismic sensors around a volcano, the files can be downloaded from the [competition's page on Kaggle.](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe/data)

## Task
The goal was to predict the time until the next volcanic eruption given several seismic sensor readings. The metric utilized for the evaluation of predictions was the mean absolute error (MAE).

## Organization of files
`data/`
* `train.csv` : metadata for the training files containing and ID code for each data segment, i.e. `segment_id` and the target value `time_to_eruption`
* `sample_submission` : metadata for the test files containing the test `segment_id`
* `1136037770.csv` : example data segment containing ten minutes of logs from ten different sensors arrayed around a volcano
* `train_gen.csv` : generated features from the example training data segment file `1136037770.csv` by runnning `src/preprocessing.py`

`models/`
* contains the 1st and 2nd level models including LightGBM, XGBoost and Neural Network models

`oof_and_sub/`
* out-of-fold (OOF) and test predictions of the 1st level models used as meta-features in the 2nd level stacking
* 1: LightGBM, 2: XGBoost, 3: NN

`src/`
* contains scripts for the data processing

`blending\`
* `sub_lgbm_level21.csv` : test predictions of the first 2nd level LightGBM model
* `sub_lgbm_level21.csv` : test predictions of the second 2nd level LightGBM model
* `avg_blending` : average blending of the 2nd level models' predictions, final submission file in the competition

## Methodology
A detailed write-up of my approach can be found [here](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe/discussion/209766). The visualization of my modelling is depicted below.

![Alt text](https://github.com/leventelippenszky/INGV-Volcanic-Eruption-Prediction/blob/main/ingv_flowchart.PNG)

