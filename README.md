# INGV - Volcanic Eruption Prediction, 20th place solution
Kaggle competition hosted by Italy's Istituto Nazionale di Geofisica e Vulcanologia (INGV).

## Overview
What if scientists could anticipate volcanic eruptions as they predict the weather? While determining rain or shine days in advance is more difficult, weather reports become more accurate on shorter time scales. A similar approach with volcanoes could make a big impact. Just one unforeseen eruption can result in tens of thousands of lives lost. If scientists could reliably predict when a volcano will next erupt, evacuations could be more timely and the damage mitigated.
Currently, scientists often identify "time to eruption" by surveying volcanic tremors from seismic signals. In some volcanoes, this intensifies as volcanoes awaken and prepare to erupt. Unfortunately, patterns of seismicity are difficult to interpret. In very active volcanoes, current approaches predict eruptions some minutes in advance, but they usually fail at longer-term predictions.
The INGV's main objective is to contribute to the understanding of the Earth's system while mitigating the associated risks. Tasked with the 24-hour monitoring of seismicity and active volcano activity across Italy, the INGV seeks to find the earliest detectable precursors that provide information about the timing of future volcanic eruptions.
In this competition, competitors' task was to predict when a volcano's next eruption will occur using a large geophysical dataset collected by sensors deployed on active volcanos.

## Data
The training and test datasets contain readings from several seismic sensors around a volcano, the files can be downloaded from the [competition's Kaggle page.](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe/data)

## Task
The goal was to predict the time until the next volcanic eruption given several seismic sensor readings. The metric utilized for the evaluation of predictions was the mean absolute error (MAE).

## Organization of files
