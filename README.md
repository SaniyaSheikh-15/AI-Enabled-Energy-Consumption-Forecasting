# Energy Consumption Forecasting — LSTM (Multi-Household) - CLEAN

This repo contains a multi-household energy consumption forecasting pipeline built with an LSTM model.
The dataset and LSTM-ready scripts are included. Training isn't executed here (requires TensorFlow).

## Contents
- `data/energy_multi_household.csv` — synthetic multi-household hourly dataset (~6 months, 34560 rows)
- `train.py` — training script (LSTM) - run locally after installing requirements
- `predict.py` — inference script - run locally with trained model
- `models/` — saved model (place trained model here)
- `notebooks/Energy_Consumption_Analysis.ipynb` — EDA starter notebook

## Quickstart
1. Create virtual env and install requirements: `pip install -r requirements.txt`
2. Train (locally): `python train.py`
3. Predict: `python predict.py`

## Notes for reviewers/interviewers
- Model expects past 24 hours of energy/temperature/humidity per household.
- To reproduce results, install TensorFlow (>=2.10) and run `python train.py`.
