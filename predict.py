
"""Predict next-hour energy usage for a household.
Requires trained model (models/model_lstm.h5) and saved scalers (models/scalers.pkl).
Run locally after training.
"""
import os, joblib, numpy as np, pandas as pd
from tensorflow.keras.models import load_model

def load_model_and_scalers(model_path):
    model = load_model(model_path)
    scalers = joblib.load(os.path.join(os.path.dirname(model_path),'scalers.pkl'))
    return model, scalers

def prepare_latest_sequence(df, household_id, seq_len=24):
    g = df[df['household_id']==household_id].sort_values('datetime').reset_index(drop=True)
    past = g.iloc[-seq_len:]
    seq = np.zeros((1, seq_len, 3))
    seq[0,:,0] = past['energy_kwh'].values
    seq[0,:,1] = past['temperature'].values
    seq[0,:,2] = past['humidity'].values
    meta = np.array([[int(household_id[1:]), int(past.iloc[-1]['hour'])]])
    return seq, meta

def predict_next(model_path, data_path, household_id='H01'):
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    model, scalers = load_model_and_scalers(model_path)
    seq, meta = prepare_latest_sequence(df, household_id)
    seq_2d = seq.reshape(-1,3)
    seq_2d_scaled = scalers['seq_scaler'].transform(seq_2d)
    seq_scaled = seq_2d_scaled.reshape(1, seq.shape[1], 3)
    meta_scaled = scalers['meta_scaler'].transform(meta)
    pred_scaled = model.predict([seq_scaled, meta_scaled])[0,0]
    pred = scalers['target_scaler'].inverse_transform([[pred_scaled]])[0,0]
    print(f'Predicted next hour energy (kWh) for {household_id}: {pred:.3f}')
    return pred

if __name__ == '__main__':
    predict_next(os.path.join('models','model_lstm.h5'), os.path.join('data','energy_multi_household.csv'), 'H01')
