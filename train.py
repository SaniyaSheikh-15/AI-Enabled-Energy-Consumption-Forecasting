
"""Train LSTM model for multi-household energy forecasting.
Run locally after installing requirements (tensorflow, scikit-learn, etc.).
"""
import os, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_sequences(df, seq_len=24):
    seqs = []
    for hid, group in df.groupby('household_id'):
        g = group.sort_values('datetime').reset_index(drop=True)
        for i in range(seq_len, len(g)):
            past = g.loc[i-seq_len:i-1]
            target = g.loc[i,'energy_kwh']
            seqs.append((past['energy_kwh'].values, past['temperature'].values, past['humidity'].values,
                         int(hid[1:]), int(g.loc[i-1,'hour']), target))
    return seqs

def train(data_path, model_path):
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    seqs = create_sequences(df, seq_len=24)
    N = len(seqs)
    X_seq = np.zeros((N,24,3))
    X_meta = np.zeros((N,2))
    y = np.zeros((N,))
    for i,s in enumerate(seqs):
        X_seq[i,:,0] = s[0]
        X_seq[i,:,1] = s[1]
        X_seq[i,:,2] = s[2]
        X_meta[i,0] = s[3]
        X_meta[i,1] = s[4]
        y[i] = s[5]
    seq_scaler = MinMaxScaler()
    X_seq_2d = X_seq.reshape(-1,3)
    X_seq_2d_scaled = seq_scaler.fit_transform(X_seq_2d)
    X_seq_scaled = X_seq_2d_scaled.reshape(N,24,3)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1,1)).reshape(-1)
    meta_scaler = StandardScaler()
    X_meta_scaled = meta_scaler.fit_transform(X_meta)
    X_train, X_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(X_seq_scaled, X_meta_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True)
    # build model
    seq_in = Input(shape=(24,3), name='seq_input')
    x = LSTM(64, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    meta_in = Input(shape=(2,), name='meta_input')
    m = Dense(16, activation='relu')(meta_in)
    combined = Concatenate()([x,m])
    d = Dense(32, activation='relu')(combined)
    out = Dense(1, activation='linear')(d)
    model = Model(inputs=[seq_in, meta_in], outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    model.fit([X_train, X_meta_train], y_train, validation_data=([X_test, X_meta_test], y_test),
              epochs=30, batch_size=128, callbacks=[es, mc], verbose=2)
    joblib.dump({'seq_scaler': seq_scaler, 'target_scaler': target_scaler, 'meta_scaler': meta_scaler}, os.path.join(os.path.dirname(model_path),'scalers.pkl'))

if __name__ == '__main__':
    train(os.path.join('data','energy_multi_household.csv'), os.path.join('models','model_lstm.h5'))
