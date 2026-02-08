import pandas as pd
import pickle # atau import joblib jika pakai joblib

# Load model
with open('model_fraud_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

# Cek fitur yang diharapkan
try:
    # Untuk Scikit-Learn versi baru / XGBoost
    print("=== MODEL MENGHARAPKAN KOLOM INI ===")
    print(model.feature_names_in_)
except:
    try:
        # Alternatif untuk XGBoost
        print(model.get_booster().feature_names)
    except:
        print("Gagal membaca nama fitur otomatis. Cek manual di notebook training kamu (X_train.columns)")