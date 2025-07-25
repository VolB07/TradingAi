import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

# Klasör oluştur
os.makedirs("result", exist_ok=True)

# Çoklu istekle geçmiş veri çekme
def fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=1000, total=5000):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = int(time.time() * 1000)

    while len(all_data) < total:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_time
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not data or 'code' in data:
                print(f"⚠️ Hata veya boş veri: {data}")
                break
            all_data = data + all_data  # Baştan ekle
            end_time = data[0][0] - 1   # Önceki batch'in başından devam et
            time.sleep(0.3)  # API limitine dikkat
        except Exception as e:
            print(f"⚠️ Veri çekme hatası: {e}")
            break

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df.set_index("open_time", inplace=True)
    df["close"] = df["close"].astype(float)
    return df[["close"]]

# Model ve scaler dosya yolları
model_path = r"C:\Users\VolB\PycharmProjects\TradingAi\core\models\cnn_model.h5"
scaler_path = r"C:\Users\VolB\PycharmProjects\TradingAi\core\models\scaler.pkl"

# Model yükle
model = load_model(model_path, custom_objects={'mse': tf.keras.losses.mse})
print(f"✅ Model yüklendi: {model_path}")

# Scaler yükle
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print(f"✅ Scaler yüklendi: {scaler_path}")

# 15 dakikalık veriyi çek ve test et
interval = "15m"
total_points = 2000
print(f"\n⏳ {interval} verisi çekiliyor...")
df_live = fetch_binance_klines(interval=interval, total=total_points)
print(f"📈 Çekilen veri adedi: {len(df_live)}")

close_prices = df_live[["close"]].values
scaled = scaler.transform(close_prices)
scaled = np.clip(scaled, -1.1, 1.1)

window_size = 60
X_live, y_live = [], []
for i in range(len(scaled) - window_size):
    X_live.append(scaled[i:i + window_size])
    y_live.append(scaled[i + window_size])
X_live = np.array(X_live)
y_live = np.array(y_live)

predicted_scaled = model.predict(X_live, verbose=0)
predicted_price = scaler.inverse_transform(predicted_scaled)
actual_price = scaler.inverse_transform(y_live.reshape(-1, 1))

mae = mean_absolute_error(actual_price, predicted_price)
rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))
r2 = r2_score(actual_price, predicted_price)

print(f"📊 {interval} Orijinal Tahmin - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(actual_price, label='Gerçek Fiyat')
plt.plot(predicted_price, label='Tahmin')
plt.title(f'CNN Model - Canlı BTC/USDT Tahmin ({interval})')
plt.xlabel("Zaman")
plt.ylabel("Fiyat")
plt.legend()
plt.tight_layout()
plt.savefig("result/cnn_model_live_test_results_15m.png")
plt.close()
print("✅ Grafik kaydedildi: result/cnn_model_live_test_results_15m.png")

with open("result/test_results_15m.txt", "w", encoding="utf-8") as f:
    f.write(f"CNN Model Canlı Veri Test Sonuçları - {interval}\n")
    f.write(f"Orijinal Tahminler\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n")
print("✅ Performans metrikleri kaydedildi: result/test_results_15m.txt")
