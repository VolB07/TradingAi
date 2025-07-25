import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
import pickle

# Klasörleri oluştur
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Veri yükleme
df = pd.read_csv(r"C:\Users\VolB\PycharmProjects\TradingAi\datasets\btc_15m_data_2018_to_2025.csv")
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df['open_time'] = pd.to_datetime(df['open_time'])
df.set_index('open_time', inplace=True)
df = df[['close']].copy()
df.dropna(inplace=True)

# Normalizasyon
values = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# CNN veri seti oluşturma
X, y = [], []
window_size = 60
for i in range(len(scaled) - window_size):
    X.append(scaled[i:i + window_size])
    y.append(scaled[i + window_size])
X = np.array(X)
y = np.array(y)

# Eğitim ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model tanımı
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Model eğitimi
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Model ve scaler kaydetme
model_path = "models/cnn_model.h5"
model.save(model_path)
print(f"✅ Model kaydedildi: {model_path}")

scaler_path = "models/scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler kaydedildi: {scaler_path}")

# Test seti tahmini
predicted = model.predict(X_test)

# Scaler yüklenip inverse transform için kullanma (alternatif olarak, zaten elimizde scaler var)
# with open(scaler_path, "rb") as f:
#     loaded_scaler = pickle.load(f)
# predicted = loaded_scaler.inverse_transform(predicted)
# actual = loaded_scaler.inverse_transform(y_test.reshape(-1, 1))

# Burada doğrudan scaler kullanıyoruz
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Grafik çiz ve kaydet
plt.figure(figsize=(10, 5))
plt.plot(actual, label='Gerçek')
plt.plot(predicted, label='Tahmin')
plt.title('CNN Model Test Sonuçları')
plt.legend()
plt.savefig("results/cnn_model_test_results.png")
plt.close()
print("✅ Grafik kaydedildi: results/cnn_model_test_results.png")

# Performans metrikleri hesapla
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

# Sonuçları konsola yazdır
print(f"📊 Test Sonuçları:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Performans metriklerini txt dosyasına kaydet
with open("results/test_results.txt", "w", encoding="utf-8") as f:
    f.write("CNN Model Test Sonuçları\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")
print("✅ Performans metrikleri kaydedildi: results/test_results.txt")
