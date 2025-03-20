import pandas as pd
import numpy as np
from constants import Constants
class PrepareDatasets:
    def __init__(self,file_path):
        self.file_path = file_path

    def load_crypto_data_from_csv(self):
    # CSV dosyasını oku ve 'Open time' sütununu tarih formatına çevir
        df = pd.read_csv(self.file_path, parse_dates=["Open time"], index_col="Open time")
        # 'Close' sütununu kullanarak fiyat verisini al
        crypto_prices = df[["Close"]].ffill()
        return crypto_prices

    def preprocess(self):
        scaled_data = Constants.SCALER.fit_transform(self.load_crypto_data_from_csv())
        return scaled_data
    
    def create_sequences(self):
        X, y = [], []
        data = self.preprocess()
        for i in range(len(data) - Constants.SEQUENCE_LENGTH):
            X.append(data[i:i + Constants.SEQUENCE_LENGTH])
            y.append(data[i + Constants.SEQUENCE_LENGTH, 0])  # Sadece fiyatı tahmin edeceğiz
        return np.array(X), np.array(y)

    def split_train_test(self):
        X, y = self.create_sequences()
        # Eğitim ve test verisine ayırma
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, y_train,X_test,y_test,X,y,train_size