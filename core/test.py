import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from constants import Constants
from utils.prepare_datasets import PrepareDatasets


class Test():
    def __init__(self, model_path, dataset_file_path, save_path):
        self.model_path = model_path
        self.scaler = Constants.SCALER
        self.sequence_length = Constants.SEQUENCE_LENGTH
        self.dataset_file_path = dataset_file_path
        self.prepare_datasets = PrepareDatasets(file_path=self.dataset_file_path)
        self.crypto_prices = self.prepare_datasets.load_crypto_data_from_csv()
        self.save_path = save_path
        self.future_days = 30 * 24 * 4

    def test_load_model(self):
        _, _, X_test, _, _, _, _ = self.prepare_datasets.split_train_test()
        model = load_model(self.model_path)
        # 📈 Test verisi için tahmin yap
        predictions = model.predict(X_test)
        predictions_price = self.scaler.inverse_transform(predictions.reshape(-1, 1))

        # 📆 30 gün ileriye tahmin yap (15 dakikalık adımlarla)
        # 30 gün * 24 saat * 4 (15 dakikalık adımlar)
        future_predictions = []
        last_sequence = X_test[-1].reshape(1, self.sequence_length, 1)

        sayac=0
        for _ in range(self.future_days):
            sayac+=1
            print(sayac)
            next_pred = model.predict(last_sequence)[0, 0]
            next_pred_original = self.scaler.inverse_transform([[next_pred]])[0, 0]
            future_predictions.append(next_pred_original)
            next_scaled = (next_pred_original - self.scaler.mean_[0]) / self.scaler.scale_[0]
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_scaled

        # 📆 Gelecek tarihleri oluştur
        future_dates = pd.date_range(self.crypto_prices.index[-1], periods=self.future_days + 1, freq='15T')[1:]

        # 📊 Grafik Çizimi
        fig = plt.figure(figsize=(16, 9), dpi=200)
        plt.plot(self.crypto_prices.index[-len(predictions):], self.crypto_prices['Close'][-len(predictions):],
                 label="Gerçek Fiyat", color="royalblue", linewidth=2)
        plt.plot(self.crypto_prices.index[-len(predictions):], predictions_price,
                 label="Test Tahmini", linestyle="--", color="orange", linewidth=2)
        plt.plot(future_dates, future_predictions,
                 label="30 Günlük Tahmin", linestyle=":", color="red", linewidth=3)

        plt.title("Bitcoin Fiyat Tahmini (15 Dakikalık Veri)", fontsize=18, fontweight='bold')
        plt.xlabel("Tarih", fontsize=14)
        plt.ylabel("Fiyat (USD)", fontsize=14)
        plt.legend(fontsize=12, loc='upper left', frameon=False)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        plt.savefig(self.save_path, dpi=300, bbox_inches="tight")
        print(f"Grafik kaydedildi: {self.save_path}")


dataset_file_path = r"C:\Users\VolB\PycharmProjects\TradingAi\datasets\btc_15m_data_2018_to_2025.csv"
Test(
    model_path=os.path.join("models", "bitcoin_price_lstm_15m_model1.h5"),
    dataset_file_path=dataset_file_path,
    save_path=r"C:\Users\VolB\PycharmProjects\TradingAi\results\test.png"
).test_load_model()
