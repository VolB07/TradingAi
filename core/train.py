import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

from utils.prepare_datasets import PrepareDatasets
from constants import Constants


class Train:
    def __init__(self, model_path, dataset_file_path):
        self.model_path = model_path
        self.dataset_file_path = dataset_file_path
        self.prepare_datasets = PrepareDatasets(file_path=self.dataset_file_path)
        self.crypto_prices = self.prepare_datasets.load_crypto_data_from_csv()
        self.epochs = 150
        self.batch_size = 64
        self.patience = 20

    def train(self):
        X_train, y_train, X_test, y_test, X, y, train_size = self.prepare_datasets.split_train_test()

        print("Kaydedilmiş model bulunamadı. Yeni bir model oluşturuluyor...")
        model = Sequential([
            LSTM(128, return_sequences=True, activation='tanh', input_shape=(
                Constants.SEQUENCE_LENGTH, 1),
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)
                 ),
            Dropout(0.3),  # Dropout oranı artırıldı
            LSTM(64, return_sequences=False,
                 activation='tanh',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)
                 ),
            Dropout(0.3),  # Dropout oranı artırıldı
            Dense(25, activation='relu'),
            Dropout(0.3),  # Yeni Dropout katmanı eklendi
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error"
        )

        # Erken durdurma callback'i (patience artırıldı)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        # Modeli eğit
        history = model.fit(X_train, y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping]
                            )

        # Modeli kaydet
        save_model(model, self.model_path)
        print("Model kaydedildi")


# Eğitim başlat
dataset_file_path = r"C:\Users\VolB\PycharmProjects\TradingAi\datasets\btc_15m_data_2018_to_2025.csv"
Train(
    model_path=os.path.join("models", "bitcoin_price_lstm_15m_model1.h5"),
    dataset_file_path=dataset_file_path
).train()