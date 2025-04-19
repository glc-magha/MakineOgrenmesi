"""Zaman Serisi Analizi (Time Series Analysis)
Zaman serisi, zamanla değişen veriler üzerinde analiz yapmayı hedefler.
Hedef, gelecekteki değerleri tahmin etmektir.

Algoritmalar:
ARIMA (Autoregressive Integrated Moving Average)
LSTM (Long Short-Term Memory) ağları
Prophet


1. ARIMA Modeli ile Zaman Serisi Tahmin (Python)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Zaman serisi verisi
data = pd.read_csv('your_timeseries_data.csv', index_col='date', parse_dates=True)
data = data['value']  # 'value' kolonu zaman serisi verisidir

# ARIMA Modeli (p, d, q parametrelerini ayarlayın)
model = ARIMA(data, order=(5, 1, 0))
model_fit = model.fit()

# Tahminler
forecast = model_fit.forecast(steps=10)  # Gelecek 10 adım için tahmin
print(forecast)

# Grafik
plt.plot(data, label='Gerçek Veri')
plt.plot(np.arange(len(data), len(data)+10), forecast, label='Tahmin Edilen', color='red')
plt.legend()
plt.show()

2. LSTM Modeli ile Zaman Serisi Tahmini (Python)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Zaman serisi verisini yükleyin
data = pd.read_csv('your_timeseries_data.csv', index_col='date', parse_dates=True)
data = data['value']  # 'value' kolonu zaman serisi verisidir

# Veriyi normalize etme
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Eğitim ve test verisine ayırma
train_size = int(len(data) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# LSTM için veri hazırlama fonksiyonu
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Tahmin yapma
predicted = model.predict(X_test)

# Sonuçları görselleştirme
predicted_values = scaler.inverse_transform(predicted)
true_values = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(true_values, label='Gerçek Değer')
plt.plot(predicted_values, label='Tahmin Edilen', color='red')
plt.legend()
plt.show()


3. Prophet ile Zaman Serisi Tahmini (Python)
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Veriyi yükleyin ve 'date' ve 'value' kolonlarını hazırlayın
data = pd.read_csv('your_timeseries_data.csv')
data = data[['date', 'value']]
data.columns = ['ds', 'y']  # Prophet için gerekli kolon isimleri

# Prophet modelini oluşturun
model = Prophet()
model.fit(data)

# Gelecek için tahmin yapma
future = model.make_future_dataframe(data, periods=365)  # 365 gün tahmin
forecast = model.predict(future)

# Tahminleri görselleştirme
model.plot(forecast)
plt.show()


4. ARIMA Modeli ile Zaman Serisi Tahmin (Karmaşık Versiyon)
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Zaman serisi verisini yükleme
data = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')

# Veriyi görselleştirme
data.plot()
plt.show()

# ARIMA Modelini oluşturma ve eğitme
model = ARIMA(data, order=(3, 1, 2))  # p, d, q parametrelerini ayarlayın
model_fit = model.fit()

# Modelin özetini yazdırma
print(model_fit.summary())

# Gelecek 30 adım için tahmin yapma
forecast = model_fit.forecast(steps=30)
plt.plot(forecast)
plt.show()


5. LSTM Modeli ile Çoklu Zaman Serisi Tahmin
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Çoklu zaman serisi verisini yükleyin
data = pd.read_csv('your_multivariate_timeseries_data.csv')
data = data[['value1', 'value2', 'value3']]  # Seçilen kolonlar

# Veriyi normalleştirme
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Eğitim ve test verisine ayırma
train_size = int(len(data) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# LSTM için veri hazırlama
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back)])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(look_back, 3)))
model.add(Dense(units=3))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Tahmin yapma
predicted = model.predict(X_test)

# Sonuçları görselleştirme
predicted_values = scaler.inverse_transform(predicted)
true_values = scaler.inverse_transform(y_test)

plt.plot(true_values[:, 0], label='Gerçek Değer')
plt.plot(predicted_values[:, 0], label='Tahmin Edilen', color='red')
plt.legend()
plt.show()


6. ARIMA ile Mevsimsel Zaman Serisi Tahmini
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Zaman serisi verisini yükleyin
data = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')

# Mevsimsel ARIMA (SARIMA) modelini oluşturun
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Modelin özetini yazdırma
print(model_fit.summary())

# Gelecek 30 adım için tahmin yapma
forecast = model_fit.forecast(steps=30)
plt.plot(forecast)
plt.show()

7. LSTM Modeli ile Tek Değişkenli Zaman Serisi Tahmini
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Zaman serisi verisini yükleyin
data = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')
data = data['value']

# Veriyi normalleştirme
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Eğitim ve test verisine ayırma
train_size = int(len(data) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# LSTM için veri hazırlama
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Tahmin yapma
predicted = model.predict(X_test)

# Sonuçları görselleştirme
predicted_values = scaler.inverse_transform(predicted)
true_values = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(true_values, label='Gerçek Değer')
plt.plot(predicted_values, label='Tahmin Edilen', color='red')
plt.legend()
plt.show()


8. Prophet ile Haftalık Zaman Serisi Tahmini
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

9. ARIMA Modeli ile Zaman Serisi Tahmini (Özelleştirilmiş Parametrelerle)
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Veriyi yükleyin
data = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')

# ARIMA Modeli - Özelleştirilmiş p, d, q değerleri ile
model = ARIMA(data, order=(2, 1, 2))  # p=2, d=1, q=2
model_fit = model.fit()

# Modelin özetini yazdırma
print(model_fit.summary())

# Gelecek 30 adım için tahmin yapma
forecast = model_fit.forecast(steps=30)

# Sonuçları görselleştirme
plt.plot(data, label='Gerçek Veriler')
plt.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast, label='Tahmin Edilen', color='red')
plt.legend()
plt.show()


10. Prophet ile Çoklu Zaman Serisi Tahmini
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Veriyi yükleyin
data = pd.read_csv('your_timeseries_data.csv')
data = data[['date', 'value']]
data.columns = ['ds', 'y']  # Prophet için gerekli kolon isimleri

# Prophet Modelini oluşturun
model = Prophet()
model.fit(data)

# Gelecek 365 gün için tahmin yapma
future = model.make_future_dataframe(data, periods=365)
forecast = model.predict(future)

# Tahminleri görselleştirme
model.plot(forecast)
plt.show()
"""