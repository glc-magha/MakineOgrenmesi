""" Anomali Tespiti (Anomaly Detection)
Anomali tespiti, normal veriden sapmaların (anomalilerin) belirlenmesini amaçlar.
 Finansal dolandırıcılık veya güvenlik tehditlerinin tespitinde kullanılır.

Algoritmalar:
Isolation Forest
One-Class SVM
Autoencoders (Anomali Tespiti için)

1. Isolation Forest
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun
model = IsolationForest(n_estimators=100, contamination=0.05)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin (-1: Anomali, 1: Normal)
df['anomaly'] = y_pred
print(df.head())
2. One-Class SVM
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun
model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.1)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin (-1: Anomali, 1: Normal)
df['anomaly'] = y_pred
print(df.head())
3. Autoencoders - Anomali Tespiti için
import keras
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Veriyi normalize edin
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayırın
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# Autoencoder modelini oluşturun
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Modeli eğitin
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Yeniden yapılandırılmış veriyi tahmin edin
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Yeniden yapılandırma hatalarını hesaplayın
train_errors = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
test_errors = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Anomalileri tespit edin (hata eşiği belirleme)
threshold = np.percentile(train_errors, 95)
train_anomalies = train_errors > threshold
test_anomalies = test_errors > threshold

print("Train Anomalies:", np.sum(train_anomalies))
print("Test Anomalies:", np.sum(test_anomalies))
4. Isolation Forest (Hyperparameters)
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun
model = IsolationForest(n_estimators=200, max_samples='auto', contamination=0.1, random_state=42)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin (-1: Anomali, 1: Normal)
df['anomaly'] = y_pred
print(df.head())
5. One-Class SVM (Kernel Functions)
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun (farklı kernel fonksiyonu ile)
model = OneClassSVM(kernel='sigmoid', gamma=0.01, nu=0.1)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin (-1: Anomali, 1: Normal)
df['anomaly'] = y_pred
print(df.head())
6. Autoencoder (Keras Model for Anomaly Detection)
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Veriyi normalize edin
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayırın
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# Autoencoder modelini oluşturun
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(X_train.shape[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

# Modeli eğitin
model.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Yeniden yapılandırma hatalarını hesaplayın
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_errors = np.mean(np.power(X_train - train_pred, 2), axis=1)
test_errors = np.mean(np.power(X_test - test_pred, 2), axis=1)

threshold = np.percentile(train_errors, 95)
train_anomalies = train_errors > threshold
test_anomalies = test_errors > threshold

print("Train Anomalies:", np.sum(train_anomalies))
print("Test Anomalies:", np.sum(test_anomalies))
7. Isolation Forest (Outlier Detection)
from sklearn.ensemble import IsolationForest
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun
model = IsolationForest(contamination=0.05)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin
df['anomaly'] = y_pred
df_anomalies = df[df['anomaly'] == -1]
print(df_anomalies)
8. One-Class SVM (Scikit-learn Implementation)
from sklearn.svm import OneClassSVM
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Modeli oluşturun
model = OneClassSVM(kernel='linear', nu=0.05)

# Modeli eğitin
model.fit(X)

# Anomalileri tahmin edin
y_pred = model.predict(X)

# Anomalileri belirleyin (-1: Anomali, 1: Normal)
df['anomaly'] = y_pred
print(df.head())
9. Autoencoders (Autoencoder Threshold for Anomaly)
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Veriyi normalize edin
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayırın
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# Autoencoder modelini oluşturun
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(X_train.shape[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

# Modeli eğitin
model.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Yeniden yapılandırma hatalarını hesaplayın
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_errors = np.mean(np.power(X_train - train_pred, 2), axis=1)
test_errors = np.mean(np.power(X_test - test_pred, 2), axis=1)

threshold = np.percentile(train_errors, 90)
train_anomalies = train_errors > threshold
test_anomalies = test_errors > threshold

print("Train Anomalies:", np.sum(train_anomalies))
print("Test Anomalies:", np.sum(test_anomalies))
10. Anomali Tespiti ve Sonuçların Görselleştirilmesi
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler
X = df.drop(columns=['target'])

# Anomalileri tespit et (örneğin Isolation Forest ile)
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
y_pred = model.fit_predict(X)

# Görselleştirme
sns.scatterplot(x=df['feature1'], y=df['feature2'], hue=y_pred, palette='coolwarm')
plt.title('Anomaly Detection')
plt.show()


"""