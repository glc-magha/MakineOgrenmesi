"""Derin Öğrenme (Deep Learning)
Derin öğrenme, çok katmanlı yapay sinir ağlarını kullanarak büyük veri setleri üzerinde özellikleri otomatik olarak öğrenir.

Yapay Sinir Ağları (ANN): Temel sinir ağı yapıları.

Konvolüsyonel Sinir Ağları (CNN): Görüntü tanıma ve analiz için kullanılır.

Yinelemeli Sinir Ağları (RNN): Zaman serisi verisi ve dil işleme için kullanılır.

Algoritmalar:
CNN (Convolutional Neural Networks)
RNN (Recurrent Neural Networks)
LSTM (Long Short-Term Memory)
GAN (Generative Adversarial Networks)
Transformer


1. Yapay Sinir Ağı (ANN) - Basit Örnek
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Model oluşturuluyor
model = Sequential([
    Dense(64, input_dim=8, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma için
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
# varsayalım X_train ve y_train verisi mevcut
# model.fit(X_train, y_train, epochs=10, batch_size=32)
2. Konvolüsyonel Sinir Ağı (CNN) - Görüntü Tanıma
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN modeli oluşturuluyor
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 sınıf için
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
# varsayalım X_train ve y_train verisi mevcut
# model.fit(X_train, y_train, epochs=10, batch_size=32)
3. Yinelemeli Sinir Ağı (RNN) - Zaman Serisi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# RNN modeli oluşturuluyor
model = Sequential([
    SimpleRNN(50, input_shape=(10, 1), activation='relu'),
    Dense(1)  # Çıktı
])

model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
# varsayalım X_train ve y_train verisi mevcut
# model.fit(X_train, y_train, epochs=10, batch_size=32)
4. LSTM (Long Short-Term Memory) - Zaman Serisi Verisi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LSTM modeli oluşturuluyor
model = Sequential([
    LSTM(50, input_shape=(10, 1), activation='relu'),
    Dense(1)  # Çıktı
])

model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
# varsayalım X_train ve y_train verisi mevcut
# model.fit(X_train, y_train, epochs=10, batch_size=32)
5. Generative Adversarial Networks (GAN) - Basit GAN Örneği
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generator
def build_generator():
    model = Sequential([
        Dense(128, input_dim=100, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(28*28, activation='sigmoid', reshape=(28, 28))
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Dense(1024, input_dim=28*28, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
generator = build_generator()
discriminator = build_discriminator()

# Discriminator'ı eğitilebilir yapmak
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Eğitim döngüsü
for epoch in range(10000):
    noise = np.random.normal(0, 1, (32, 100))
    generated_images = generator.predict(noise)
    real_images = np.random.rand(32, 28, 28)  # Gerçek görüntüleri burada kullanın

    labels_real = np.ones((32, 1))
    labels_fake = np.zeros((32, 1))

    # Discriminator eğitimi
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator eğitimi
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, labels_real)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
6. Transformer - Basit Transformer Modeli
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Transformer

# Transformer model oluşturuluyor
inputs = Input(shape=(None, 512))  # Input shape: (batch_size, sequence_length, feature_size)
x = Transformer(num_heads=8, ff_dim=512)(inputs)
x = Dense(1, activation='sigmoid')(x)  # Sonuç sınıflandırma için

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
# varsayalım X_train ve y_train verisi mevcut
# model.fit(X_train, y_train, epochs=10, batch_size=32)
7. CNN - MNIST ile Görüntü Sınıflandırma
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# MNIST veri setini yükleyin
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# CNN modeli oluşturuluyor
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 sınıf için
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32)
8. LSTM - Zaman Serisi Tahmin
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Zaman serisi verisi
X_train = np.random.rand(100, 10, 1)  # 100 örnek, 10 zaman dilimi, 1 özellik
y_train = np.random.rand(100, 1)  # 100 hedef değeri

# LSTM modeli oluşturuluyor
model = Sequential([
    LSTM(50, input_shape=(10, 1)),
    Dense(1)  # Çıktı
])

model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, batch_size=32)
9. RNN - Zaman Serisi Tahmin
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Zaman serisi verisi
X_train = np.random.rand(100, 10, 1)  # 100 örnek, 10 zaman dilimi, 1 özellik
y_train = np.random.rand(100, 1)  # 100 hedef değeri

# RNN modeli oluşturuluyor
model = Sequential([
    SimpleRNN(50, input_shape=(10, 1)),
    Dense(1)  # Çıktı
])

model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, batch_size=32)
10. GAN - Basit GAN Eğitim Döngüsü
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generator
def build_generator():
    model = Sequential([
        Dense(128, input_dim=100, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(28*28, activation='sigmoid', reshape=(28, 28))
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Dense(1024, input_dim=28*28, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
generator = build_generator()
discriminator = build_discriminator()

# Discriminator'ı eğitilebilir yapmak
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Eğitim döngüsü
for epoch in range(10000):
    noise = np.random.normal(0, 1, (32, 100))
    generated_images = generator.predict(noise)
    real_images = np.random.rand(32, 28, 28)  # Gerçek görüntüleri burada kullanın

    labels_real = np.ones((32, 1))
    labels_fake = np.zeros((32, 1))

    # Discriminator eğitimi
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator eğitimi
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, labels_real)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")"""