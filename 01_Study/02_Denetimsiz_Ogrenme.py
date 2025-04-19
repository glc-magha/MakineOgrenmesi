"""Denetimsiz Öğrenme (Unsupervised Learning)
Denetimsiz öğrenme, etiketlenmemiş veri kullanarak modelin verideki örüntüleri keşfetmesine olanak tanır.
Kümeleme (Clustering): Verileri benzer özelliklere sahip gruplara ayırmak. Örneğin, müşteri segmentasyonu.

Boyut İndirgeme (Dimensionality Reduction):
Özellik sayısını azaltarak veri setini daha yönetilebilir hale getirmek.
Örneğin, PCA (Principal Component Analysis).

Algoritmalar:
K-means
DBSCAN
Hiyerarşik kümeleme
PCA (Ana bileşenler analizi)
Autoencoder

1. K-means Kümeleme (K-means Clustering)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


2. DBSCAN Kümeleme (DBSCAN Clustering)
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

3. Hiyerarşik Kümeleme (Hierarchical Clustering)
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply Hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

4. PCA (Ana Bileşenler Analizi - Principal Component Analysis)
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the reduced dimensions
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
plt.title('PCA - Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


5. Autoencoder (Basit Yapay Sinir Ağı ile Boyut İndirgeme)
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load iris dataset
data = load_iris()
X = data.data

# Define Autoencoder model
model = Sequential()
model.add(Dense(2, input_dim=4, activation='relu'))  # 4 input features -> 2 encoded features
model.add(Dense(4, activation='sigmoid'))  # Reconstruct back to 4 features

# Compile and fit the model
model.compile(optimizer='adam', loss='mse')
model.fit(X, X, epochs=100)

# Encode data using the Autoencoder
encoded_X = model.predict(X)

# Plot the encoded data
plt.scatter(encoded_X[:, 0], encoded_X[:, 1])
plt.title('Autoencoder - Encoded Data')
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.show()


6. K-means Kümeleme (K-means Clustering) Elbow Method ile Optimal K Seçimi
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Elbow method to find optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()


7. DBSCAN Parametre Ayarı (DBSCAN Parameter Tuning)
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply DBSCAN with adjusted parameters
dbscan = DBSCAN(eps=0.7, min_samples=10)
labels = dbscan.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('DBSCAN Clustering with Adjusted Parameters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

8. Hiyerarşik Kümeleme (Hierarchical Clustering) Dendrogram Görselleştirmesi
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply linkage method to compute distances
Z = linkage(X, method='ward')

# Plot dendrogram
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

9. PCA ile Boyut İndirgeme ve Görselleştirme (2B)
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
plt.title('PCA - 2D Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

10. Autoencoder ile Boyut İndirgeme (2B Encoder)
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data

# Define Autoencoder model for 2D encoding
model = Sequential()
model.add(Dense(2, input_dim=4, activation='relu'))  # Encode to 2D
model.add(Dense(4, activation='sigmoid'))  # Decode back to 4D

# Compile and fit the model
model.compile(optimizer='adam', loss='mse')
model.fit(X, X, epochs=100)

# Encode the data
encoded_X = model.predict(X)

# Plot the encoded data
plt.scatter(encoded_X[:, 0], encoded_X[:, 1])
plt.title('Autoencoder - 2D Encoded Data')
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.show()

"""