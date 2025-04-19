"""Özellik Seçimi ve Mühendisliği (Feature Selection and Engineering)
Özellik mühendisliği, veriden anlamlı özelliklerin çıkarılmasını ve dönüştürülmesini içerir.
Özellik seçimi ise en etkili özelliklerin seçilmesini sağlar.

Özellik Seçimi Yöntemleri:
Korelasyon analizi
Ağaç tabanlı yöntemler (Random Forest, XGBoost)
Sıralı özellik seçimi
Özellik Mühendisliği Yöntemleri:
Kategorik özelliklerin dönüştürülmesi
Metin verisinin vektörleştirilmesi (TF-IDF, Word2Vec)

1. Korelasyon Analizi - Özellikler Arasındaki Korelasyonu Bulma
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleyin
# varsayalım df bir pandas DataFrame
df = pd.read_csv('data.csv')

# Korelasyon matrisini hesapla
correlation_matrix = df.corr()

# Korelasyon matrisini görselleştir
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
2. Ağaç Tabanlı Yöntemler ile Özellik Seçimi (Random Forest)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Random Forest modelini oluşturun
model = RandomForestClassifier(n_estimators=100)

# Modeli eğitin
model.fit(X, y)

# Özelliklerin önem derecelerini alın
feature_importances = model.feature_importances_

# Özellikleri ve önem derecelerini birleştirip sıralayın
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)
3. Ağaç Tabanlı Yöntemler ile Özellik Seçimi (XGBoost)
import xgboost as xgb
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# XGBoost modelini oluşturun
model = xgb.XGBClassifier()

# Modeli eğitin
model.fit(X, y)

# Özelliklerin önem derecelerini alın
importance = model.get_booster().get_score(importance_type='weight')

# Özelliklerin sıralanmış önem derecelerini yazdırın
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)
4. Sıralı Özellik Seçimi (Sequential Feature Selection)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Logistic Regression modeli oluşturun
model = LogisticRegression()

# Özellik seçimi (forward selection)
sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward')
sfs.fit(X, y)

# Seçilen özellikler
selected_features = X.columns[sfs.get_support()]
print("Selected Features:", selected_features)
5. Kategorik Özelliklerin Dönüştürülmesi - One Hot Encoding
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Kategorik değişkeni one hot encoding ile dönüştürme
df_encoded = pd.get_dummies(df, columns=['categorical_column'])

print(df_encoded.head())
6. Kategorik Özelliklerin Dönüştürülmesi - Label Encoding
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# LabelEncoder ile kategorik değişkeni sayısal hale getirme
le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['categorical_column'])

print(df.head())
7. Metin Verisinin Vektörleştirilmesi - TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Örnek metin verisi
corpus = ['this is a sentence', 'this is another sentence']

# TF-IDF vektörleştiricisini oluşturun
vectorizer = TfidfVectorizer()

# Veriyi dönüştürün
X_tfidf = vectorizer.fit_transform(corpus)

print(X_tfidf.toarray())
8. Metin Verisinin Vektörleştirilmesi - Word2Vec
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Örnek metin verisi
corpus = ['this is a sentence', 'this is another sentence']

# Tokenize etme
tokenized_corpus = [nltk.word_tokenize(text) for text in corpus]

# Word2Vec modelini eğitme
model = Word2Vec(tokenized_corpus, vector_size=50, window=3, min_count=1, workers=4)

# Kelimelerin vektörlerini yazdırma
print(model.wv['sentence'])
9. Özellik Seçimi - Korelasyon Tabanlı Özellik Seçimi
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Korelasyon matrisini hesapla
correlation_matrix = df.corr()

# Hedef değişken ile yüksek korelasyona sahip özellikleri seçme
target_corr = correlation_matrix['target']
selected_features = target_corr[target_corr > 0.5].index.tolist()

print("Selected Features based on correlation:", selected_features)
10. Özellik Seçimi - ANOVA F-Statistiği
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# ANOVA F-Statistiği ile en iyi 5 özelliği seçme
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Seçilen özelliklerin indekslerini yazdırma
selected_columns = X.columns[selector.get_support()]
print("Selected Features using ANOVA F-statistic:", selected_columns)"""