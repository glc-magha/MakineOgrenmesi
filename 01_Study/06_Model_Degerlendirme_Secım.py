""" Model Değerlendirme ve Seçimi (Model Evaluation and Selection)
Modelin başarısını ölçmek ve en iyi modelin seçilmesi için çeşitli metrikler ve çapraz doğrulama yöntemleri kullanılır.

Metrikler:
Doğruluk (Accuracy)
Hata Kareler Ortalaması (RMSE)
F1 Skoru
Precision, Recall
AUC-ROC Eğrisi
Yöntemler:
Çapraz doğrulama (Cross-validation)
Hiperparametre optimizasyonu (Grid Search, Random Search)

1. Doğruluk (Accuracy) Değerlendirmesi
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# Doğruluğu hesaplayın
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
2. Hata Kareler Ortalaması (RMSE) Değerlendirmesi
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = LinearRegression()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# RMSE hesaplayın
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
3. F1 Skoru
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# F1 skorunu hesaplayın
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)
4. Precision ve Recall
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# Precision ve Recall hesaplayın
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
5. AUC-ROC Eğrisi
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapın
y_pred_proba = model.predict_proba(X_test)[:, 1]

# AUC-ROC hesaplayın
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC:", auc)

# ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
6. Çapraz Doğrulama (Cross-validation)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Modeli oluşturun
model = RandomForestClassifier()

# Çapraz doğrulama ile modelin doğruluğunu ölçün
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("Cross-validation Accuracy:", scores)
print("Mean Accuracy:", scores.mean())
7. Hiperparametre Optimizasyonu - Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Modeli oluşturun
model = RandomForestClassifier()

# Hiperparametreler için bir grid belirleyin
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}

# Grid search kullanarak en iyi hiperparametreleri bulun
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# En iyi parametreler
print("Best Parameters:", grid_search.best_params_)
8. Hiperparametre Optimizasyonu - Random Search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Modeli oluşturun
model = RandomForestClassifier()

# Hiperparametreler için bir distribütör belirleyin
param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}

# Random search kullanarak en iyi hiperparametreleri bulun
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, n_iter=10, scoring='accuracy')
random_search.fit(X, y)

# En iyi parametreler
print("Best Parameters from Random Search:", random_search.best_params_)
9. Karmaşıklık Matrisi (Confusion Matrix)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# Karmaşıklık matrisini hesaplayın
cm = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştirin
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
10. Modelin Performansını Özetleme (Classification Report)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükleyin
df = pd.read_csv('data.csv')

# Özellikler ve hedef değişkeni ayırın
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli oluşturun
model = RandomForestClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Tahminleri yapın
y_pred = model.predict(X_test)

# Modelin performansını özetleme
report = classification_report(y_test, y_pred)
print(report)"""