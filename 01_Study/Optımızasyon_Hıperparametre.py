"""12. Optimizasyon ve Hiperparametre Ayarı (Optimization and Hyperparameter Tuning)
Makine öğrenmesi modelinin başarısını artırmak için hiperparametrelerin doğru ayarlanması önemlidir.

Yöntemler:
Grid Search
Random Search
Bayesian Optimization

1. Grid Search (Random Forest için)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
model = RandomForestClassifier()
grid = GridSearchCV(model, params, cv=3)
grid.fit([[1,2],[3,4],[5,6],[7,8]], [0,1,0,1])

print("En iyi parametreler:", grid.best_params_)
2. Randomized Search (Logistic Regression için)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

params = {'C': uniform(0.1, 10)}
model = LogisticRegression()
search = RandomizedSearchCV(model, params, n_iter=10, cv=3, random_state=42)
search.fit([[1,2],[3,4],[5,6],[7,8]], [0,1,0,1])

print("En iyi parametreler:", search.best_params_)
3. Grid Search ile SVC (Support Vector Classifier)
from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit([[1,2],[3,4],[5,6],[7,8],[9,10]], [0,1,0,1,0])

print("Best Parameters:", grid_search.best_params_)
4. Randomized Search ile XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rand_search = RandomizedSearchCV(model, params, n_iter=5, cv=3)
rand_search.fit([[1,2],[2,3],[3,4],[4,5]], [0,1,0,1])

print("Best Params:", rand_search.best_params_)
5. Optuna ile Hiperparametre Ayarı (LightGBM)
import optuna
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = lgb.LGBMClassifier(**params)
    return cross_val_score(model, X, y, cv=3).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("En iyi parametreler:", study.best_params)
6. Bayesian Optimization ile Random Forest
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

def rf_cv(n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth))
    return cross_val_score(model, X, y, cv=3).mean()

optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds={'n_estimators': (10, 200), 'max_depth': (2, 10)},
    random_state=42
)

optimizer.maximize(init_points=2, n_iter=5)
print("Best Params:", optimizer.max)
7. Grid Search ile Ridge Regression

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.1, 1.0, 10.0]}
model = Ridge()
grid = GridSearchCV(model, params, cv=3)
grid.fit([[1,2],[3,4],[5,6]], [2,3,4])

print("Best alpha:", grid.best_params_)
8. Randomized Search ile DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

params = {'max_depth': [3, 5, 7, 9], 'min_samples_split': [2, 5, 10]}
model = DecisionTreeClassifier()
search = RandomizedSearchCV(model, params, n_iter=5, cv=3)
search.fit([[1,2],[2,3],[3,4],[4,5]], [0,1,0,1])

print("Best Params:", search.best_params_)
9. Optuna ile Logistic Regression
def objective(trial):
    C = trial.suggest_float("C", 0.01, 10.0)
    clf = LogisticRegression(C=C, solver='liblinear')
    return cross_val_score(clf, X, y, cv=3).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best C value:", study.best_params)
X ve y veri seti için sklearn.datasets.load_iris() gibi bir örnek kullanabilirsin.

10. Grid Search + Pipeline ile TF-IDF + Naive Bayes
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

text = ["Kedi mırlıyor", "Köpek havlıyor", "Kuş ötüyor"]
labels = [0, 1, 2]

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

params = {
    'tfidf__max_df': [0.9, 1.0],
    'clf__alpha': [0.1, 1.0]
}

grid = GridSearchCV(pipe, params, cv=2)
grid.fit(text, labels)

print("Best Params:", grid.best_params_)"""