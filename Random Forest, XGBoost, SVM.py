import pandas as pd
import numpy as Np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

df = pd.read_csv('Telco-Customer-Churn.csv')


df = pd.get_dummies(df, drop_first=True)

x = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Random Forest Classifier Algorithm

rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)

rf.fit(X_train, y_train)

Y_pred = rf.predict(X_test)


print(classification_report(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))


# Using GridSearch Optimization technique
param_grid = {
    'n_estimators': [120, 210, 350],
    'max_depth': [11, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Using RandomSearch Optimization Technique
param_dist = {
    'n_estimators': randint(50, 250),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

