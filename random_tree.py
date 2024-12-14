# Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and Preprocess the Dataset
data_train = pd.read_csv('adult/adult.data', header=None)
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data_train.columns = column_names

data_test = pd.read_csv('adult/adult.test', header=None, skiprows=1)
data_test.columns = column_names

data_test['income'] = data_test['income'].str.strip().replace({' >50K.': ' >50K', ' <=50K.': ' <=50K'})
data_train.replace(' ?', pd.NA, inplace=True)
data_test.replace(' ?', pd.NA, inplace=True)
data_train.dropna(inplace=True)
data_test.dropna(inplace=True)

data_train = pd.get_dummies(data_train, columns=['workclass', 'education', 'marital-status',
                                                 'occupation', 'relationship', 'race', 'sex', 'native-country'], drop_first=True)
data_test = pd.get_dummies(data_test, columns=['workclass', 'education', 'marital-status',
                                               'occupation', 'relationship', 'race', 'sex', 'native-country'], drop_first=True)

data_test = data_test.reindex(columns=data_train.columns, fill_value=0)
data_train['income'] = data_train['income'].apply(lambda x: 1 if x == ' >50K' else 0)
data_test['income'] = data_test['income'].apply(lambda x: 1 if x == ' >50K' else 0)

X = data_train.drop('income', axis=1)
y = data_train['income']

# Stratified splitting for balanced test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Verify class distributions in train and test sets
print("Class Distribution in Training Set:\n", y_train.value_counts())
print("Class Distribution in Test Set:\n", y_test.value_counts())

# Step 2: Train and Evaluate Random Forest Model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("Classification Report (Basic Model):")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (Basic Model):")
print(confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score (Basic Model):", roc_auc)

# Step 3: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid=param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

print("\nUpdated Classification Report (Optimized Model):")
print(classification_report(y_test, y_pred_best))
roc_auc_best = roc_auc_score(y_test, y_prob_best)
print("\nROC-AUC Score (Optimized Model):", roc_auc_best)

# Step 4: Feature Importance Visualization
feature_importances = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[-10:]

plt.figure(figsize=(8, 6))
plt.barh(X_train.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance in Random Forest")
plt.tight_layout()
plt.show()
