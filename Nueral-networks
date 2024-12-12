import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

# Load the data
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
train_data = pd.read_csv('adult/adult.data', header=None, names=column_names, na_values=' ?')

# Data preprocessing
train_data.dropna(inplace=True)
train_data['income'] = train_data['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Feature encoding
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Split features and target
target = 'income'
X = train_data.drop(columns=[target])
y = train_data[target]

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Neural network model creation
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perform k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kfold.split(X_scaled, y_balanced):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]

    # Create and train the model
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
              callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    scores.append(accuracy)

# Final evaluation
print(f"Mean cross-validation accuracy: {np.mean(scores):.2f}")
print(f"Standard deviation: {np.std(scores):.2f}")

# Final model training
final_model = create_model()
final_model.fit(X_scaled, y_balanced, epochs=50, batch_size=32, verbose=1)

# Predict and assess performance on the entire balanced dataset
y_pred = (final_model.predict(X_scaled) > 0.5).astype(int)
print(classification_report(y_balanced, y_pred))
conf_matrix = confusion_matrix(y_balanced, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
