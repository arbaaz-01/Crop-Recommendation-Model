# First
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Loading the dataset
df = pd.read_csv('/content/drive/MyDrive/Project/Crop_recommendation.csv')

# Feature Engineering
def create_features(df):
    df['rainfall_humidity'] = df['rainfall'] * df['humidity']
    df['pH_K'] = df['ph'] * df['K']
    df['N_P'] = df['N'] * df['P']
    df['temp_sq'] = df['temperature'] ** 2
    df['rainfall_log'] = np.log1p(df['rainfall'])
    df['climate_index'] = df['temperature'] * df['humidity'] / 100
    return df

df = create_features(df)

# Encode target variable
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Split features and target
X = df.drop('label', axis=1)
y = df['label']

# Data Augmentation
def augment_data(X, y, n_samples=1000):
    X_aug = X.sample(n_samples, replace=True, random_state=42)
    noise = np.random.normal(0, 0.1, X_aug.shape)
    X_aug = X_aug + noise
    X_aug = X_aug.clip(lower=0)
    y_aug = y.loc[X_aug.index]
    return pd.concat([X, X_aug]), pd.concat([y, y_aug])

X_aug, y_aug = augment_data(X, y, n_samples=1000)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data preparation completed. Shape of X_train:", X_train.shape)


# Second
from lightgbm import LGBMClassifier, early_stopping
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle

# Load data if not already in memory
# X_train = pd.read_csv('X_train.csv')
# y_train = pd.read_csv('y_train.csv')

# Hyperparameter Tuning with Optuna
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
        'min_split_gain': 0.1,
        'verbose': -1
    }

    model = LGBMClassifier(**params, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=10, verbose=False)]
        )
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
lgbm_params = study.best_params
print("Best LightGBM Params:", lgbm_params)
print("Best CV Accuracy:", study.best_value)

# Train final LightGBM model
lgbm_params['verbose'] = -1
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(X_train, y_train)

# Save the trained model
with open('lgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm, f)
print("LightGBM model trained and saved as 'lgbm_model.pkl'")


# third
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load data if not already in memory
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')

# Load trained models
models = {}
for name in ['lgbm', 'catboost', 'stacking']:
    with open(f'{name}_model.pkl', 'rb') as f:
        models[name.capitalize()] = pickle.load(f)

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")



# fourth
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap
import pickle
import numpy as np

# Load data and models if not already in memory
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')
with open('lgbm_model.pkl', 'rb') as f:
    lgbm = pickle.load(f)
with open('stacking_model.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Confusion Matrix for Stacking Model
y_pred_stack = stacking_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Stacking Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# fifth
import numpy as np
import shap
shap_values_2d = np.abs(shap_values).mean(axis=2)
shap.summary_plot(shap_values_2d, X_test, plot_type="bar", feature_names=X_test.columns)


# sixth
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load LightGBM model
with open('lgbm_model.pkl', 'rb') as f:
    lgbm = pickle.load(f)

# Load data for feature names
X_test = pd.read_csv('X_test.csv')

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': lgbm.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance - LightGBM Model')
plt.tight_layout()
plt.show()

# Print for table in paper
print(feature_importance)


# seventh
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd

# Load data and stacking model
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
with open('stacking_model.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Predict (fast since model is already trained)
y_pred = stacking_clf.predict(X_test)

# Compute and normalize confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Normalized Confusion Matrix - Stacking Model')
plt.xlabel('Predicted Crop')
plt.ylabel('True Crop')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()