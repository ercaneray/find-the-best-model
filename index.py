import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.base import clone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib  # For machine learning models
from tensorflow.keras.models import save_model, load_model  # For deep learning models
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Suggested structure for dividing code into functions
def veri_on_isleme(data):
    # Data preprocessing steps...
    pass

def model_egitimi(X_train, y_train):
    # Model training steps...
    pass

def performans_degerlendirme(y_true, y_pred):
    # Calculate performance metrics...
    pass

# Loading the dataset
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Data file not found!")
    sys.exit(1)
data = data.drop(columns=['id'])  # Remove 'id' column
data.columns = data.columns.str.strip()  # Clean unnecessary spaces in column names

print("------------------------------Information About Numerical Values---------------------------------")
print(data.describe())  # General information about the dataset

# Missing value analysis
print("------------------------------Missing Value Analysis----------------------------------------------")
missing_values = data.isnull().sum()  # Number of missing values in each column
if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print("Number of missing values: ", missing_values.sum())
    print("Distribution of missing values by columns: ")
    print(missing_values)

# Outlier Analysis and Processing (IQR Method)
print("------------------------------Removing Outliers with IQR---------------------------------")
numeric_columns = data.select_dtypes(include=np.number).columns
etkilenen_satirlar = set()

for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    aykiri_satirlar = data.index[(data[col] < lower_bound) | (data[col] > upper_bound)].tolist()
    etkilenen_satirlar.update(aykiri_satirlar)

    col_mean = data[col].mean()
    data[col] = np.where(
        (data[col] < lower_bound) | (data[col] > upper_bound), 
        col_mean,
        data[col]
    )

print(f"Total number of rows affected by outliers: {len(etkilenen_satirlar)}")

# Feature (X) and target variable (y) separation
X = data.drop(columns=[data.columns[-1]])
y = data[data.columns[-1]]

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determining threshold value for binary classification
threshold = y.median()
y_train_binary = (y_train >= threshold).astype(int)
y_test_binary = (y_test >= threshold).astype(int)

# Model definitions and names
mlModels = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
}

# List to store results
results = []

# Modellers training and evaluation loop
print("------------------------------Training Machine Learning Models---------------------------------")
for model_name, model in mlModels.items():
    print(f"{model_name} model training...")
    
    model_clone = clone(model)
    model_clone.fit(X_train_scaled, y_train_binary)
    y_pred = model_clone.predict(X_test_scaled)
    y_pred_prob = model_clone.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred)
    recall = recall_score(y_test_binary, y_pred)
    f1 = f1_score(y_test_binary, y_pred)
    kappa = cohen_kappa_score(y_test_binary, y_pred)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        "Cohen's Kappa": kappa,
        'AUC': roc_auc
    })

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Results DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# ROC Curves Plot
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Machine Learning Models')
plt.legend(loc='lower right')
plt.show()

# Deep Learning Models Training
print("------------------------------Training Deep Learning Models---------------------------------")
dlModels = {
    'CNN': Sequential([
        Input(shape=(X_train_scaled.shape[1], 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ]),
    'FCNN': Sequential([
         Input(shape=(X_train_scaled.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ]),
    'MLP': Sequential([
        Input(shape=(X_train_scaled.shape[1],)),  # Specify input shape
        Dense(256, activation='relu'),  # First hidden layer
        Dropout(0.3),
        Dense(128, activation='relu'),  # Second hidden layer
        Dropout(0.3),
        Dense(64, activation='relu'),  # Third hidden layer
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output layer
    ])
}

# Deep Learning Models Training Process
results_dl = []
for model_name, model in dlModels.items():
    print(f"{model_name} model training...")

    if model_name == 'CNN':
        # Reshaping input data for CNN
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train_cnn,
            y_train_binary,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Performance Evaluation on Test Data
        test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_binary, verbose=0)
        y_pred_prob = model.predict(X_test_cnn).flatten()  # Probability predictions
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Binary predictions
    else:  # For FCNN and MLP
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train_scaled,
            y_train_binary,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Performance Evaluation on Test Data
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_binary, verbose=0)
        y_pred_prob = model.predict(X_test_scaled).flatten()  # Probability predictions
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Binary predictions

    # Calculate Performance Metrics
    test_precision = precision_score(y_test_binary, y_pred_binary)
    test_recall = recall_score(y_test_binary, y_pred_binary)
    test_f1 = f1_score(y_test_binary, y_pred_binary)
    test_auc = roc_auc_score(y_test_binary, y_pred_prob)

    print(f"{model_name} model test accuracy: {test_accuracy:.4f}")
    print(f"{model_name} model precision: {test_precision:.4f}")
    print(f"{model_name} model recall: {test_recall:.4f}")
    print(f"{model_name} model F1-Score: {test_f1:.4f}")
    print(f"{model_name} model AUC: {test_auc:.4f}")
    
    results_dl.append({
        'Model': model_name,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1-Score': test_f1,
        'AUC': test_auc
    })

# Deep learning model results
results_dl_df = pd.DataFrame(results_dl)
print(results_dl_df)
# ROC Curves Plot
print("------------------------------ROC Curves of Deep Learning Models---------------------------------")
plt.figure(figsize=(10, 8))

# Machine Learning ROC Curves Plot
for result in results:
    model_name = result['Model']
    roc_auc = result['AUC']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Deep Learning ROC Curves Plot
for model_name, model in dlModels.items():
    if model_name == 'CNN':
        # Reshape data for CNN model
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        y_pred_prob = model.predict(X_test_cnn).flatten()  # Probability predictions
    else:
        y_pred_prob = model.predict(X_test_scaled).flatten()  # Probability predictions
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# ROC Eğrisi Grafiği
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Machine Learning and Deep Learning Models')
plt.legend(loc='lower right')
plt.show()
# Combining Machine Learning and Deep Learning Results
all_results = pd.concat([
    pd.DataFrame(results),  # Machine learning results
    pd.DataFrame(results_dl)  # Deep learning results
])

# Sort to select best model
sorted_results = all_results.sort_values(by=['Accuracy', 'F1-Score', 'AUC'], ascending=False)

# Print all results and best model
print("\n--- Model Performance Comparison Table ---\n")
print(sorted_results)

# Select best model
sorted_results = all_results.sort_values(by=['Accuracy', 'F1-Score'], ascending=False)
best_model_name = sorted_results.iloc[0]['Model']
print(f"Best model: {best_model_name}")

# Save best model
if best_model_name in mlModels.keys():
    # Select and save machine learning model
    best_model = mlModels[best_model_name]
    best_model.fit(X_train_scaled, y_train_binary)
    joblib.dump(best_model, f'{best_model_name}.pkl')
    print(f"Machine learning model saved: {best_model_name}.pkl")
else:
    # Select and save deep learning model
    best_model = dlModels[best_model_name]
    if best_model_name == 'CNN':
        # Reshape data for CNN model
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        best_model.fit(X_train_cnn, y_train_binary, epochs=100, verbose=0)
    else:
        best_model.fit(X_train_scaled, y_train_binary, epochs=100, verbose=0)
    save_model(best_model, f'{best_model_name}.h5')
    print(f"Deep learning model saved: {best_model_name}.h5")

# Tahmin yapacak bir fonksiyon
def predict_new_sample(sample):
    """
    Makes predictions for a new sample.
    :param sample: New sample (pandas Series or numpy array, shape: (n_features,))
    :return: Prediction result (label and probability)
    """
    if isinstance(sample, pd.Series):
        sample = sample.values.reshape(1, -1)  # Convert pandas Series to numpy array and reshape
    else:
        sample = np.array(sample).reshape(1, -1) # Convert sample to numpy array and reshape
    
    # Convert sample to DataFrame with feature names
    sample_df = pd.DataFrame(sample, columns=X.columns)
    sample_scaled = scaler.transform(sample_df)  # Scale input

    if best_model_name in mlModels.keys():
        # Makine öğrenmesi modeli kullanarak tahmin
        loaded_model = joblib.load(f'{best_model_name}.pkl')
        prediction = loaded_model.predict(sample_scaled)
        probability = loaded_model.predict_proba(sample_scaled)[:, 1] if hasattr(loaded_model, 'predict_proba') else None
    else:
            # Derin öğrenme modeli kullanarak tahmin
        loaded_model = load_model(f'{best_model_name}.h5')
        if best_model_name == 'CNN':
            # Reshape input for CNN
            sample_scaled = sample_scaled.reshape(1, sample_scaled.shape[1], 1)
        prediction = (loaded_model.predict(sample_scaled) >= 0.5).astype(int)
        probability = loaded_model.predict(sample_scaled).flatten()
    
    return {
        'prediction': int(prediction[0]), 
        'probability': float(probability[0] if probability is not None else 0)
    }

# Example usage
new_sample = X_test.iloc[0]  # One sample from test set
result = predict_new_sample(new_sample)
print(f"Prediction Result: {result}")





