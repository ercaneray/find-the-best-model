import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.base import clone

# Veri setini yükleme
data = pd.read_csv('data.csv')  # Dosyayı yükle
data = data.drop(columns=['id'])  # 'id' sütununu kaldır
data.columns = data.columns.str.strip()  # Sütun adlarındaki gereksiz boşlukları temizle

print("------------------------------Sayısal Değerler Hakkında Bilgiler---------------------------------")
print(data.describe())  # Veri seti hakkında genel bilgi

# Eksik değer analizi
print("------------------------------Eksik Değer Analizi----------------------------------------------")
missing_values = data.isnull().sum()  # Her sütundaki eksik değer sayısı
if missing_values.sum() == 0:
    print("Eksik değer bulunmamaktadır.")
else:
    print("Eksik değerlerin sayısı: ", missing_values.sum())
    print("Eksik değerlerin sütunlara göre dağılımı: ")
    print(missing_values)

# Aykırı Değer Analizi ve İşleme (IQR Yöntemi)
print("------------------------------Aykırı Değerleri IQR ile Çıkarma---------------------------------")
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

print(f"Etkilenen toplam aykırı değer satırı: {len(etkilenen_satirlar)}")

# Özellikler (X) ve hedef değişken (y) ayrımı
X = data.drop(columns=[data.columns[-1]])
y = data[data.columns[-1]]

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataseti ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Binary sınıflandırma için eşik değeri belirleme
threshold = y.median()
y_train_binary = (y_train >= threshold).astype(int)
y_test_binary = (y_test >= threshold).astype(int)


# Model ve isim tanımlamaları
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42)
}

# Sonuçları saklamak için bir liste
results = []

# Modellerin döngü ile eğitimi ve değerlendirilmesi
print("------------------------------Makine Öğrenmesi Modellerini Eğitme---------------------------------")
for model_name, model in models.items():
    print(f"{model_name} modeli eğitiliyor...")

    if model_name == 'Linear Regression':
        model_clone = clone(model)
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        y_pred_binary = (y_pred >= threshold).astype(int)

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary)
        recall = recall_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_test_binary, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred)
        roc_auc = auc(fpr, tpr)

    else:
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

# Sonuçları DataFrame olarak yazdır
results_df = pd.DataFrame(results)
print(results_df)

# ROC Eğrileri çizim sonlandırma
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

print()