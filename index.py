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
import joblib  # Makine öğrenmesi modelleri için
from tensorflow.keras.models import save_model, load_model  # Derin öğrenme modelleri için
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Kodu fonksiyonlara bölmek için önerilen yapı
def veri_on_isleme(data):
    # Veri ön işleme adımları...
    pass

def model_egitimi(X_train, y_train):
    # Model eğitim adımları...
    pass

def performans_degerlendirme(y_true, y_pred):
    # Performans metrikleri hesaplama...
    pass

# Veri setini yükleme
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Veri dosyası bulunamadı!")
    sys.exit(1)
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
mlModels = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
}

# Sonuçları saklamak için bir liste
results = []

# Modellerin döngü ile eğitimi ve değerlendirilmesi
print("------------------------------Makine Öğrenmesi Modellerini Eğitme---------------------------------")
for model_name, model in mlModels.items():
    print(f"{model_name} modeli eğitiliyor...")
    
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
plt.title('Makine Öğrenmesi Modellerinin ROC Eğrileri')
plt.legend(loc='lower right')
plt.show()

#* Derin Öğrenme Modelleri
print("------------------------------Derin Öğrenme Modellerini Eğitme---------------------------------")
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
        Dense(1, activation='sigmoid')  # Binary sınıflandırma için sigmoid aktivasyonu
    ]),
    'MLP': Sequential([
        Input(shape=(X_train_scaled.shape[1],)),  # Giriş boyutunu belirtiyoruz
        Dense(256, activation='relu'),  # İlk gizli katman
        Dropout(0.3),
        Dense(128, activation='relu'),  # İkinci gizli katman
        Dropout(0.3),
        Dense(64, activation='relu'),  # Üçüncü gizli katman
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Çıkış katmanı
    ])
}

# Derin Öğrenme Modelleri Eğitim Süreci
results_dl = []
for model_name, model in dlModels.items():
    print(f"{model_name} modeli eğitiliyor...")

    if model_name == 'CNN':
        # CNN için giriş verilerinin yeniden şekillendirilmesi
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

        # Test Verisinde Performansı Değerlendirme
        test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_binary, verbose=0)
        y_pred_prob = model.predict(X_test_cnn).flatten()  # Olasılık tahminleri
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Binary tahminler
    else:  # FCNN ve MLP için
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

        # Test Verisinde Performansı Değerlendirme
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_binary, verbose=0)
        y_pred_prob = model.predict(X_test_scaled).flatten()  # Olasılık tahminleri
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Binary tahminler

    # Performans Metriklerini Hesaplama
    test_precision = precision_score(y_test_binary, y_pred_binary)
    test_recall = recall_score(y_test_binary, y_pred_binary)
    test_f1 = f1_score(y_test_binary, y_pred_binary)
    test_auc = roc_auc_score(y_test_binary, y_pred_prob)

    print(f"{model_name} modeli test doğruluğu: {test_accuracy:.4f}")
    print(f"{model_name} modeli kesinlik (Precision): {test_precision:.4f}")
    print(f"{model_name} modeli duyarlılık (Recall): {test_recall:.4f}")
    print(f"{model_name} modeli F1-Score: {test_f1:.4f}")
    print(f"{model_name} modeli AUC: {test_auc:.4f}")
    
    results_dl.append({
        'Model': model_name,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1-Score': test_f1,
        'AUC': test_auc
    })

# Derin öğrenme modellerinin sonuçlarını yazdır
results_dl_df = pd.DataFrame(results_dl)
print(results_dl_df)
# ROC Eğrileri çizim sonlandırma - Derin Öğrenme Modelleri
print("------------------------------Derin Öğrenme Modellerinin ROC Eğrileri---------------------------------")
plt.figure(figsize=(10, 8))

# Makine Öğrenmesi ROC eğrilerini çizme
for result in results:
    model_name = result['Model']
    roc_auc = result['AUC']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Derin Öğrenme ROC eğrilerini çizme
for model_name, model in dlModels.items():
    if model_name == 'CNN':
        # CNN için giriş verilerinin yeniden şekillendirilmesi
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        y_pred_prob = model.predict(X_test_cnn).flatten()  # Olasılık tahminleri
    else:
        y_pred_prob = model.predict(X_test_scaled).flatten()  # Olasılık tahminleri
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# ROC Eğrisi Grafiği
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Makine Öğrenmesi ve Derin Öğrenme Modellerinin ROC Eğrileri')
plt.legend(loc='lower right')
plt.show()
# Makine Öğrenmesi ve Derin Öğrenme Sonuçlarını Birleştirme
all_results = pd.concat([
    pd.DataFrame(results),  # Makine öğrenmesi sonuçları
    pd.DataFrame(results_dl)  # Derin öğrenme sonuçları
])

# En iyi modeli seçmek için sıralama
sorted_results = all_results.sort_values(by=['Accuracy', 'F1-Score', 'AUC'], ascending=False)

# Tüm sonuçları ve en iyi modeli yazdırma
print("\n--- Model Performans Karşılaştırma Tablosu ---\n")
print(sorted_results)


# En iyi modeli seçme
sorted_results = all_results.sort_values(by=['Accuracy', 'F1-Score'], ascending=False)
best_model_name = sorted_results.iloc[0]['Model']
print(f"En iyi model: {best_model_name}")

# En iyi modeli kaydetme
if best_model_name in mlModels.keys():
    # Makine öğrenmesi modelini seç ve kaydet
    best_model = mlModels[best_model_name]
    best_model.fit(X_train_scaled, y_train_binary)
    joblib.dump(best_model, f'{best_model_name}.pkl')
    print(f"Makine öğrenmesi modeli kaydedildi: {best_model_name}.pkl")
else:
    # Derin öğrenme modelini seç ve kaydet
    best_model = dlModels[best_model_name]
    if best_model_name == 'CNN':
        # CNN modeli için veriyi yeniden şekillendir
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        best_model.fit(X_train_cnn, y_train_binary, epochs=100, verbose=0)
    else:
        best_model.fit(X_train_scaled, y_train_binary, epochs=100, verbose=0)
    save_model(best_model, f'{best_model_name}.h5')
    print(f"Derin öğrenme modeli kaydedildi: {best_model_name}.h5")

# Tahmin yapacak bir fonksiyon
def predict_new_sample(sample):
    """
    Yeni bir örnek için tahmin yapar.
    :param sample: Yeni örnek (pandas Series veya numpy array, şekli: (özellik_sayısı,))
    :return: Tahmin sonucu (etiket ve olasılık)
    """
    if isinstance(sample, pd.Series):
        sample = sample.values.reshape(1, -1)  # Pandas Series'i numpy array'e dönüştür ve yeniden şekillendir
    else:
        sample = np.array(sample).reshape(1, -1)  # Numpy array'e dönüştür ve yeniden şekillendir
    
    # Özellik isimlerini içerecek şekilde DataFrame'e dönüştür
    sample_df = pd.DataFrame(sample, columns=X.columns)
    sample_scaled = scaler.transform(sample_df)  # Girdiyi ölçeklendir

    if best_model_name in mlModels.keys():
        # Makine öğrenmesi modeli kullanarak tahmin
        loaded_model = joblib.load(f'{best_model_name}.pkl')
        prediction = loaded_model.predict(sample_scaled)
        probability = loaded_model.predict_proba(sample_scaled)[:, 1] if hasattr(loaded_model, 'predict_proba') else None
    else:
        # Derin öğrenme modeli kullanarak tahmin
        loaded_model = load_model(f'{best_model_name}.h5')
        if best_model_name == 'CNN':
            # CNN için girişin yeniden şekillendirilmesi
            sample_scaled = sample_scaled.reshape(1, sample_scaled.shape[1], 1)
        prediction = (loaded_model.predict(sample_scaled) >= 0.5).astype(int)
        probability = loaded_model.predict(sample_scaled).flatten()
    
    return {
        'prediction': int(prediction[0]), 
        'probability': float(probability[0] if probability is not None else 0)
    }

# Örnek kullanım
new_sample = X_test.iloc[0]  # Test setinden bir örnek
result = predict_new_sample(new_sample)
print(f"Tahmin Sonucu: {result}")





