import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
data = pd.read_csv('data.csv')  # Dosyayı yükle
data = data.drop(columns=['id'])  # 'id' sütununu kaldır
data.columns = data.columns.str.strip()  # Sütun adlarındaki gereksiz boşlukları temizle

print("------------------------------Sayısal Değerler Hakkında Bilgiler---------------------------------")
print(data.describe())  # Veri seti hakkında genel bilgi

# Eksik değer analizi
print("------------------------------Eksik Değer Analizi---------------------------------")
missing_values = data.isnull().sum()  # Her sütundaki eksik değer sayısı
if missing_values.sum() == 0:
    print("Eksik değer bulunmamaktadır.")
else:
    print("Eksik değerlerin sayısı: ", missing_values.sum())
    print("Eksik değerlerin sütunlara göre dağılımı: ")
    print(missing_values)

# Aykırı Değer Analizi ve İşleme (IQR Yöntemi)
print("------------------------------Aykırı Değerleri IQR ile Çıkarma---------------------------------")

# Sadece sayısal sütunlar üzerinde işlem yapalım
numeric_columns = data.select_dtypes(include=np.number).columns

# Aykırı değer içeren satırların toplanacağı bir set
etkilenen_satirlar = set()

for col in numeric_columns:
    # Çeyrekler arası aralık (IQR) hesaplama
    Q1 = data[col].quantile(0.25)  # 1. Çeyrek
    Q3 = data[col].quantile(0.75)  # 3. Çeyrek
    IQR = Q3 - Q1  # IQR hesaplama

    # Alt ve üst sınırları belirleme
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değer olan satırların indekslerini tespit et
    aykiri_satirlar = data.index[(data[col] < lower_bound) | (data[col] > upper_bound)].tolist()
    etkilenen_satirlar.update(aykiri_satirlar)

    # Aykırı değerlerin yerine sütun ortalamasını koyma
    col_mean = data[col].mean()
    data[col] = np.where(
        (data[col] < lower_bound) | (data[col] > upper_bound), 
        col_mean,  # Aykırı değerlerin yerine sütun ortalamasını koy
        data[col]
    )

# Toplamda etkilenen satır sayısını belirt
print(f"Etkilenen toplam aykırı değer satırı: {len(etkilenen_satirlar)}")