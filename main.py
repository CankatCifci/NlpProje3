import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
import numpy as np

# Veri setini yükle
df = pd.read_csv('legal_text_classification.csv')  # CSV dosyanızın yolu

# 1. Veri Temizleme: NaN ve sayısal değerleri temizle
# NaN ve sayısal değerleri temizle
df_clean = df[~df['case_text'].apply(lambda x: isinstance(x, (float, int)))]  # Sayısal verileri çıkar
df_clean = df_clean[df_clean['case_text'].notna()]  # NaN değerleri çıkar

# Temizlenmiş verinin ilk birkaç satırını kontrol et
print("Temizlenmiş veri:")
print(df_clean.head())

# 2. Modeli başlat
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Batch işlemesi: Veri çok büyükse, batch boyutunda vektörleştirme yapalım
batch_size = 100  # Batch boyutu
embeddings = []

for i in range(0, len(df_clean), batch_size):
    batch = df_clean['case_text'][i:i + batch_size].tolist()
    batch_embeddings = model.encode(batch, show_progress_bar=True)  # Batch halinde encode et
    embeddings.append(batch_embeddings)

# Vektörleri birleştir
embeddings = np.vstack(embeddings)

# 4. Aykırı değer tespiti: Isolation Forest kullanarak aykırı değerleri tespit et
clf = IsolationForest(contamination=0.1, random_state=42)
outlier_preds = clf.fit_predict(embeddings)

# Standart dışı maddeleri işaretle
outlier_cases = df_clean[outlier_preds == -1]

# 5. Sonuçları yazdır
print(f"Toplamda {len(outlier_cases)} standart dışı dava tespit edildi.")
print(outlier_cases[['case_id', 'case_title', 'case_text']])

# Eğer dilerseniz, sonuçları bir CSV dosyasına kaydedebilirsiniz
outlier_cases.to_csv('outlier_cases.csv', index=False)
