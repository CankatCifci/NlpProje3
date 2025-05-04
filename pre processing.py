import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

# Gerekli NLTK dosyalarını indir
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# İngilizce stopwords ve stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    # 1. HTML temizliği
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Küçük harfe çevir
    text = text.lower()

    # 3. Özel karakterleri kaldır
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stop word çıkarımı
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Stemming
    stemmed = [stemmer.stem(word) for word in tokens]

    return " ".join(stemmed)

# CSV verisini yükle
df = pd.read_csv('legal_text_classification.csv')

# Sayısal ve boş değerleri çıkar
df_clean = df[~df['case_text'].apply(lambda x: isinstance(x, (float, int)))]
df_clean = df_clean[df_clean['case_text'].notna()]

# ÖN İŞLEME uygulama (uzun sürebilir)
print("Ön işleme uygulanıyor...")
df_clean['clean_text'] = df_clean['case_text'].apply(preprocess_text)

# Sentence-BERT ile embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
batch_size = 100
embeddings = []

print("Vektörleştirme başlıyor...")
for i in range(0, len(df_clean), batch_size):
    batch = df_clean['clean_text'][i:i + batch_size].tolist()
    batch_embeddings = model.encode(batch, show_progress_bar=True)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)

# Aykırı değer analizi (Isolation Forest)
clf = IsolationForest(contamination=0.1, random_state=42)
outlier_preds = clf.fit_predict(embeddings)

# Aykırı davalar
outlier_cases = df_clean[outlier_preds == -1]

# Sonuçları yazdır
print(f"Toplam {len(outlier_cases)} aykırı dava bulundu.")
print(outlier_cases[['case_id', 'case_title', 'case_text']].head())

# CSV’ye kaydet
outlier_cases.to_csv('outlier_cases.csv', index=False)
