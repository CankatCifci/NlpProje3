import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter

# Gerekli NLTK dosyalarını indir
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# İngilizce stopwords, stemmer ve lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


# Metin ön işleme fonksiyonu
def preprocess_text(text, apply_stemming=True, apply_lemmatization=True):
    # 1. HTML temizliği
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Küçük harfe çevir
    text = text.lower()

    # 3. Özel karakterleri kaldır
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 4. Tokenization (Kelimeye ayırma)
    tokens = word_tokenize(text)

    # 5. Stop word çıkarımı
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Stemming (Kök bulma)
    if apply_stemming:
        tokens = [stemmer.stem(word) for word in tokens]

    # 7. Lemmatization (Kelimeleri köklerine indirgeme)
    if apply_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# CSV verisini yükle
df = pd.read_csv('legal_text_classification.csv')

# Sayısal ve boş değerleri çıkar
df_clean = df[~df['case_text'].apply(lambda x: isinstance(x, (float, int)))]

# ÖN İŞLEME uygulama (uzun sürebilir)
print("Ön işleme uygulanıyor...")

# Stemming sonucu
df_clean['stemmed_text'] = df_clean['case_text'].apply(preprocess_text, apply_stemming=True, apply_lemmatization=False)
# Lemmatization sonucu
df_clean['lemmatized_text'] = df_clean['case_text'].apply(preprocess_text, apply_stemming=False,
                                                          apply_lemmatization=True)

# Stemming sonucu CSV dosyasını kaydet
df_clean[['case_id', 'stemmed_text']].to_csv('stemmed_data.csv', index=False)

# Lemmatization sonucu CSV dosyasını kaydet
df_clean[['case_id', 'lemmatized_text']].to_csv('lemmatized_data.csv', index=False)


# Zipf grafiği çizme fonksiyonu
def plot_zipf(text_column, title):
    # Kelimeleri ayır
    all_words = ' '.join(text_column).split()

    # Kelimeleri say
    word_counts = Counter(all_words)

    # En yaygın kelimeleri sıralayıp görselleştir
    word_freq = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    freq = [x[1] for x in word_freq]

    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(freq) + 1), freq)
    plt.title(f"Zipf Yasası - {title}")
    plt.xlabel('Sıra')
    plt.ylabel('Frekans')
    plt.show()


# Zipf grafiği çiz
plot_zipf(df_clean['stemmed_text'], "Stemming Sonrası Metin")
plot_zipf(df_clean['lemmatized_text'], "Lemmatization Sonrası Metin")

# Yeni veri boyutlarını ve çıkarılan veriyi yazdır
original_size = len(df_clean['case_text'])
stemmed_size = len(df_clean['stemmed_text'])
lemmatized_size = len(df_clean['lemmatized_text'])

print(f"Orijinal veri boyutu: {original_size} cümle")
print(f"Stemming sonucu veri boyutu: {stemmed_size} cümle")
print(f"Lemmatization sonucu veri boyutu: {lemmatized_size} cümle")
print(f"Stemming sırasında çıkarılan veri: {original_size - stemmed_size} cümle")
print(f"Lemmatization sırasında çıkarılan veri: {original_size - lemmatized_size} cümle")
