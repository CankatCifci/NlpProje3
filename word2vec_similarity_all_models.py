import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Model klasöründe yer alan tüm .model dosyalarını bul
model_files = [f for f in os.listdir() if f.endswith(".model")]

# Verileri yükle
df_lem = pd.read_csv("lemmatized_data.csv")
df_stem = pd.read_csv("stemmed_data.csv")

# Giriş metni (Case12)
query_lem = df_lem[df_lem["case_id"] == "Case12"]["lemmatized_text"].values[0]
query_stem = df_stem[df_stem["case_id"] == "Case12"]["stemmed_text"].values[0]

# Ortalama vektör fonksiyonu
def get_avg_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Her bir model için işlem yap
for model_file in sorted(model_files):
    print(f"\n🔹 {model_file} - En Benzer 5:")

    # Modeli yükle
    model = Word2Vec.load(model_file)

    # Giriş metni belirle
    if "lemmatized" in model_file:
        df = df_lem
        query = query_lem
    else:
        df = df_stem
        query = query_stem

    # Giriş cümlesinin ortalama vektörü
    query_vec = get_avg_vector(query, model)

    # Tüm veri setindeki cümlelerin ortalama vektörleri
    sentence_vectors = [get_avg_vector(sent, model) for sent in df[df.columns[1]]]

    # Cosine similarity hesapla
    similarities = cosine_similarity([query_vec], sentence_vectors)[0]
    top5_indices = similarities.argsort()[::-1][1:6]  # kendisini dışarıda tut

    # Sonuçları yazdır
    for i in top5_indices:
        print(f"{df['case_id'][i]} → {df[df.columns[1]][i][:80]}...")
