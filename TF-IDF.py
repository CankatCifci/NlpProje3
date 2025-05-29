import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Verileri yükle
df_lem = pd.read_csv("lemmatized_data.csv")
df_stem = pd.read_csv("stemmed_data.csv")

# Giriş metni (Case12)
query_lem = df_lem[df_lem["case_id"] == "Case12"]["lemmatized_text"].values[0]
query_stem = df_stem[df_stem["case_id"] == "Case12"]["stemmed_text"].values[0]

# TF-IDF vektörleştirici
vectorizer_lem = TfidfVectorizer()
vectorizer_stem = TfidfVectorizer()

# Tüm metinleri vektörleştir
tfidf_matrix_lem = vectorizer_lem.fit_transform(df_lem["lemmatized_text"])
tfidf_matrix_stem = vectorizer_stem.fit_transform(df_stem["stemmed_text"])

# Sorgu metnini vektörleştir
query_vec_lem = vectorizer_lem.transform([query_lem])
query_vec_stem = vectorizer_stem.transform([query_stem])

# Cosine similarity hesapla
cos_sim_lem = cosine_similarity(query_vec_lem, tfidf_matrix_lem)[0]
cos_sim_stem = cosine_similarity(query_vec_stem, tfidf_matrix_stem)[0]

# En benzer 5 sonuç
top5_lem = cos_sim_lem.argsort()[::-1][1:6]  # 0. sıradaki kendisi olacak, o yüzden 1'den başlıyoruz
top5_stem = cos_sim_stem.argsort()[::-1][1:6]

print("🔹 TF-IDF Lemmatized - En Benzer 5:")
for i in top5_lem:
    print(f"{df_lem['case_id'][i]} → {df_lem['lemmatized_text'][i][:80]}...")

print("\n🔹 TF-IDF Stemmed - En Benzer 5:")
for i in top5_stem:
    print(f"{df_stem['case_id'][i]} → {df_stem['stemmed_text'][i][:80]}...")
