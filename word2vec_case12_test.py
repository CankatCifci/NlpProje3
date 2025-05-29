import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Veri ve model dosyası
df_lem = pd.read_csv("lemmatized_data.csv")
model = Word2Vec.load("word2vec_lemmatized_cbow_win2_dim300.model")

# Giriş metni (Case12)
query = df_lem[df_lem["case_id"] == "Case12"]["lemmatized_text"].values[0]

# Cümleyi tokenize et
def get_avg_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Giriş vektörü
query_vec = get_avg_vector(query, model)

# Veri setindeki tüm cümlelerin vektör ortalaması
sentence_vectors = []
for sent in df_lem["lemmatized_text"]:
    vec = get_avg_vector(sent, model)
    sentence_vectors.append(vec)

# Benzerlik hesapla
similarities = cosine_similarity([query_vec], sentence_vectors)[0]
top5_indices = similarities.argsort()[::-1][1:6]  # ilk sıradaki kendisi olduğu için [1:6]

# Sonuçları yazdır
print("🔹 Word2Vec (Lemmatized, CBOW, win=2, dim=300) - En Benzer 5:")
for i in top5_indices:
    print(f"{df_lem['case_id'][i]} → {df_lem['lemmatized_text'][i][:80]}...")
