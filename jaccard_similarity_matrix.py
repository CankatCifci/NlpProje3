import pandas as pd

# Her modelin en benzer 5 sonucu
top5_dict = {
    "TF-IDF Lemmatized": ["Case24963", "Case24962", "Case5965", "Case20245", "Case23331"],
    "TF-IDF Stemmed": ["Case24963", "Case24962", "Case5965", "Case20245", "Case17115"],
    "Word2Vec Lemma CBOW w2 d100": ["Case24963", "Case24962", "Case5965", "Case20245", "Case14615"],
    "Word2Vec Lemma SG w4 d300": ["Case23331", "Case24963", "Case24962", "Case20245", "Case9805"],
    "Word2Vec Lemma SG w2 d100": ["Case24962", "Case24963", "Case5965", "Case20245", "Case23331"],
}

# Boş Jaccard matrisi oluştur
model_names = list(top5_dict.keys())
jaccard_matrix = pd.DataFrame(index=model_names, columns=model_names)

# Jaccard skorlarını hesapla
for model_a in model_names:
    set_a = set(top5_dict[model_a])
    for model_b in model_names:
        set_b = set(top5_dict[model_b])
        if model_a == model_b:
            score = 1.0
        else:
            score = len(set_a & set_b) / len(set_a | set_b)
        jaccard_matrix.loc[model_a, model_b] = round(score, 2)

# Matris yazdır
print(jaccard_matrix)
