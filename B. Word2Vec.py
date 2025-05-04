import pandas as pd
from gensim.models import Word2Vec

# Word2Vec vektörleştirme fonksiyonu
def create_word2vec_model(text_data, model_name, model_params):
    # Tokenize the text (split by spaces or punctuation)
    tokenized_text = [sentence.split() for sentence in text_data]

    # Train the Word2Vec model
    model = Word2Vec(sentences=tokenized_text, vector_size=model_params['vector_size'],
                     window=model_params['window'], sg=1 if model_params['model_type'] == 'skipgram' else 0,
                     min_count=1)

    # Kaydetme işlemi
    model.save(model_name)
    print(f"{model_name} başarıyla kaydedildi.")

# Veri dosyalarını oku
df_stemmed = pd.read_csv("stemmed_data.csv")
df_lemmatized = pd.read_csv("lemmatized_data.csv")

# Parametre setleri
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Stemmed veri seti için modelleri eğit
for param in parameters:
    model_name = f"word2vec_stemmed_{param['model_type']}_win{param['window']}_dim{param['vector_size']}.model"
    create_word2vec_model(df_stemmed['stemmed_text'], model_name, param)

# Lemmatized veri seti için modelleri eğit
for param in parameters:
    model_name = f"word2vec_lemmatized_{param['model_type']}_win{param['window']}_dim{param['vector_size']}.model"
    create_word2vec_model(df_lemmatized['lemmatized_text'], model_name, param)
