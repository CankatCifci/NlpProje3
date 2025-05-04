import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF vektörleştirme fonksiyonu
def create_tfidf_df(text_data, file_name):
    try:
        vectorizer = TfidfVectorizer()
        print("Vektörleştirme işlemi başlatıldı...")
        X = vectorizer.fit_transform(text_data)
        df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        print(f"TF-IDF işlemi tamamlandı. Dosya kaydediliyor: {file_name}")
        df_tfidf.to_csv(file_name, index=False)
        print(f"{file_name} başarıyla kaydedildi.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Veri dosyalarını kontrol et ve oku
try:
    df_stemmed = pd.read_csv("stemmed_data.csv")
    print("Stemmed veri seti başarıyla yüklendi.")
    print(df_stemmed.head())  # İlk 5 satırı kontrol et
    if 'stemmed_text' not in df_stemmed.columns:
        print("'stemmed_text' kolonu bulunamadı. Kolon adı yanlış olabilir.")

    df_lemmatized = pd.read_csv("lemmatized_data.csv")
    print("Lemmatized veri seti başarıyla yüklendi.")
    print(df_lemmatized.head())  # İlk 5 satırı kontrol et
    if 'lemmatized_text' not in df_lemmatized.columns:
        print("'lemmatized_text' kolonu bulunamadı. Kolon adı yanlış olabilir.")

    # TF-IDF işlemi uygula
    create_tfidf_df(df_stemmed['stemmed_text'], "tfidf_stemmed.csv")
    create_tfidf_df(df_lemmatized['lemmatized_text'], "tfidf_lemmatized.csv")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
