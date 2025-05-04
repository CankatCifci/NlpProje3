import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# 1. CSV dosyasını yükle
df = pd.read_csv('legal_text_classification.csv')

# 2. NaN ve sayısal olmayan metinleri temizle
df = df[df['case_text'].notna()]
df = df[~df['case_text'].apply(lambda x: isinstance(x, (float, int)))]

# 3. Tüm metni birleştir
full_text = ' '.join(df['case_text'].tolist()).lower()

# 4. Noktalama işaretlerini kaldır, kelimelere ayır
words = re.findall(r'\b[a-z]{2,}\b', full_text)  # sadece harf ve 2+ uzunlukta

# 5. Kelime frekanslarını say
word_counts = Counter(words)

# 6. Frekansları sırala
sorted_counts = sorted(word_counts.values(), reverse=True)

# 7. Sıralama ve frekansları log-log olarak çiz
ranks = np.arange(1, len(sorted_counts) + 1)
plt.figure(figsize=(10,6))
plt.loglog(ranks, sorted_counts, marker=".")
plt.title("Zipf Yasası - Legal Text Classification Dataset")
plt.xlabel("Kelime Sıralaması (log)")
plt.ylabel("Kelime Frekansı (log)")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
