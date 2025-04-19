"""Doğal Dil İşleme (Natural Language Processing - NLP)
NLP, insan dilini anlamak ve işlemek için makine öğrenmesi yöntemlerini kullanır. Metin verileriyle çalışır.
Uygulamalar:

Metin sınıflandırma (örneğin, spam tespiti)
Duygu analizi
Çeviri ve dil modelleme
Sözcük gömme (Word Embeddings)


1. Temel Metin Ön İşleme (Tokenization, Stopword Removal)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

text = "Doğal dil işleme, insan dilini anlamak için harika bir alandır!"
tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stopwords.words('turkish')]

print("Temizlenmiş Tokenlar:", filtered)
2. Metin Sınıflandırma (Spam Detection)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = ["Bu mesajı kazanmak için yanıtla!", "Yarın toplantımız var.", "Kazandınız, hemen tıklayın!", "Nasılsın bugün?"]
labels = [1, 0, 1, 0]  # 1: spam, 0: normal

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, pred))
3. Duygu Analizi (Sentiment Analysis) — Türkçe BERT

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
result = classifier("Bu film gerçekten çok güzeldi, bayıldım!")
print("Duygu:", result)
4. Çeviri (English to Turkish Translation)
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-tr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "I love natural language processing!"
tokens = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**tokens)
translation = tokenizer.decode(translated[0], skip_special_tokens=True)

print("Çeviri:", translation)
5. TF-IDF ile Vektörleştirme
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["Makine öğrenmesi harika bir konudur.", "Doğal dil işleme ile metinler anlaşılabilir."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

print("TF-IDF Matrisi:")
print(X.toarray())
6. Word2Vec ile Sözcük Gömme
from gensim.models import Word2Vec

sentences = [["doğal", "dil", "işleme"], ["makine", "öğrenmesi"], ["sinir", "ağı"]]
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=2)

print("Vektör (doğal):", model.wv['doğal'])
7. Lemmatizasyon (spaCy)
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The children are playing in the garden.")

print("Lemmatize Edilmiş:")
for token in doc:
    print(f"{token.text} -> {token.lemma_}")
8. Named Entity Recognition (Varlık Tanıma)
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Barack Obama was the president of the United States.")

print("Tanımlanan Varlıklar:")
for ent in doc.ents:
    print(ent.text, ent.label_)
9. Dil Modeli Kullanarak Metin Tamamlama
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Natural language processing is", max_length=30, num_return_sequences=1)

print("Tamamlanan Metin:\n", result[0]['generated_text'])
10. Benzer Sözcükleri Bulma (Word2Vec)

from gensim.models import Word2Vec

sentences = [["nlp", "doğal", "dil", "işleme"], ["makine", "öğrenme"], ["sinir", "ağları"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)

similar_words = model.wv.most_similar('nlp')
print("nlp kelimesine benzer sözcükler:", similar_words)"""