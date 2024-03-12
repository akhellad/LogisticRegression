from sqlalchemy import create_engine
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

engine = create_engine('postgresql://akhellad:Mamandu91000.@localhost/AvisClients')

query = "SELECT * FROM avis"

df = pd.read_sql_query(query, con=engine)

nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

def process_review(review):
    doc = nlp(review)
    token = [token.lemma_.lower() for token in doc if token.text.lower() not in stop_words and token.text not in string.punctuation]
    return ' '.join(token)

df['processed_review'] = df['texte_avis'].apply(process_review)

X = df['processed_review']  
y = df['note']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors, y_train)

review_text = input("Review: ")


processed_text = process_review(review_text) 


vectorised_text = vectorizer.transform([processed_text])


prediction = model.predict(vectorised_text)
prediction_proba = model.predict_proba(vectorised_text)


note = prediction[0]
probabilities = prediction_proba[0]
print(f"Note prediction : {note}")
print(f"Probabilities : {probabilities}")


if note == 0:
    print("The review is predicted as negative.")
else:
    print("The review is predicted as positive.")