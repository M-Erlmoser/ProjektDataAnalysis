import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

## Daten bereinigung
# Textbereinigungsfunktion
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text
# CSV-Dateien laden
train_df = pd.read_csv('twitter_training.csv')
test_df = pd.read_csv('twitter_validation.csv')
# Sicherstellen, dass keine NaN-Werte enthalten sind
train_df.iloc[:, 3] = train_df.iloc[:, 3].fillna('')
test_df.iloc[:, 3] = test_df.iloc[:, 3].fillna('')
# Bereinigen des Textteils
train_df['cleaned_text'] = train_df.iloc[:, 3].apply(str).apply(clean_text)
test_df['cleaned_text'] = test_df.iloc[:, 3].apply(str).apply(clean_text)
# Features und Labels extrahieren
X = train_df['cleaned_text']
y = train_df.iloc[:, 2]
# Aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Vektoriesierung
# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

## Modell
# Training
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
# Vorhersagen und Evaluation
y_pred = model.predict(X_test_tfidf)
print(f'Genauigkeit vom Testset: {accuracy_score(y_test, y_pred)}')
print("Testset:")
print(classification_report(y_test, y_pred))
# Validierungsdaten bewerten
X_val_tfidf = tfidf.transform(test_df['cleaned_text'])
y_val_pred = model.predict(X_val_tfidf)
# Genauigkeit für das Validierungs-Set berechnen
validation_accuracy = accuracy_score(test_df.iloc[:, 2], y_val_pred)
print(f'Genauigkeit vom Validierungsset: {validation_accuracy}')
# Bericht für das Validierungsset
validation_report = classification_report(test_df.iloc[:, 2], y_val_pred)
print("Validierungsset:")
print(validation_report)

## Output CSV Datei
# Vergleich: Succes vs. Fail
test_df['actual_sentiment'] = test_df.iloc[:, 2]
test_df['predicted_sentiment'] = y_val_pred       
# Erfolg/Nicht-Erfolg Spalte hinzufügen
test_df['success'] = test_df['actual_sentiment'] == test_df['predicted_sentiment']
test_df['success'] = test_df['success'].replace({True: 'succes', False: 'fail'})
# In eine neue CSV-Datei speichern
output_path = 'Vorhersage_Sentiment.csv'
test_df.to_csv(output_path, index=False)
print(f'Strukturierte Vorhersagen wurden in {output_path} gespeichert.')
