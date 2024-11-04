import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

##Stopwörter wurden manuell ausgewählt
custom_stopwords = [
 "game", "nvidia", "im", "overwatch", "world", "warcraft", "xbox", "series", "xboxseriesx", "verizon","rainbowsixsiege",
    "siege", "ghostreconbreakpoint", "just", "rdr2", "pubg", "ps5", "makes", "really", "make", "gotta", "today", "franchise"
    "league", "legends", "leagueoflegends", "home", "depot", "play", "guys", "new", "gta", "google", "dont", "fortnite",
    "facebook", "didnt", "know", "20", "time", "dota", "cyberpunk2077", "cyberpunk", "black", "ops", "cold", "duty",
    "callofduty", "csgo", "want", "borderlands", "worldofwarcraft", "playstation5", "ive", "madden", "franchise", "league",
    "dota2", "esports", "wait", "blackopscoldwar", "war", "warzone", "modernwarfare", "battlefield", "creed", "assassins",
    "apexlegends", "got", "odyssey",
]
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
## TOPIC MODELING
# Themen nach Kategorien gruppieren
grouped_topics = test_df.groupby(test_df.columns[1])
def topic_modeling_per_category(text_data, n_topics=1, no_top_words=10):
    # Bereinigen der Texte
    cleaned_texts = text_data.apply(clean_text)
    # Vektorisierung der bereinigten Texte
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words=custom_stopwords, max_features=10000)
    text_counts = count_vect.fit_transform(cleaned_texts)
    # LDA Topic Modeling
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=52)
    lda.fit(text_counts)
    # Themen anzeigen
    feature_names = count_vect.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append(f"Top Wörter : {top_words}")
    return topics
# Alle Themen pro Kategorie speichern
all_topics = {}
for category, group in grouped_topics:
    print(f"\nThemen für Kategorie: {category}")
    text_data = group.iloc[:, 3] 
    topics = topic_modeling_per_category(text_data)
    all_topics[category] = topics
    for topic in topics:
        print(topic)

## Output CSV Datei
# Ergebnisse der Themenanalyse in eine CSV-Datei schreiben
output_topics = []
for category, topics in all_topics.items():
    for topic in topics:
        output_topics.append({"Kategorie": category, "Thema": topic})

topics_output_df = pd.DataFrame(output_topics)
topics_output_df.to_csv('categorized_topics.csv', index=False)
print("Themen für jede Kategorie wurden in 'categorized_topics.csv' gespeichert.")

# Vergleich: Succes vs. Fail
test_df['actual_sentiment'] = test_df.iloc[:, 2]
test_df['predicted_sentiment'] = y_val_pred       
# Erfolg/Nicht-Erfolg Spalte hinzufügen
test_df['success'] = test_df['actual_sentiment'] == test_df['predicted_sentiment']
test_df['success'] = test_df['success'].replace({True: 'succes', False: 'fail'})
# In eine neue CSV-Datei speichern
output_path = 'Prediction_Sentiment.csv'
test_df.to_csv(output_path, index=False)
print(f'Strukturierte Vorhersagen wurden in {output_path} gespeichert.')
