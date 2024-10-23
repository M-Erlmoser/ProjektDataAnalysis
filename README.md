# Sentiment Analysis with Logistic Regression

## Projektbeschreibung

Dieses Projekt führt eine **Sentiment-Analyse** durch, bei der ein **Logistic-Regression-Modell** auf einem Datensatz trainiert wird, um die Stimmung (Sentiment) von Texten zu klassifizieren. Der Text wird dabei in verschiedene Kategorien wie **Positiv**, **Negativ**, **Neutral** oder **Irrelevant** eingeteilt. Das Modell wird mit einem Trainingsdatensatz trainiert und mit einem Validierungsdatensatz getestet.

Am Ende werden die **Vorhersagen** in einer strukturierten CSV-Datei gespeichert, die auch Informationen über den Erfolg der Vorhersagen sowie die **Genauigkeit**, **Präzision**, **Recall** und den **F1-Score** des Modells enthält.

## Struktur

Das Skript führt die folgenden Schritte durch:
1. **Datenaufbereitung**:
   - Die Textdaten werden bereinigt (Entfernung von URLs, Erwähnungen, Sonderzeichen, etc.).
   - Fehlende Werte (NaN) in den Textspalten werden durch leere Strings ersetzt.
   
2. **Modelltraining**:
   - Das Modell wird mit dem Trainingsdatensatz trainiert.
   - Eine **TF-IDF-Vektorisierung** wird verwendet, um den Text in numerische Features zu überführen.
   - Ein **Logistic-Regression-Modell** wird trainiert.

3. **Vorhersage und Evaluation**:
   - Vorhersagen werden für das Test- und Validierungs-Set gemacht.
   - Die **Genauigkeit**, **Präzision**, **Recall** und der **F1-Score** werden für das Validierungs-Set berechnet und angezeigt.

4. **Erfolg/Nicht-Erfolg**:
   - Die Vorhersagen werden mit den tatsächlichen Sentiments verglichen, und es wird festgelegt, ob die Vorhersage erfolgreich war oder nicht.

5. **Ergebnisse speichern**:
   - Eine CSV-Datei mit den strukturierten Vorhersagen wird generiert, die folgende Informationen enthält:
     - Tatsächliches Sentiment
     - Vorhergesagtes Sentiment
     - Erfolg oder Misserfolg der Vorhersage
     - Validierungsgenauigkeit (optional)

## Dateien

- **`twitter_training.csv`**: Trainingsdatensatz, der die Textdaten und ihre zugehörigen Sentiments enthält.
- **`twitter_validation.csv`**: Validierungsdatensatz, der zum Testen des Modells verwendet wird.
- **`Prediction_Sentiment.csv`**: Ergebnisdatei mit den strukturierten Vorhersagen und Erfolgsangaben.

## Benötigte Bibliotheken

Die folgenden Python-Bibliotheken werden verwendet und müssen installiert werden:

```bash
pip install pandas scikit-learn
```
## Ausführung des Skripts
**Um das Skript auszuführen**:

1. Lade die notwendigen CSV-Dateien (twitter_training.csv und twitter_validation.csv) in das Projektverzeichnis.
2.	Stelle sicher, dass alle benötigten Bibliotheken installiert sind.
3.	Führe das Skript aus:
 ```bash
 python SentimentAnalyse.py
 ```
## Ausgabe

**Das Skript generiert folgende Ausgaben**:

1.	Konsolenausgaben:
	-	Genauigkeit des Modells auf dem Test-Dataset.
	-	Klassifikationsbericht für das Test- und Validierungs-Dataset (inkl. Präzision, Recall, F1-Score).
	-	Genauigkeit des Validierungs-Sets.
2.	CSV-Ausgabe (**.csv):
**Diese Datei enthält**:

  -	Den Originaltext.
  -	Das tatsächliche Sentiment.
  -	Das vorhergesagte Sentiment.
  -	Eine Spalte, die angibt, ob die Vorhersage erfolgreich war (Succes/Fail).
  -	Die Genauigkeit des Modells auf dem Validierungs-Set.

## Beispielausgabe
**Beispiel für Konsolenausgabe**:

    Genauigkeit vom Validierungsset: 0.79
    Validierungsset:
                   precision    recall    f1-score   support

    Irrelevant         0.77      0.66      0.71       171
      Negative         0.73      0.88      0.80       266
      Neutral          0.86      0.72      0.78       285
      Positive         0.79      0.85      0.82       277

    accuracy                               0.79       999
    macro avg          0.79      0.78      0.78       999
    weighted avg       0.79      0.79      0.79       999


 
