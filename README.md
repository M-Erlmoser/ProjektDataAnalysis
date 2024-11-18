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

## Topic Modeling

1. **Themen nach Kategorie gruppieren**:
   - Die Themen werden mit der 2. Spalte kategoriesiert.
   - Danach erfolt die Vektorisierung der bereintigten Texte.
   - Mittels LDA wird das Topic Modeling durchgeführt.
   - Danach werden die Themen angezeigt.
2. **Alle Themen pro Kategorie speichern**:
   - Hierbei werden alle Themen pro Kategorie gespeichert.

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

  Genauigkeit vom Testset: 0.6739639820579768
Testset:
              precision    recall  f1-score   support

  Irrelevant       0.67      0.50      0.57      2592
    Negative       0.67      0.79      0.73      4519
     Neutral       0.66      0.60      0.63      3596
    Positive       0.69      0.72      0.71      4230

    accuracy                           0.67     14937
   macro avg       0.67      0.65      0.66     14937
weighted avg       0.67      0.67      0.67     14937

Genauigkeit vom Validierungsset: 0.7867867867867868
Validierungsset:
              precision    recall  f1-score   support

  Irrelevant       0.77      0.66      0.71       171
    Negative       0.73      0.88      0.80       266
     Neutral       0.86      0.72      0.78       285
    Positive       0.79      0.85      0.82       277

    accuracy                           0.79       999
   macro avg       0.79      0.78      0.78       999
weighted avg       0.79      0.79      0.79       999


Themen für Kategorie: Amazon
Top Wörter : win chance luck try interesting rewards exciting quiz played 100

Themen für Kategorie: ApexLegends
Top Wörter : ranked xboxshare apex need gaming good work day twitch season

Themen für Kategorie: AssassinsCreed
Top Wörter : like actually games good finished assassinscreed love flag early kassandra

Themen für Kategorie: Battlefield
Top Wörter : ban details occurred games player battlefieldv havent issue bfv battlefield4

Themen für Kategorie: Borderlands
Top Wörter : stream love dlc days help say playing fun excited weekend

Themen für Kategorie: CS-GO
Top Wörter : best win night awesome player like people good viewers minecraft

Themen für Kategorie: CallOfDuty
Top Wörter : streamer stream playing like fuck oh callofdutymodernwarfare going enjoy god

Themen für Kategorie: CallOfDutyBlackopsColdWar
Top Wörter : cod streaks gonna getting teaser fun probably oh beta definitely

Themen für Kategorie: Cyberpunk2077
Top Wörter : great delayed rtxon looks 2077 good story night going better

Themen für Kategorie: Dota2
Top Wörter : amazing like best ti9 fucking pls making looks happy finally

Themen für Kategorie: FIFA
Top Wörter : team toxic 21 ea week fucking player best thing mean

Themen für Kategorie: Facebook
Top Wörter : feel racist bitch people post mad live work aint glad

Themen für Kategorie: Fortnite
Top Wörter : like fuck care thank team song montage man actually good

Themen für Kategorie: Google
Top Wörter : life use twitter able shit companies list apple thanks great

Themen für Kategorie: GrandTheftAuto(GTA)
Top Wörter : shit love online good best real need people fuck hes

Themen für Kategorie: Hearthstone
Top Wörter : fucking playing hero week power did like thanks think day

Themen für Kategorie: HomeDepot
Top Wörter : ur like company longer work bad run having place customer

Themen für Kategorie: LeagueOfLegends
Top Wörter : love good super check playing like video 2020alienwaregames fucking captured

Themen für Kategorie: MaddenNFL
Top Wörter : love nfl 21 nfldropea mut working stuck smh games ratings

Themen für Kategorie: Microsoft
Top Wörter : like word fucking stupid entire software microsofts love 10 baby

Themen für Kategorie: NBA2K
Top Wörter : like yall nba2k fix ass 2k nba2k21 fuck wont garbage

Themen für Kategorie: Nvidia
Top Wörter : price geforce games people 3080 buy 3070 interesting good 3090

Themen für Kategorie: Overwatch
Top Wörter : love skins people started playing like fun shit overwatchleague atm

Themen für Kategorie: PlayStation5(PS5)
Top Wörter : like ill cause shit god playstation buy think eventually gonna

Themen für Kategorie: PlayerUnknownsBattlegrounds(PUBG)
Top Wörter : ban best pubgmobile good people mobile tonight unban love problem

Themen für Kategorie: RedDeadRedemption(RDR)
Top Wörter : bear getting like year life shit playing games pass look

Themen für Kategorie: TomClancysGhostRecon
Top Wörter : like update fix work doesnt support good playing ghostreconwildlands ubisoft

Themen für Kategorie: TomClancysRainbowSix
Top Wörter : love ranked super streamer 4k youre like stream head miss

Themen für Kategorie: Verizon
Top Wörter : data phone hey service customer yall experience app speed hands

Themen für Kategorie: WorldOfCraft
Top Wörter : earned achievement stop games like various months mmorpg gaming beautiful

Themen für Kategorie: Xbox(Xseries)
Top Wörter : like games microsoft look xsxfridgesweeps going pass ultimate xboxone amazing

Themen für Kategorie: johnson&johnson
Top Wörter : johnson vaccine powder baby covid19 companies selling cancer asbestos trust

Themen für jede Kategorie wurden in 'categorized_topics.csv' gespeichert.
Strukturierte Vorhersagen wurden in Prediction_Sentiment.csv gespeichert.

Process finished with exit code 0
    weighted avg       0.79      0.79      0.79       999


 
