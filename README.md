Intelligent Data Analysis (IDA) 🧮
Meine Lösungen und Notebooks aus dem Kurs Intelligent Data Analysis – eine
praktische Tour durch die Grundlagen des maschinellen Lernens, Schritt für Schritt
in Jupyter Notebooks umgesetzt (Python).
Von der ersten Datenexploration mit Pandas bis hin zu neuronalen Netzen, Kernel-Methoden
und Bayes'schem Lernen: jedes Lab baut auf dem vorherigen auf.
---
📚 Inhalt
Die Labs decken die wichtigsten ML-Themen in logischer Reihenfolge ab:
Lab	Thema	Worum es geht
01	Pandas & Pylab	Daten laden, erkunden, visualisieren
02	Model Analysis	Modelle verstehen und bewerten
03	Problem Analysis & Data Preprocessing	Datenaufbereitung, Feature Engineering
04	Decision Trees	Entscheidungsbäume
05	Random Forest	Ensemble-Methoden
06	Linear Classification	Lineare Klassifikatoren
07	Linear Regression	Lineare Regression
08	Evaluation	Modellbewertung & Metriken
09	Neural Networks	Neuronale Netze
10	Kernel Methods	Kernel-basierte Verfahren (z.B. SVM)
11	Bayesian Learning	Bayes'sches Lernen
12	Logistic Regression	Logistische Regression
Zusätzliche Notebooks & Experimente
Bayesian Optimization – Hyperparameter-Optimierung
Gaussian Process – Gaußprozesse zur Regression
Monte Carlo Integration – numerische Integration per Sampling
Markov Text Generator – Textgenerierung mit Markov-Ketten
Einkommensklassen – Klassifikation von Einkommensgruppen
SPECT – Klassifikation auf dem SPECT-Heart-Datensatz
---
🛠️ Verwendete Technologien
Python & Jupyter Notebook
NumPy, Pandas, Matplotlib
scikit-learn
(je nach Notebook) weitere Pakete für spezielle Verfahren
---
🚀 Nutzung
```bash
# Repo klonen
git clone https://github.com/mendeltem/IDA.git
cd IDA

# Empfohlen: virtuelle Umgebung
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install numpy pandas matplotlib scikit-learn jupyter

# Jupyter starten
jupyter notebook
```
Anschließend einfach den gewünschten Lab-Ordner öffnen und das Notebook durchlaufen.
---
📂 Struktur
```
IDA/
├── Lab01_LearningPandasandPylab/
├── Lab02_Model_Analysis/
├── Lab03_Problem_Analysis_and_Data_Preprocessing/
├── ...
├── Lab12_LogisticRegression/
├── Projekte/                  # größere Projektarbeiten
├── BayesianOptimization.py
├── MonteCarloIntegration.py
└── ...
```
---
🎯 Ziel
Dieses Repo dokumentiert meinen Lernweg durch die Grundlagen der intelligenten
Datenanalyse – als Referenz und zum Nachvollziehen der einzelnen ML-Verfahren
anhand konkreter, lauffähiger Beispiele.
> Hinweis: Es handelt sich um Übungs- und Kursmaterial. Die Notebooks sind als
> Lernartefakte gedacht, nicht als produktionsreifer Code.
