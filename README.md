<h1>Mushroom Classification Project</h1>
Questo progetto si focalizza sull'analisi di un dataset relativo alla classificazione dei funghi, 
con l'obiettivo di costruire un modello di machine learning in grado di prevedere se un fungo è 
commestibile o velenoso basandosi su una serie di caratteristiche. L'obiettivo principale è 
identificare i fattori più rilevanti nella classificazione dei funghi e sviluppare un modello predittivo accurato.

<h3>Gli script inclusi nel progetto sono:</h3> <ul> <li>`mushroom_classification.py`: Questo script contiene 
  il codice principale per la pulizia dei dati, l'addestramento e la valutazione di modelli di classificazione, tra 
  cui la regressione logistica e altri algoritmi di machine learning. Include anche la gestione del preprocessing e 
  la valutazione delle metriche di performance.</li> </ul> <h3>I file CSV associati al progetto includono:</h3> 
  <ul> <li>`mushroom_cleaned.csv`: Il dataset già pulito che contiene caratteristiche dei funghi, come forma del 
    cappello, colore delle lamelle, odore, habitat e altre informazioni rilevanti per la classificazione tra commestibile e velenoso.</li> 
  </ul> <h2>Funzionalità degli script</h2> 
  <ol> 
    <li>Preprocessing dei dati: Lo script gestisce la pulizia dei dati, inclusa la gestione di valori mancanti o errati, 
    la trasformazione delle variabili categoriali e la preparazione del dataset per l'addestramento del modello.</li> 
    <li>Exploratory Data Analysis (EDA): segue un'analisi esplorativa del dataset, visualizzando la distribuzione delle caratteristiche attraverso grafici come boxplot. 
      Viene anche generata una matrice di correlazione per esaminare la relazione tra le variabili.</li> 
    <li>Training dei modelli: Vengono addestrati diversi modelli di machine 
        learning per la classificazione, tra cui regressione logistica, XGBoost e Random Forest. Lo script include anche la ricerca degli iperparametri ottimali 
        per migliorare le prestazioni.</li> 
    <li>Valutazione delle performance: Ogni modello viene valutato utilizzando la metrica accuracy. 
          Viene inoltre generata una matrice di confusione per studiare i falsi positivi e negativi, cruciali per la corretta classificazione dei funghi velenosi.</li> 
    </ol> 
  <h2>Dataset</h2>
  Il dataset utilizzato è trovabile su kaggle: https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset.
