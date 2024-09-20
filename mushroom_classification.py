"""
Questo script implementa un'analisi e un modello predittivo basato sul dataset "Mushroom Classification Dataset".
Il dataset è composto da 9 colonne, di cui 1 colonna di output ("class") che rappresenta la classificazione del fungo
tra due categorie (1=commestibile, 0=velenoso) e 8 colonne di input che descrivono le caratteristiche morfologiche dei funghi.

Il dataset è osservabile su kaggle: https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset/

@author: roberto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carico il CSV in un Dataframe Pandas
path = f"data/mushroom_cleaned.csv"
data = pd.read_csv(path)

"""
FASE DI EDA
"""

# Stampo le informazioni sui dati per verificare la presenza di NaN e per controllare il tipo dei dati
print(data.info())  # Nessun valore null e tutti dati numerici
print(data.head())

# Inizializzo lista con i nomi delle variabili input
features = data.columns.drop("class").values

"""
PLOTTING PER OSSERVARE OUTLIERS
"""
print("---------------------BOXPLOT----------------------")
for feature in features:
    plt.title(label=feature)
    plt.boxplot(data[feature], sym="o")
    plt.show()

# Funzione per rimuovere gli outliers con il metodo IQR
def remove_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Rimuovo gli outliers e sovrascrivo il dataset
data_cleaned = remove_outliers_iqr(data)

# Rimuovo eventuali NaN residui dopo la rimozione degli outliers
data_cleaned = data_cleaned.dropna()

print("\n---------------------MATRICE DI CORRELAZIONE ----------------------")
plt.matshow(data_cleaned.drop('class', axis=1).corr(), vmin=-1, vmax=1)
plt.xticks(np.arange(0, data_cleaned.drop('class', axis=1).shape[1]), features, rotation=90)
plt.yticks(np.arange(0, data_cleaned.drop('class', axis=1).shape[1]), features)
plt.title("Matrice di Correlazione dei valori input")
plt.colorbar()
plt.show()

"""
FASE DI SPLITTING
"""
from sklearn.model_selection import train_test_split

X = data_cleaned.drop('class', axis=1)
y = data_cleaned['class']

# Divido il dataset in train+val e test (80% train+val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Divido il dataset train+val in train e validation set (80% train, 20% validation dei rimanenti dati)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

"""
FASE DI ADDESTRAMENTO E HYPERPARAMETER TUNING
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Parametri da esplorare per Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None]
}

# Valutazione con Cross-Validation per Random Forest
print("\n---------------------HYPERPARAMETER TUNING RFC ----------------------")
best_rf_model = None
best_rf_score = 0

for n_estimators in rf_params['n_estimators']:
    for max_depth in rf_params['max_depth']:
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        rf_model.fit(X_train, y_train)
        
        # Valutazione sul validation set
        y_val_pred = rf_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        if val_accuracy > best_rf_score:
            best_rf_score = val_accuracy
            best_rf_model = rf_model
        
        print(f"Random Forest: n_estimators={n_estimators}, max_depth={max_depth}, "
              f"accuracy={val_accuracy:.4f}")

print(f"\nMiglior modello Random Forest: accuracy={best_rf_score:.4f}")

# Aggiungiamo la valutazione per Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_val_pred_lr = lr_model.predict(X_val)
val_accuracy_lr = accuracy_score(y_val, y_val_pred_lr)
print(f"Logistic Regression: accuracy={val_accuracy_lr:.4f}")

# Aggiungiamo la valutazione per XGBoost
print("\n---------------------ADDESTRAMENTO XGBOOST ----------------------")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Valutazione sul validation set
y_val_pred_xgb = xgb_model.predict(X_val)
val_accuracy_xgb = accuracy_score(y_val, y_val_pred_xgb)
print(f"XGBoost: accuracy={val_accuracy_xgb:.4f}")

"""
CONFRONTO TRA MODELLI SUL TEST SET
"""
# Valutazione sul test set con il miglior Random Forest
y_test_pred_rf = best_rf_model.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)


# Valutazione sul test set con Logistic Regression
y_test_pred_lr = lr_model.predict(X_test)
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)


# Valutazione sul test set con XGBoost
y_test_pred_xgb = xgb_model.predict(X_test)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)

# Confronto delle performance finali (Random Forest, Logistic Regression e XGBoost)
print("\n---------------------CONFRONTO TRA MODELLI----------------------")
print(f"Accuratezza Random Forest sul test set: {test_accuracy_rf:.4f}")
print(f"Accuratezza Logistic Regression sul test set: {test_accuracy_lr:.4f}")
print(f"Accuratezza XGBoost sul test set: {test_accuracy_xgb:.4f}")

"""
SI E' RISCONTRATO CHE L'APPROCCIO CON RANDOM FOREST PRODUCE UN RISULTATO MIGLIORE
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n---------------------MATRICE DI CONFUSIONE RFC----------------------")
cm = confusion_matrix(y_test, y_test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Commestibile", "Velenoso"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione - Random Forest")
plt.xlabel("Valori Predetti")
plt.ylabel("Valori Reali")
plt.show()

# Calcolo delle metriche della matrice di confusione
tn, fp, fn, tp = cm.ravel()
print(f"Veri Negativi (TN): {tn}")
print(f"Falsi Positivi (FP): {fp}")
print(f"Falsi Negativi (FN): {fn}")
print(f"Veri Positivi (TP): {tp}")
print(f"Sensitivitá (TPR): {tp/(tp+fn)}")
print(f"Specificitá (TNR): {tn/(tn+fp)}")
print(f"Precisione (PPV): {tp/(tp+fp)}")


