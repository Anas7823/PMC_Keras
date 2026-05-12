import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# RÉCUPÉRATION ET PRÉPARATION DES DONNÉES
# ==========================================
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# ==========================================
# ARCHITECTURE DU MODÈLE
# ==========================================
def build_regression_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==========================================
# ENTRAÎNEMENT AVEC TENSORBOARD
# ==========================================
def train_with_tensorboard(X_train_data, y_train_data, X_val_data, y_val_data, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""
    
    # Création du timestamp et du chemin
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = os.path.join("logs", "fit", f"{run_name}_{timestamp}")
    
    # Instanciation du callback TensorBoard
    # histogram_freq=1 permet de suivre la distribution des poids du réseau à chaque epoch
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Construction du modèle adaptatif selon la dimension d'entrée
    model = build_regression_model(input_dim=X_train_data.shape[1])
    
    print(f"Démarrage de l'entraînement pour le run : '{run_name}'...")
    
    # Entraînement en mode silencieux (verbose=0) car on va lire les logs sur TensorBoard
    history = model.fit(
        X_train_data, y_train_data,
        validation_data=(X_val_data, y_val_data),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0 
    )
    
    print(f"✅ Run '{run_name}' terminé. Logs enregistrés dans {log_dir}")
    return model, history

# ==========================================
# EXÉCUTION DES DEUX RUNS COMPARATIFS
# ==========================================
print("--- Lancement des entraînements ---")

# Run 1 : données normalisées (bon comportement attendu)
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val, 
    run_name="california_norm"
)

# Run 2 : données brutes (comportement dégradé à observer)
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val, 
    run_name="california_raw"
)

# ==========================================
# INTERPRÉTATION DES RÉSULTATS (Hypothèse)
# ==========================================
# DIAGNOSTIC OBSERVÉ POUR LE RUN "california_norm" :
# Nous sommes dans la situation (b) : "train descend, val stagne ou remonte (overfitting)".
# Comme vu lors de la phase 2, sur les 40-50 premières epochs, les courbes de loss 
# (train et validation) descendent ensemble harmonieusement (situation "a"). 
# Mais vers la fin de l'entraînement (epochs 60 à 100), la loss d'entraînement continue 
# de creuser lentement, tandis que la val_loss stagne autour de 0.28 et se met à osciller. 
# Le modèle commence à faire du surapprentissage (mémorisation du set d'entraînement).
# C'est ce que nous corrigerons à la phase suivante avec du Early Stopping.
#
# DIAGNOSTIC OBSERVÉ POUR LE RUN "california_raw" :
# La val_loss est cataclysmique (des valeurs immenses, parfois des NaN). 
# L'absence de normalisation empêche les gradients de converger. TensorBoard va 
# montrer une échelle Y complètement écrasée à cause de ces valeurs aberrantes.