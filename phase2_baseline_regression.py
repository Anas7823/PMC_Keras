import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# RÉCUPÉRATION ET PRÉPARATION DES DONNÉES (Phase 1)
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
# CONSTRUCTION DU MODÈLE
# ==========================================

# POURQUOI NE PAS METTRE `activation='sigmoid'` SUR LA COUCHE DE SORTIE ?
# Réponse : La fonction sigmoïde "écrase" toutes ses sorties entre 0 et 1.
# Dans notre dataset, la cible (le prix) est exprimée en centaines de milliers de dollars 
# et varie entre 0.15 et 5.0 (soit de 15 000 $ à 500 000 $). 
# Si on utilise un sigmoïde, le réseau ne pourra mathématiquement jamais prédire un prix 
# supérieur à 1.0 (100 000 $). Le modèle sera donc "plafonné" et inutilisable pour 80% des maisons.
# Pour une régression continue non bornée (ou bornée au-delà de 1), on laisse une activation linéaire (par défaut).

def build_regression_model(input_dim):
    """
    Construit et compile un Perceptron Multicouche (PMC) pour la régression.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        # Couche de sortie : 1 seul neurone, pas d'activation (linéaire) car on prédit une valeur continue
        layers.Dense(1) 
    ])
    
    # Compilation : 
    # - Adam est un optimiseur robuste par défaut
    # - MSE (Mean Squared Error) pénalise fortement les grandes erreurs, guidant bien l'optimiseur
    # - MAE (Mean Absolute Error) est ajouté car plus interprétable humainement
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# ==========================================
# ENTRAÎNEMENT ET ÉVALUATION
# ==========================================

input_dim = X_train_norm.shape[1]
model = build_regression_model(input_dim)

print("\n--- Architecture du modèle ---")
model.summary()

print("\n--- Début de l'entraînement ---")
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_norm, y_val),
    verbose=1
)

print("\n--- Évaluation sur le set de test ---")
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")
print(f"Erreur moyenne estimée : {test_mae * 100000:.0f} $")