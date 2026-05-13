import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# CHOIX D'ARCHITECTURE (Multiclass vs Regression vs Ordinal)
# ==========================================
# La qualité du vin est une variable ordinale (3 < 4 < 5...). 
# 1. Classification Multiclasse (Choix actuel) : 
#    - Avantage : Simple à implémenter.
#    - Perte : On perd l'information d'ordre. Pour le modèle, l'erreur entre "low" (0) et 
#      "high" (2) est pénalisée de la même façon qu'entre "low" (0) et "medium" (1).
# 2. Régression (prédire un seul nombre) :
#    - Avantage : Préserve l'ordre et la notion de distance continue.
#    - Perte : Le modèle peut prédire des valeurs hors limites (ex: 9.5 ou 2.1) et 
#      nécessite d'arrondir les prédictions a posteriori.
# 3. Classification Ordinale :
#    - Avantage : Le plus rigoureux mathématiquement pour ce type de données.
#    - Perte : Pas de support natif simple dans Keras de base (nécessite des encodages 
#      spécifiques ou des architectures multi-sorties complexes).

# ==========================================
# 1. CHARGEMENT ET AGRÉGATION DES DONNÉES
# ==========================================
print("--- Chargement du dataset Wine Quality ---")
wine_url = "./dataset/phase7/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')

print("\nDistribution des qualités brutes :")
print(df_wine['quality'].value_counts().sort_index())

def map_quality(q):
    if q <= 4:
        return 0  # low
    elif q <= 6:
        return 1  # medium
    else:
        return 2  # high

df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)

print("\nDistribution agrégée (3 classes) :")
print(df_wine['quality_3class'].value_counts().sort_index())

X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

# ==========================================
# 2. PRÉPARATION DES DONNÉES (Stratify)
# ==========================================
# Le stratify est OBLIGATOIRE ici vu le déséquilibre (plus de 82% de classe 1).
# Sans lui, le test set pourrait ne contenir aucun vin "low" (classe 0).
X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==========================================
# 3. ARCHITECTURE DU MODÈLE MULTICLASSE
# ==========================================
def build_wine_baseline():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(11,)),
        layers.Dense(32, activation='relu'),
        # Sortie : 3 neurones (un par classe), activation softmax pour obtenir des probabilités
        layers.Dense(3, activation='softmax')
    ])
    
    # sparse_categorical_crossentropy accepte directement [0, 1, 2] comme cible
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_wine_baseline()

print("\n--- Démarrage de l'entraînement ---")
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0 # Silencieux pour lire directement les résultats finaux
)

# ==========================================
# 4. ÉVALUATION ET DIAGNOSTICS
# ==========================================
val_accuracies = history.history['val_accuracy']
print(f"\nBaseline max val_accuracy : {max(val_accuracies):.4f}")

print("\n--- Diagnostic : Le modèle capitule-t-il sur la classe 1 (Medium) ? ---")
# On fait prédire le set de test
y_pred_probs = model.predict(X_test_norm)
# On récupère l'index de la proba la plus haute pour chaque ligne
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\nMatrice de confusion sur le set de test :")
print(confusion_matrix(y_test, y_pred_classes))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred_classes, zero_division=0))