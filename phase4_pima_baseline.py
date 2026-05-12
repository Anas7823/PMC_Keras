import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# RÉPONSE À LA QUESTION PRÉLIMINAIRE
# ==========================================
# Un modèle "stupide" qui prédirait systématiquement 0 (non-diabétique) 
# pour chaque patient sans rien apprendre atteindrait une accuracy de 65.1% 
# (car 500 cas sur 768 sont négatifs). 
# Conséquence : Si notre modèle stagne autour de 0.65 d'accuracy, c'est un red flag. 
# Cela signifie qu'il n'a rien appris et se contente de parier sur la classe majoritaire.

# ==========================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# ==========================================
print("--- Chargement du dataset Pima Indians Diabetes ---")
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

print("\n--- Distribution des classes (Outcome) ---")
class_counts = df['Outcome'].value_counts()
class_props = df['Outcome'].value_counts(normalize=True) * 100
for cls, count, prop in zip(class_counts.index, class_counts.values, class_props.values):
    print(f"Classe {cls} : {count} ({prop:.1f}%)")

print("\n--- Diagnostic de la donnée : Zéros physiologiquement impossibles ---")
zeros_count = (df == 0).sum()
print(zeros_count[zeros_count > 0])
# Note : Ces zéros sont des NaN déguisés. Pour un pipeline strict, 
# il faudrait les imputer (par exemple avec la médiane). On les laisse bruts 
# pour cette baseline afin de voir la capacité d'adaptation du réseau.

# ==========================================
# 2. PRÉPARATION DES DONNÉES
# ==========================================
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation stricte sans data leakage
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==========================================
# 3. ARCHITECTURE ET ENTRAÎNEMENT DU MODÈLE
# ==========================================
def build_binary_classifier():
    model = keras.Sequential([
        keras.layers.Input(shape=(8,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        # Couche de sortie : 1 neurone + activation Sigmoid (renvoie une proba entre 0 et 1)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilation : Binary Crossentropy est LA loss standard pour la classification à 2 classes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_binary_classifier()

print("\n--- Démarrage de l'entraînement ---")
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2, # Keras prélève 20% de X_train_norm pour la validation
    verbose=1
)

# ==========================================
# 4. ÉVALUATION ET DIAGNOSTICS
# ==========================================
print("\n--- Résultats de la Baseline ---")
val_accuracies = history.history['val_accuracy']
max_val_acc = max(val_accuracies)
print(f"Baseline max val_accuracy : {max_val_acc:.4f}")

print("\n--- Test Adversarial : Le modèle prédit-il toujours 0 ? ---")
preds = model.predict(X_test_norm)
preds_mean = preds.mean()
print(f"Moyenne des probabilités prédites sur le Test Set : {preds_mean:.4f}")
print("Attendu : ~0.35. Si proche de 0.05, le modèle a convergé vers la classe majoritaire (Red Flag).")