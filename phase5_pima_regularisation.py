import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# OBJECTIFS ET STRATÉGIE (Cible déclarée)
# ==========================================
# Objectif chiffré : Atteindre une val_accuracy stable supérieure à 79% (le baseline étant ~76-78%), 
# tout en repoussant l'overfitting (la val_loss ne doit pas remonter brutalement).
# 
# Ordre des leviers essayés et pourquoi :
# 1. Early Stopping : Le garde-fou universel. Arrête l'hémorragie dès que le modèle surapprend.
# 2. Régularisation L2 : Force le réseau à utiliser de petits poids répartis (pénalité douce), 
#    très adapté aux datasets denses et continus comme les données médicales.
# 3. Dropout : Casse la co-adaptation (pénalité forte). Sur un tout petit dataset (768 lignes), 
#    ça peut parfois être trop agressif et ralentir l'apprentissage, on le teste en dernier.
# 4. Class_weight (prochaine étape) : Si le modèle stagne, on forcera l'attention sur la classe minoritaire.

# ==========================================
# 1. RÉCUPÉRATION ET PRÉPARATION DES DONNÉES
# ==========================================
print("--- Chargement et préparation des données ---")
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation stricte sans data leakage
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==========================================
# 2. FONCTION DE CONSTRUCTION DU MODÈLE
# ==========================================
def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    """
    Modèle Pima avec régularisation L2 optionnelle et Dropout optionnel.
    """
    model = keras.Sequential()
    
    # Couche 1
    model.add(
        keras.layers.Input(shape=(8,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
        
    # Couche 2
    model.add(layers.Dense(32, activation='relu', 
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
        
    # Couche de sortie (jamais de régularisation destructive sur la sortie)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. ENTRAÎNEMENT DES 3 CONFIGURATIONS
# ==========================================
# Callback Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

# --- Configuration 1 : Baseline non régularisé ---
print("\nEntraînement Config 1 : Baseline...")
model_baseline = build_pima_regularized(l2_lambda=0.0, use_dropout=False)
history_baseline = model_baseline.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2,
    callbacks=[early_stopping], verbose=0
)
stop_epoch_base = len(history_baseline.history['val_loss'])
max_acc_base = max(history_baseline.history['val_accuracy'])
print(f"-> Arrêt à l'epoch {stop_epoch_base} | Max val_accuracy : {max_acc_base:.4f}")

# --- Configuration 2 : L2 seul ---
print("\nEntraînement Config 2 : L2 seul...")
model_l2 = build_pima_regularized(l2_lambda=0.01, use_dropout=False)
history_l2 = model_l2.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2,
    callbacks=[early_stopping], verbose=0
)
stop_epoch_l2 = len(history_l2.history['val_loss'])
max_acc_l2 = max(history_l2.history['val_accuracy'])
print(f"-> Arrêt à l'epoch {stop_epoch_l2} | Max val_accuracy : {max_acc_l2:.4f}")

# --- Configuration 3 : L2 + Dropout ---
print("\nEntraînement Config 3 : L2 + Dropout...")
model_l2_drop = build_pima_regularized(l2_lambda=0.01, use_dropout=True)
history_l2_drop = model_l2_drop.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2,
    callbacks=[early_stopping], verbose=0
)
stop_epoch_drop = len(history_l2_drop.history['val_loss'])
max_acc_drop = max(history_l2_drop.history['val_accuracy'])
print(f"-> Arrêt à l'epoch {stop_epoch_drop} | Max val_accuracy : {max_acc_drop:.4f}")

# ==========================================
# 4. VISUALISATION ET SAUVEGARDE
# ==========================================
print("\n--- Génération du graphique comparatif ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Comparatif de la Val_Loss selon la Régularisation (Pima)', fontsize=16)

configs = [
    ("Baseline (sans régul)", history_baseline, stop_epoch_base),
    ("Régularisation L2 (0.01)", history_l2, stop_epoch_l2),
    ("L2 + Dropout (0.3)", history_l2_drop, stop_epoch_drop)
]

for i, (title, hist, stop_ep) in enumerate(configs):
    ax = axes[i]
    ax.plot(hist.history['loss'], label='Train Loss', color='blue', alpha=0.6)
    ax.plot(hist.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    # Ligne verticale pour marquer l'arrêt de l'Early Stopping (-15 epochs de patience = vrai meilleur epoch)
    best_epoch = stop_ep - 15 if stop_ep > 15 else stop_ep
    ax.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Binary Crossentropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase5_pima_3configs.png', dpi=300)
print("Graphique sauvegardé sous 'phase5_pima_3configs.png'")