import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES (Phase 7)
# ==========================================
print("--- Préparation des données Wine Quality ---")
wine_url = "./dataset/phase7/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')

def map_quality(q):
    if q <= 4: return 0
    elif q <= 6: return 1
    else: return 2

df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)
X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==========================================
# DÉBATS SUR L'ORDRE D'EMPILAGE (BN vs Activation)
# ==========================================
# Ordre choisi par défaut : Dense Linéaire -> BatchNormalization -> Activation (ReLU)
# Justification : C'est la recommandation originale du papier de Ioffe & Szegedy (2015). 
# En normalisant les valeurs avant qu'elles ne passent dans le ReLU, on s'assure qu'environ 
# 50% des valeurs sont positives et 50% négatives (puisque la moyenne est de 0). 
# Cela maximise la zone active du ReLU et évite le problème des "Dead ReLUs" (neurones 
# qui ne s'activent jamais car ils reçoivent toujours des valeurs négatives).
# Note : Des papiers plus récents (comme He et al. sur ResNet V2) montrent que 
# Dense -> ReLU -> BN peut aussi bien fonctionner, d'où l'intérêt de tester les deux.

# ==========================================
# 2. FONCTION DE CONSTRUCTION DU MODÈLE
# ==========================================
def build_wine_model(use_batchnorm=False, bn_before_activation=True, extra_layer=False):
    """
    PMC multiclass Wine Quality paramétrable.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(11,)))
    
    # --- Première Couche (64 unités) ---
    if use_batchnorm and bn_before_activation:
        model.add(layers.Dense(64)) # Linéaire (sans activation)
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
    else:
        model.add(layers.Dense(64, activation='relu'))
        if use_batchnorm and not bn_before_activation:
            model.add(layers.BatchNormalization())
            
    # --- Seconde Couche (32 unités) + Régularisation L2 ---
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    
    # --- Couche Supplémentaire (16 unités) optionnelle ---
    if extra_layer:
        model.add(layers.Dense(16, activation='relu'))
        
    # --- Couche de Sortie ---
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. BOUCLE DE COMPARAISON DES CONFIGURATIONS
# ==========================================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

configs = {
    'sans_bn': lambda: build_wine_model(use_batchnorm=False),
    'bn_avant_activation': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=True),
    'bn_apres_activation': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=False),
    'bn_extra_couche': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=True, extra_layer=True),
}

results = {}

print("\n--- Démarrage de l'entraînement comparatif ---")
for name, build_fn in configs.items():
    print(f"Entraînement de la configuration : {name} ...")
    
    # Création du dossier horodaté pour TensorBoard
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/wine/{name}_{timestamp}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Génération d'un modèle frais pour éviter le transfert de poids d'une config à l'autre
    model = build_fn()
    
    history = model.fit(
        X_train_norm, y_train,
        epochs=200,
        validation_split=0.2,
        callbacks=[early_stop, tb_callback],
        verbose=0 # Silencieux
    )
    
    # Extraction des métriques
    max_val_acc = max(history.history['val_accuracy'])
    stop_epoch = len(history.history['val_loss'])
    
    # Stockage et affichage
    results[name] = {'val_accuracy': max_val_acc, 'epochs': stop_epoch}
    print(f" -> {name}: val_accuracy={max_val_acc:.4f}, stopped at epoch {stop_epoch}")

print("\nTerminé. Ouvrez TensorBoard avec : tensorboard --logdir=logs/wine")