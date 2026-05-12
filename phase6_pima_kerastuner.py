import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# RÉPONSE À LA QUESTION : LE PIÈGE DU TUNER
# ==========================================
# Stratégie de protection contre un max_epochs trop bas :
# Si on fixe max_epochs=10, le tuner va privilégier des architectures avec un learning rate 
# très agressif (ex: 0.01) ou sans dropout, car ce sont elles qui font baisser la loss le plus 
# vite dans les premiers instants. Mais ces architectures vont très vite overfitter.
# Pour se protéger, on fixe un max_epochs élevé (ici 100) couplé à un EarlyStopping. 
# Ainsi, chaque modèle a le temps de révéler son vrai potentiel (convergence lente mais stable 
# vs convergence rapide mais instable).
# Pour valider un essai, on s'assure que la patience de l'EarlyStopping est suffisante 
# (patience=10) pour passer les micro-plateaux. On sait qu'on a assez tourné si les 5 
# meilleurs trials ont des métriques très proches et partagent des invariants (ex: toujours 
# le même learning rate ou la même activation).

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==========================================
# 2. DÉFINITION DE L'HYPERMODEL
# ==========================================
def build_pima_model(hp):
    """
    HyperModel Pima : définit l'espace de recherche pour keras-tuner.
    """
    model = keras.Sequential()
    
    # Définition des espaces de recherche
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    
    # Construction de l'architecture dynamique
    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
        
    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

# ==========================================
# 3. CONFIGURATION ET LANCEMENT DU TUNER
# ==========================================
# Instanciation du RandomSearch
tuner = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=42,
    directory='tuning_pima',
    project_name='pima_random'
)

print("\n--- Résumé de l'espace de recherche ---")
tuner.search_space_summary()

# Early stopping indispensable pour le tuner
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

print("\n--- Lancement de la recherche d'hyperparamètres (15 trials) ---")
# On laisse epochs=100 pour donner une chance aux petits learning rates de converger
tuner.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0 # Silencieux pour ne pas polluer la console avec 15 x 100 epochs
)

# ==========================================
# 4. ANALYSE DES RÉSULTATS
# ==========================================
print("\n--- Analyse des meilleurs résultats ---")
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Meilleur learning_rate :", best_hp.get('learning_rate'))
print("Meilleures units_1     :", best_hp.get('units_1'))
print("Meilleures units_2     :", best_hp.get('units_2'))
print("Meilleure activation   :", best_hp.get('activation'))
print("Meilleur dropout_rate  :", best_hp.get('dropout_rate'))

print("\n--- Top 5 Trials ---")
tuner.results_summary(num_trials=5)

print("\n--- Valeurs des Hyperparamètres du Top 5 ---")
# On itère pour chercher les invariants (le signal fort)
top_hps = tuner.get_best_hyperparameters(num_trials=5)
for i, hp in enumerate(top_hps):
    print(f"Rang {i+1} : {hp.values}")

# ==========================================
# 5. ENTRAÎNEMENT DU MEILLEUR MODÈLE
# ==========================================
print("\n--- Entraînement final avec la meilleure architecture ---")
best_model = tuner.hypermodel.build(best_hp)

# On ré-entraîne le meilleur modèle avec une limite confortable (200 epochs)
history_best = best_model.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

final_val_acc = max(history_best.history['val_accuracy'])
print(f"\n✅ Val_accuracy finale de la best_config : {final_val_acc:.4f}")