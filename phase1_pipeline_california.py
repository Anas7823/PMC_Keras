import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# DÉCISION D'ARCHITECTURE DU PIPELINE
# ==========================================
# Ordre choisi : (b) split puis scaler.fit(X_train)
# Pourquoi : Il est crucial de fitter le scaler UNIQUEMENT sur les données d'entraînement. 
# Si on fitte sur X entier avant le split (ordre a), les statistiques de normalisation 
# (moyenne, écart-type) incluront des informations provenant des sets de validation et de test. 
# C'est ce qu'on appelle le "data leakage" (fuite de données). Le modèle serait alors 
# évalué de manière biaisée, car il aurait indirectement "vu" le set de test.

# 1. Charger le dataset
print("--- Chargement du dataset California Housing ---")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 2. Premier split : Train/Test (80% / 20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Second split : Train/Validation (80% / 20% du set temporaire)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 4. Instanciation et fit du StandardScaler sur X_train UNIQUEMENT
scaler = StandardScaler()
scaler.fit(X_train)

# 5. Transformation des trois sets
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# ==========================================
# VÉRIFICATIONS (Happy Path)
# ==========================================
print("\n--- Vérifications des Shapes ---")
print(f"X_train shape : {X_train_norm.shape}")
print(f"X_val shape   : {X_val_norm.shape}")
print(f"X_test shape  : {X_test_norm.shape}")

print("\n--- Vérifications des Stats Descriptives (X_train_norm) ---")
print(f"Moyenne par feature (doit être ~0) : \n{np.round(X_train_norm.mean(axis=0), 5)}")
print(f"Écart-type par feature (doit être ~1) : \n{np.round(X_train_norm.std(axis=0), 5)}")

print("\n--- Vérifications des Features ---")
print(f"Nombre de features : {len(housing.feature_names)}")
print(f"Noms des features : {housing.feature_names}")

# ==========================================
# TEST EDGE CASE : Démonstration du Data Leakage
# ==========================================
print("\n--- Test Edge Case : Data Leakage ---")
# Faux scaler fitté sur l'ensemble des données
bad_scaler = StandardScaler()
bad_scaler.fit(X) 
X_test_bad_norm = bad_scaler.transform(X_test)

print(f"Moyenne X_test avec un bon scaler (attendu : différent de 0) : \n{np.round(X_test_norm.mean(axis=0), 3)}")
print(f"Moyenne X_test avec mauvais scaler (data leakage, faussement proche de 0) : \n{np.round(X_test_bad_norm.mean(axis=0), 3)}")

# ==========================================
# TEST ADVERSARIAL : Valeurs aberrantes (Outliers)
# ==========================================
print("\n--- Test Adversarial : Comportement face aux outliers ---")
X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
X_extreme_norm = scaler.transform(X_extreme)

print(f"Valeur brute MedInc (Revenu Médian) : {X_extreme[0][0]}")
print(f"Valeur normalisée MedInc : {X_extreme_norm[0][0]:.2f}")