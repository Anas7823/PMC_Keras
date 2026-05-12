### Phase 1: Pipeline chargement et normalisation California Housing

Analyse des tests demandés

#### 1. Le Data Leakage (Edge Case)
L'écart de la moyenne sur le set de test met en évidence la fuite de données :

Avec le bon pipeline : La moyenne de X_test_norm n'est pas exactement 0 (elle tourne autour de 0.02 ou -0.01 selon les colonnes). C'est le comportement attendu. Les données de test sont nouvelles, elles ne suivent pas parfaitement la distribution exacte du set d'entraînement.

Si on avait fitté sur X entier : La moyenne de X_test_bad_norm se rapproche dangereusement du 0 parfait. Le scaler a "trichi" en incluant les valeurs de test dans son calcul d'origine.

#### 2. Comportement face aux Outliers (Test Adversarial)
En passant le tableau X_extreme dans le scaler :

Le StandardScaler applique la formule (valeur - moyenne) / ecart_type.

Pour une valeur de revenu (MedInc) de 99999 (qui est immense comparée à l'échelle d'origine en dizaines de milliers de dollars), la valeur normalisée va exploser (généralement autour de +52000 ou plus, selon l'écart-type de cette feature).

En production : Cela montre que le StandardScaler est extrêmement sensible aux outliers extrêmes. Un tel point de donnée va générer des activations colossales dans les neurones du modèle lors de l'inférence, ce qui va complètement fausser la prédiction (et potentiellement causer une explosion du gradient si cela arrive pendant l'entraînement). Si les données de production sont très bruitées, il faudrait plutôt envisager un RobustScaler (qui utilise la médiane et l'intervalle interquartile) pour protéger le modèle.

---

