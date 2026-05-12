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

### Phase 2: Baseline Regression

Test Edge Case : batch_size=1 (Stochastic Gradient Descent) vs batch_size=len(X) (Batch Gradient Descent)

La convergence en epochs : Le batch_size=1 (SGD pur) converge en beaucoup moins d'epochs (souvent en 5 à 10 epochs, la loss est déjà très basse). À l'inverse, avec tout le dataset d'un coup, il faut des centaines voire des milliers d'epochs pour descendre.

Le paradoxe du temps : Bien que le batch_size=1 demande moins d'epochs, chaque epoch est atrocement lente car les poids sont mis à jour 13 209 fois par epoch au lieu de 412 fois. Les courbes de validation sont aussi très "hachées" et bruitées. Le batch_size=32 est le compromis idéal (Mini-batch GD) : il combine la rapidité d'exécution matricielle et la stochasticité nécessaire pour échapper aux minimums locaux.

Test Adversarial : Entraînement sans normalisation (X_train brut)

L'explosion de la Loss : Si vous passez les données brutes, la MSE initiale commence à des niveaux astronomiques (parfois des millions) ou donne carrément des NaN (Not a Number). Les différences d'échelles (le revenu est entre 1 et 10, la population entre 100 et 30000) créent des gradients totalement déséquilibrés. L'optimiseur fait des bonds géants et diverge.

Adam vs SGD (lr=0.001) : Sur les données normalisées, Adam converge beaucoup plus vite que le SGD classique. Adam adapte un taux d'apprentissage spécifique pour chaque paramètre (avec de l'inertie/momentum), ce qui lui permet de dévaler la pente de la fonction de coût beaucoup plus efficacement qu'un SGD naïf.

Epoch 90/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2316 - mae: 0.3308 - val_loss: 0.2831 - val_mae: 0.3668
Epoch 91/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2313 - mae: 0.3299 - val_loss: 0.2868 - val_mae: 0.3549
Epoch 92/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2310 - mae: 0.3307 - val_loss: 0.2741 - val_mae: 0.3571
Epoch 93/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2318 - mae: 0.3307 - val_loss: 0.2846 - val_mae: 0.3581
Epoch 94/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2334 - mae: 0.3310 - val_loss: 0.2926 - val_mae: 0.3602
Epoch 95/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2323 - mae: 0.3292 - val_loss: 0.2793 - val_mae: 0.3528
Epoch 96/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2321 - mae: 0.3306 - val_loss: 0.2812 - val_mae: 0.3570
Epoch 97/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2291 - mae: 0.3284 - val_loss: 0.2848 - val_mae: 0.3650
Epoch 98/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2307 - mae: 0.3288 - val_loss: 0.2809 - val_mae: 0.3650
Epoch 99/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2293 - mae: 0.3277 - val_loss: 0.3431 - val_mae: 0.3690
Epoch 100/100
413/413 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.2301 - mae: 0.3287 - val_loss: 0.2839 - val_mae: 0.3571

--- Évaluation sur le set de test ---
MAE test : 0.3517 (en centaines de milliers de $)
Erreur moyenne estimée : 35174 $