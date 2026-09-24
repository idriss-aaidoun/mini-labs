# TP 1 — Régression Linéaire & Normal Equation

Petit guide pour comprendre et compléter le notebook `TP_1_-_Linear_Regression__Normal_Equation.ipynb`.

## 🎯 Objectif du TP

Implémenter la **normal equation** (solution fermée / closed-form) pour trouver les paramètres optimaux d'une régression linéaire, au sens de l'erreur des moindres carrés (MSE).

## 📐 Rappels théoriques

### La fonction de perte

On cherche à minimiser l'erreur entre les prédictions $X\theta$ et les vraies valeurs $y$ :

$$\hat\theta = \arg\min_\theta \|X\theta - y\|_2^2$$

C'est la somme des carrés des erreurs (SSE / MSE) : plus $X\theta$ s'éloigne de $y$, plus la perte est grande.

### La normal equation

Comme cette perte est convexe, on peut annuler son gradient pour trouver directement le minimum, **sans itérer** (contrairement à la descente de gradient) :

$$\hat\theta = (X^TX)^{-1}X^Ty$$

## 🗂️ Structure des données

`X` est la **matrice de design** : chaque **ligne** est un exemple, chaque **colonne** est une feature.

| Exemple | $x_1$ | $x_2$ | $y$ (cible) |
|---|---|---|---|
| 1 | 1.0 | 2.0 | 0.75 |
| 2 | 10.0 | 8.0 | 9.75 |
| 3 | 12.0 | 6.0 | 9.25 |
| 4 | 5.0 | 4.0 | 4.25 |

## ⚠️ Point clé : ne pas oublier le biais

Le modèle complet est :

$$h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$$

Le terme $\theta_0$ (biais / intercept) n'est multiplié par aucune vraie feature — pour l'intégrer dans le produit matriciel $X\theta$, il faut ajouter une **colonne de 1** devant $X$ :

| Exemple | biais ($x_0$) | $x_1$ | $x_2$ |
|---|---|---|---|
| 1 | **1** | 1.0 | 2.0 |
| 2 | **1** | 10.0 | 8.0 |
| ... | **1** | ... | ... |

```python
X_b = np.hstack([np.ones((X.shape[0], 1)), X])
```

Sans cette étape, `theta` n'aura que 2 composantes (au lieu de 3) et le code de visualisation du plan (qui utilise `theta[0]`, `theta[1]`, `theta[2]`) plantera.

## ✅ Solution de l'exercice

```python
import numpy as np

X = np.array([[1.0, 2.0], [10.0, 8.0], [12.0, 6.0], [5.0, 4.0], [1.0, 2.0]])
y = np.array([0.75, 9.75, 9.25, 4.25, 1.75])

# 1) ajout de la colonne de biais
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# 2) normal equation
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

**Résultat obtenu :**

| Paramètre | Valeur |
|---|---|
| $\theta_0$ (biais) | ≈ -0.726 |
| $\theta_1$ | ≈ 0.468 |
| $\theta_2$ | ≈ 0.718 |

**Prédiction pour le point $(8, 15)$ :**

$$\hat y = \theta_0 + \theta_1 \cdot 8 + \theta_2 \cdot 15 \approx 13.79$$

> 💡 Note : le point $(1, 2)$ apparaît deux fois avec des $y$ différents (0.75 et 1.75) — c'est volontaire, pour rendre le système sur-déterminé (pas de solution exacte possible), ce qui justifie l'usage des moindres carrés plutôt qu'une inversion directe.

## 🚀 Utilisation

1. Ouvrir le notebook dans Jupyter.
2. Exécuter les cellules de données et de visualisation 3D.
3. Compléter la cellule `theta = ....` avec le code ci-dessus.
4. Exécuter la cellule de visualisation finale pour voir le plan de régression tracé sur le nuage de points.

## 📚 Fonctions numpy utilisées

| Fonction | Rôle |
|---|---|
| `np.array` | Créer une matrice / un vecteur |
| `np.hstack` | Coller des colonnes côte à côte |
| `np.ones` | Créer une colonne/matrice de 1 |
| `.T` | Transposée d'une matrice |
| `@` | Produit matriciel |
| `np.linalg.inv` | Inverse d'une matrice |
