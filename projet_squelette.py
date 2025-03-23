# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:37:36 2022
Edited on Sun Mar 23 2025 23:00:00 2025

@author: JDION, Basile LE THIEC
"""

### MINI-PROJET

import numpy as np
import matplotlib.pyplot as plt

# fonction de lecture des données
def read_data():
    crime_data = np.genfromtxt('crime.csv', delimiter=',')
    return crime_data

# stockage des données dans la variable crime_data
crime_data = read_data()

"""
II. Q1a
"""
# Etant données X et y, la fonction np.linalg.lstsq(X, y, rcond=None) renvoie comme deux premières valeurs : la solution theta, puis l'erreur commise

# Extraction des variables
crime = crime_data[:, 0]    # taux global de criminalité
violent = crime_data[:, 1]  # taux de criminalité violente
funding = crime_data[:, 2]  # financement annuel de la police
hs = crime_data[:, 3]       # pourcentage avec 4 années d'études secondaires
not_hs = crime_data[:, 4]   # pourcentage sans diplôme d'études secondaires
college = crime_data[:, 5]  # pourcentage dans l'enseignement supérieur
college4 = crime_data[:, 6] # pourcentage avec 4 années d'études supérieures

# Préparation des données pour la régression
# ajout d'une colonne de 1 pour le terme constant
X = np.column_stack((np.ones(len(crime)), funding, hs, not_hs, college, college4))
y = crime  # variable à prédire

# Calcul de la solution des moindres carrés
theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

print("Coefficients de régression (theta):")
print(f"Constante: {theta[0]:.4f}")
print(f"funding: {theta[1]:.4f}")
print(f"hs: {theta[2]:.4f}")
print(f"not-hs: {theta[3]:.4f}")
print(f"college: {theta[4]:.4f}")
print(f"college4: {theta[5]:.4f}")

"""
II. Q1b
"""
# Calcul des prédictions du modèle
y_pred = X @ theta

# Calcul de l'erreur quadratique moyenne (MSE)
mse = np.mean((y - y_pred) ** 2)
print(f"Erreur quadratique moyenne (MSE): {mse:.4f}")

# Calcul du coefficient de détermination (R²)
y_mean = np.mean(y)
ss_total = np.sum((y - y_mean) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"Coefficient de détermination (R²): {r_squared:.4f}")

"""
II. Q1c
"""
# Visualisation des résultats

# 1. Valeurs prédites vs valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Taux de criminalité réel')
plt.ylabel('Taux de criminalité prédit')
plt.title('Valeurs prédites vs Valeurs réelles')
plt.grid(True)
plt.show()

# 2. Résidus
residus = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residus, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Graphique des résidus')
plt.grid(True)
plt.show()

# 3. Visualisation de l'impact de chaque variable explicative
plt.figure(figsize=(12, 10))

variables = ['funding', 'hs', 'not_hs', 'college', 'college4']
data_vars = [funding, hs, not_hs, college, college4]

for i, (var_name, var_data) in enumerate(zip(variables, data_vars), 1):
    plt.subplot(2, 3, i)
    plt.scatter(var_data, crime, alpha=0.7)
    plt.xlabel(var_name)
    plt.ylabel('Crime rate')
    plt.title(f'Crime rate vs {var_name}')
    plt.grid(True)

plt.tight_layout()
plt.show()
