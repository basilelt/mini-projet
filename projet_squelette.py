# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:37:36 2022

@author: JDION
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


"""
II. Q1b
"""


"""
II. Q1c
"""

