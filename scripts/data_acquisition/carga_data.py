#Importamos la librerias necesarias.
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Cargamos los datos.
# Los datos fueron descargados desde el repositorio indicado en el README.md y compartidos desde Drive.
# El enlace indicado abajo es el de descarga.
url = 'https://drive.google.com/uc?export=download&id=1wQL-Ip9G10kzQ4JssOjDjpB2K2fqNiAs'
data_edu = pd.read_csv(url)

#Obtenemos la información general del dataFrame.
data_edu.info()

#Visualizamos los 5 primeros registros.
data_edu.head()
