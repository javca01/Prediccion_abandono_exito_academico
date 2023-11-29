import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


#Cargamos el conjunto de datos.
url = 'https://drive.google.com/uc?export=download&id=1wQL-Ip9G10kzQ4JssOjDjpB2K2fqNiAs'
data_edu = pd.read_csv(url)

# Determinamos si existen datos faltante o de la mala calidad en el dataset.
reg_vacios = data_edu.isna().sum()#El método sum permite contar cuántos elementos cumplen la condición de isna, es decir, cuantas celdas vacías hay en cada columna.
reg_vacios[reg_vacios!=0]

# Identificamos si hay datos ilegibles
dat_cat =  data_edu.isna()
flag_emp = False
for i in dat_cat.columns:
  if dat_cat[i].value_counts()[0] != data_edu.shape[0]:
    print("La siguiente columna tiene elementos vacíos o nan: "+i)
    flag_emp = True

if flag_emp == False:
  print("Ninguna de las columnas tiene elementos vacíos o nan.")
  

# Configuración del estilo de Seaborn
sns.set(style="whitegrid")

# Crear el histograma usando Seaborn
plt.figure(figsize=(8, 6))
sns.countplot(x='Target', data=data_edu, palette="pastel")

# Añadir etiquetas y título
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Histograma de la Columna "Target"')

# Mostrar el histograma
plt.show()

# Obtenemos las estadísticas de los 8 primeros campos. 
data_edu.iloc[:,:8].describe()

# Obtenemos las estadísticas de los campos del 9 al 16. 
data_edu.iloc[:,8:16].describe()

# Obtenemos las estadísticas de los campos del 17 al 24. 
data_edu.iloc[:,16:24].describe()

# Obtenemos las estadísticas de los campos del 25 al 32. 
data_edu.iloc[:,24:32].describe()

# Obtenemos las estadísticas de los campos del 33 al 35. 
data_edu.iloc[:,32:-1].describe()

# Obtenemos la cantiadad de valores presentes por cada una de las etiquetas.
data_edu.iloc[:,-1].value_counts()

# Obtenemos la matriz de correlación del conjunto de datos.
import pandas as pd


# Obtener la matriz de correlación
matriz_correlacion = data_edu.corr()

# Imprimir la matriz de correlación
print(matriz_correlacion)

# Obtener la matriz de correlación
matriz_correlacion = data_edu.corr()

# Crear el mapa de calor con Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=False, cmap='coolwarm', vmin=-1, vmax=1)

# Añadir título
plt.title('Mapa de Calor - Matriz de Correlación')

# Mostrar el mapa de calor
plt.show()

# Obtenemos la matriz de dispersión del conjunto de datos.
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Obtener la matriz de dispersión
matriz_dispersión = scatter_matrix(data_edu, alpha=0.8, figsize=(8, 8), diagonal='hist')

plt.tight_layout()
# Ajustar diseño y mostrar la matriz de dispersión
plt.show()
# Guardamos la imagen de la matriz de dispersión.
plt.savefig('matriz_dispersion.png')