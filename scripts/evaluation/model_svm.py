#Importamos la librerias necesarias.
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

url = 'https://drive.google.com/uc?export=download&id=1wQL-Ip9G10kzQ4JssOjDjpB2K2fqNiAs'
data_edu = pd.read_csv(url)

# Obtener la matriz de correlación
matriz_correlacion = data_edu.corr()

# Imprimir la matriz de correlación
print(matriz_correlacion)


# Obtenemos una mejor representación de la matriz de correlación.

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

# Obtener la matriz de dispersión
matriz_dispersión = scatter_matrix(data_edu, alpha=0.8, figsize=(8, 8), diagonal='hist')

plt.tight_layout()
# Ajustar diseño y mostrar la matriz de dispersión
plt.show()
plt.savefig('matriz_dispersion.png')

# Descargamos la imagen.
plt.savefig('matriz_dispersion.png')

# Modelado
data_edu.iloc[:,-1].value_counts()

from sklearn.preprocessing import OrdinalEncoder
#Copiamos los datos.
data_edu_new = data_edu.copy()
# Mapear las etiquetas a números
mapeo = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
data_edu_new['Target'] = data_edu_new['Target'].map(mapeo)
data_edu_new.head()

# Hacemos recounteo posterior a la transformación.
data_edu_new.iloc[:,-1].value_counts()

# Separamos los datos de entreda y salida.
X= data_edu_new.iloc[:,:-1].values
y = data_edu_new.iloc[:,-1].values

# Mostramos el tamaño de cada subconjunto
print("Tamaño de X: "+str(X.shape))
print("Tamaño de y: "+str(len(y)))

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Codificamos las etiquetas de salida
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=3)

#Construimos un modelo SVM con Kernel Gaussiano.
#Clasificador de vectores de soporte general.
from sklearn.svm import SVC

rbf_svm = SVC(kernel='rbf',   # Kernel de tipo RBF
              gamma = 0.001)  # Valor del argumento gamma

#Entrenamos el modelo.
rbf_svm.fit(X_train, y_train)

# Determinamos el error de entrenamiento y prueba.
print(f"Error en entrenamiento:\t {1-rbf_svm.score(X_train, y_train):.4f}")
print(f"Error en prueba:\t {1-rbf_svm.score(X_test, y_test):.4f}")

# Aplicamos validación cruzada con GridSearch.

# Los hiperparámetros deben estar en forma de diccionario.
param_grid = {'C':     [2**i for i in range(-5, 7, 1)],
              'gamma': [2**i for i in range(-5, 7, 1)]}

param_grid

#Imprimimos los valores de C.
print(param_grid['C'])

#Imprimimos los valores de gamma.
print(param_grid['gamma'])

# Realizamos búsqueda en cuadrícula de hiperparámetros.
from sklearn.model_selection import GridSearchCV

grid_clf = GridSearchCV(SVC(kernel='rbf'),
                   param_grid=param_grid,
                   verbose=1,
                   return_train_score=True
                   )

grid_clf.fit(X_train, y_train)

#Obtenemos la lista de resultados por elemento en la malla de parámetros
cv_results = pd.DataFrame(grid_clf.cv_results_)
cv_results

# Para encontrar las mejores configuraciones, obtenemos la tabla con los n mayores resultados con pandas.
# Método nlargest de pandas, con los n primeros valores por mean_test_score.
n = 10
cv_results.nlargest(n, 'mean_test_score')

# También podemos consultar los mejores parámetros identificados así.
print(grid_clf.best_params_)

# Puntaje de la mejor combinación de parámetros.
print(grid_clf.best_score_)

# Al haber entrenado el modelo con validación cruzada, GridSearchCV
# elige automáticamente la mejor configuración y vuelve a entrenar un modelo
# en todo el conjunto de datos de entrenamiento, de esta manera, al realizar
# el entrenamiento con fit, se pueden llamar a las funciones predict()
# y score() directamente desde el objeto de grid search.

#Reportamos sobre el conjunto de prueba.

grid_clf.score(X_test, y_test)

# Exploramos gráficamente los resultados obtenidos en todas las configuraciones.
# El número de filas del DataFrame cv_results corresponde al número de
# configuraciones de hiperparámetros que se están explorando.
len(cv_results), len(param_grid['C']) * len(param_grid['gamma'])

# Empleando la columna mean_test_score, extraemos los valores de precisión o accuracy promedio
# para organizarlos en una matriz cuyas filas son los valores del parámetro C y
# las columnas los valores del parámetro gamma.
#scores_df = cv_results.pivot(index = 'param_C',
#                            columns = 'param_gamma',
#                             values = 'mean_test_score')
scores_df

# Visualizamos la exploración de la malla de hiperparámetros a través de un mapa de calor
# empleando el método heatmap de Seaborn.
sns.heatmap(scores_df, cmap = 'inferno').set_title('Accuracy en validación');

# DESPLEGAMOS EL MODELO

#Creamos un experimento en MLFLow para el conjunto de datos.
exp_id = mlflow.create_experiment(name="svc_model", artifact_location="mlruns/") #ajustar los nombres a mi experimento
print(exp_id)


run = mlflow.start_run(experiment_id=exp_id, run_name="svm")
#model = SVC(**params).fit(features, labels)
y_pred_svm = rbf_svm.predict(X_test)

params = {"kernel": "rbf", "C": 2.0, "gamma": 0.03125}
cm = confusion_matrix(y_test, y_pred_svm)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
fig.show()
fig.savefig("confusion_matrix_svm.png")

mlflow.log_params(params)
mlflow.sklearn.log_model(rbf_svm, "model_svm")
mlflow.log_artifact("confusion_matrix_svm.png", "confusion_matrix")
mlflow.log_metrics({
    "accuracy": accuracy_score(y_test, y_pred_svm),
    "f1": f1_score(y_test, y_pred_svm, average = "micro" ),
    "precision": precision_score(y_test, y_pred_svm, average = "micro"),
    "recall": recall_score(y_test, y_pred_svm, average = "micro")
    })
mlflow.end_run()

# Finalizamos el run
mlflow.end_run()