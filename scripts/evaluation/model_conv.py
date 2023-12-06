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

# Creamos la red neuronal convolucional
model = tf.keras.models.Sequential()
# Capa de entrada
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],1)))
# Capa de convolución
model.add(tf.keras.layers.Conv1D(32, 2, activation="relu"))
# Capa de MaxPooling1D
model.add(tf.keras.layers.MaxPooling1D(2, strides=2))
# Capa Flatten
model.add(tf.keras.layers.Flatten())
# Capa Dense
model.add(tf.keras.layers.Dense(3, activation="softmax"))
model.summary()

#Compilamos el modelo
model.compile(loss="binary_crossentropy",
#model.compile(loss="categorical_crossentropy", #Dado que es un problema multiclase.
                 optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

tf.keras.utils.plot_model(model,show_shapes=True)

#Defimos los respectivos Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
                  filepath='best_weights.h5', # Path del archivo donde se guardarán los pesos o el modelo.
                  monitor="val_accuracy",              # La métrica que se va a monitorear.
                  mode="max",                 # Se quiere guardar el modelo que reporte el accuracy máximo: max.
                  save_best_only=True,        # Si se define True, entonces solo se guarda el mejor modelo.
                  save_weights_only=True      # Si se define True, solo se guardan los pesos, no la arquitectura.
              )

stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",            # La métrica que se va a monitorear.
                patience=50,              # Si después de 50 epochs la métrica no mejora, se detiene el entrenamiento.
                mode="max",               # Se quiere guardar el modelo que reporte el accuracy máximo: max.
                restore_best_weights=True # Si True, automaticamente se cargan al modelo los mejores pesos.
            )

#Entrenamos el modelo
history = model.fit(X_train, y_train_encoded, epochs=20, validation_data=(X_test, y_test_encoded), callbacks=[checkpoint, stopping])

# Generamos las gráficas de error vs epochs y de accuracy vs epochs
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Error vs epochs")
plt.xlabel("epochs")
plt.ylabel("error")
plt.legend(["loss", "val_loss"])
plt.show()
plt.grid()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy vs epochs")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["accuracy", "val_accuracy"])
plt.grid()
plt.show()

# Predecimos en función de un solo registro.
i = 98 # Registro a evaluar.
datar = data_edu_new.iloc[i,:-1].values
datar = datar.reshape(1,-1)
yt = model.predict(datar)
escala_pred = np.argmax(yt)+1 # recordemos que habiamos escalado los valores de 0 a 4, luego, para volver a las clases
# reales debemos sumar 1, con la función argmax, encontramos el indice donde se encuentra el valor máximo, pues el areglo
# que devuelve la predicción es un vector con las probabilidades de pertenencia de la entrada a cada una de las clases.
etiq_real = data_edu_new.iloc[i,-1]
print("La predicción es: "+str(escala_pred)+", el valor real es: "+str(data_edu_new.iloc[i,-1]))
if escala_pred == etiq_real :
  print("La predicción es correcta.")
else:
  print("La predicción es incorrecta.")

#Podemos comparar con el head extraido en la celda anterior.
#La variable i representa el registro pasado como entrada.

# Predecimos sobre el conjunto de datos de prueba.
y_pred1 = model.predict(x=X_test)
print(y_pred1)
y_pred_int1 = np.argmax(y_pred1, axis=1)
print(y_pred_int1)

#Generamos el clasification report
from sklearn.metrics import classification_report
#print(classification_report(y_test_encoded, y_pred1))
print(classification_report(y_test, y_pred_int1))

# DESPLIEGUE

#Obtenemos las métricas de desempeño del modelo.
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score



# Calcular precision score
precision = precision_score(y_test, y_pred_int1, average = "micro")
print(f'Precision Score: {precision}')

# Calcular f1_score
f1 = f1_score(y_test, y_pred_int1,average = "micro")
print(f'F1 Score: {f1}')

# Calcular accuracy score
accuracy = accuracy_score(y_test, y_pred_int1)
print(f'Accuracy Score: {accuracy}')

# SUBIMOS EL MODELOS
!cat /etc/os-release

# Instalamos MLFlow
!pip install mlflow==2.1.0

#Instalamos herramientas auxiliares.
!apt install tree

# Validamos la instalación de la herramienta
!mlflow --version

# Importamos la librerias que requerimos.
import os
import mlflow
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display

#Creamos una carpeta donde se guardarán todos los datos de MLFlow
!mkdir mlruns

# Lanzamos el servidor de MLFlow empleando una DB llamada tracking.db y definimos que los archivos sean guardados en el directorio mlruns
command = """
mlflow server \
        --backend-store-uri sqlite:///tracking.db \
        --default-artifact-root file:mlruns \
        -p 5000 &
"""
#Personal:la instrucción -p 5000 & indica que se define el puerto 5000 en el lanzamiento de mlflow.
get_ipython().system_raw(command)

# Instalamos ngrok
!pip install pyngrok

# Colocamos el token de autenticación
token = "INGRESE TOKEN" # Agregue el token dentro de las comillas
# Token copiado desde la cuenta infotecno de ngrok.
os.environ["NGROK_TOKEN"] = token

# Nos autenticamos en ngrok
!ngrok authtoken $NGROK_TOKEN

# Lanzamos la conexión con ngrok
from pyngrok import ngrok
ngrok.connect(5000, "http")
#accedemos al servicio a través de la URL que se genera abajo.

#Especificamos que MLFlow debe usar el servidor antes referenciado.
mlflow.set_tracking_uri("http://localhost:5000")

#DEFINIMOS EL MODELO EN MLFLOW

#Creamos un experimento en MLFLow para el conjunto de datos.
exp_id = mlflow.create_experiment(name="conv_network", artifact_location="mlruns/") #ajustar los nombres a mi experimento
print(exp_id)

# Definimos que el entrenamiento del modelo sea dentro de una run de MLFlow
run = mlflow.start_run(
    experiment_id = exp_id,
    run_name="default_logistic"
    )
print(run)

# Registramos las métricas de desempeño dentro de nuestro experimento de MLFlow
mlflow.log_metrics({
    "accuracy": accuracy_score(y_test, y_pred_int1),
    "f1": f1_score(y_test, y_pred_int1, average="micro"),
    "precision": precision_score(y_test, y_pred_int1, average ="micro")
    #"recall": recall_score(y_test, y_pred_int1)
    }) 

# Almacenamos el modelo dentro de MLFlow
mlflow.sklearn.log_model(history, "model")

# Creamos una matriz de confusión del modelo y lo la almacenamos en MLFLow
from sklearn.metrics import confusion_matrix
import seaborn as sns
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred_int1)
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
fig.show()
fig.savefig("confusion_matrix.png") 

# Almacenamos la imagen dentro de MLFlow
mlflow.log_artifact("confusion_matrix.png", "confusion_matrix")

# Terminamos la ejecución del run
mlflow.end_run()
