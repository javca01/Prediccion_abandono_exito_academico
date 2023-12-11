import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

url = 'https://drive.google.com/uc?export=download&id=1wQL-Ip9G10kzQ4JssOjDjpB2K2fqNiAs'
data_edu = pd.read_csv(url)

from sklearn.preprocessing import OrdinalEncoder
#Copiamos los datos.
data_edu_new = data_edu.copy()
# Mapear las etiquetas a números
mapeo = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
data_edu_new['Target'] = data_edu_new['Target'].map(mapeo)
data_edu_new.head()

# Separamos los datos de entreda y salida.
X= data_edu_new.iloc[:,:-1].values
y = data_edu_new.iloc[:,-1].values

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Codificamos las etiquetas de salida
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Creamos la red neuronal convolucional
model1 = tf.keras.models.Sequential()
# Capa de entrada
model1.add(tf.keras.layers.Input(shape=(X_train.shape[1],1)))
# Capa de convolución
model1.add(tf.keras.layers.Conv1D(32, 2, activation="relu"))
# Capa de MaxPooling1D
model1.add(tf.keras.layers.MaxPooling1D(2, strides=2))
# Capa Flatten
model1.add(tf.keras.layers.Flatten())
# Capa Dense
model1.add(tf.keras.layers.Dense(3, activation="softmax"))
model1.summary()

#Compilamos el modelo
model1.compile(loss="categorical_crossentropy", #Dado que es un problema multiclase.
                 optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

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
history1 = model1.fit(X_train, y_train_encoded, epochs=20, validation_data=(X_test, y_test_encoded), callbacks=[checkpoint, stopping])

# Generamos las gráficas de error vs epochs y de accuracy vs epochs
plt.plot(history1.history["loss"])
plt.plot(history1.history["val_loss"])
plt.title("Error vs epochs")
plt.xlabel("epochs")
plt.ylabel("error")
plt.legend(["loss", "val_loss"])
plt.show()
plt.grid()

plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
plt.title("Accuracy vs epochs")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["accuracy", "val_accuracy"])
plt.grid()
plt.show()

# Predecimos sobre el conjunto de datos de prueba.
y_pred1b = model1.predict(x=X_test)
print(y_pred1b)
y_pred_int1b = np.argmax(y_pred1b, axis=1)
print(y_pred_int1b)

#Generamos el clasification report
from sklearn.metrics import classification_report
#print(classification_report(y_test_encoded, y_pred1b))
print(classification_report(y_test, y_pred_int1b))

#Obtenemos las métricas de desempeño del modelo.
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

# Calcular precision score

print("Las métricas precision, f1_score y recall_score, son reportadas con average macro.")

precision = precision_score(y_test, y_pred_int1b, average = "macro")
print(f'Precision Score: {precision}')

# Calcular f1_score
f1 = f1_score(y_test, y_pred_int1b,average = "macro")
print(f'F1 Score: {f1}')

# Calcular accuracy score
accuracy = accuracy_score(y_test, y_pred_int1b)
print(f'Accuracy Score: {accuracy}')

# Calculamos el recall_score
recall = recall_score(y_test, y_pred_int1b, average = "macro")
print(f'Recall Score: {recall}')

#Creamos un experimento en MLFLow para el conjunto de datos.
exp_id = mlflow.create_experiment(name="conv_network_multiclass", artifact_location="mlruns/") 
print(exp_id)

# Definimos que el entrenamiento del modelo sea dentro de una run de MLFlow
run = mlflow.start_run(
    experiment_id = exp_id,
    run_name="conv_net_multic"
    )
print(run)

mlflow.log_metrics({
    "accuracy": accuracy_score(y_test, y_pred_int1),
    "f1": f1_score(y_test, y_pred_int1, average="micro"),
    "precision": precision_score(y_test, y_pred_int1, average ="micro"),
    "recall": recall_score(y_test, y_pred_int1, average = "micro")
    })

# Almacenamos el modelo dentro de MLFlow
mlflow.sklearn.log_model(history1, "model")

# Creamos una matriz de confusión del modelo y lo la almacenamos en MLFLow
from sklearn.metrics import confusion_matrix
import seaborn as sns
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred_int1b)
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
fig.show()
fig.savefig("confusion_matrix_b.png")

mlflow.log_artifact("confusion_matrix_b.png", "confusion_matrix")

# Terminamos la ejecución del run
mlflow.end_run()


data_edu["Target"].value_counts()