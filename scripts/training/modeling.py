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
  print("La predicción es incorrecta."# Predecimos en función de un solo registro.
i = 98 # Registro a evaluar.
datar = data_edu_new.iloc[i,:-1].values
datar = datar.reshape(1,-1)
yt = model.predict(datar)
escala_pred = np.argmax(yt)+1 
etiq_real = data_edu_new.iloc[i,-1]
print("La predicción es: "+str(escala_pred)+", el valor real es: "+str(data_edu_new.iloc[i,-1]))
if escala_pred == etiq_real :
  print("La predicción es correcta.")
else:
  print("La predicción es incorrecta."
        
# Predecimos sobre el conjunto de datos de prueba.
y_pred1 = model.predict(x=X_test)
print(y_pred1)
y_pred_int1 = np.argmax(y_pred1, axis=1)
print(y_pred_int1)

#Generamos el clasification report
from sklearn.metrics import classification_report
#print(classification_report(y_test_encoded, y_pred1))
print(classification_report(y_test, y_pred_int1))
