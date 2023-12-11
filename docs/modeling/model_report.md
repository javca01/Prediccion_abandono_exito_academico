# Reporte del Modelo Final

## Resumen Ejecutivo

En este proyecto se tomó un conjunto de datos proveniente del repositorio de Kaggle nominado como Predict students' dropout and academic success, al cual se puede acceder por medio del siguiente enlace:  https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention, este conjunto de datos viene en un archivo en formato csv (valores separados por comas), y tiene un tamaño aproximado de 89 KB. 

## Descripción del Problema

El conjunto de datos con el que se cuenta está definido a través de 35 campos, los cuales representan variables sociodemográficas, económicas y de rendimiento académico, donde el último campo representa las etiquetas que permiten distinguir a cada registro en cualquiera de tres categorías, Dropout [Abandonó], Graduated [Graduado], Enrolled [matriculado], las anteriores características permiten predecir las probabilidades de abandono y el éxito académico de los estudiantes, a través de la aplicación de técnicas de Machine Learning orientadas al aprendizajes supervisado, para el caso de este proyecto, se aplicaron dos técnicas la primera es una red convolucional con una capa convolucional en 1D, una capa de aplanamiento (flatten) y una capa de salida densamente conectada, en el caso de la segunda técnicas se tiene una máquinas de soporte vectorial denotada como SVM por sus siglas en inglés.

## Descripción del Modelo

El proyecto de predicción del abandono y éxito académico contó con la implementación de dos modelos de aprendizaje supervisado con los que se buscaba determinar cuál de ellos ofrecía un mayor desempeño frente a la clasificación del conjunto de datos disponible en función de los predictores definidos para dicho conjunto, para este propósito, se definieron un conjunto de métricas que permitirían comparar el desempeño de ambos modelos y elegir cuál de ellos cumplía de mejor manera con el objetivo del proyecto.
A continuación, se definen de forma más detallada los atributos de los modelos implementados en el proyecto.

Red Neuronal Convolucional:

La red neuronal convolucional fue definida a través de una capa de entrada con 34 neuronas, una capa convolucional 1D también con 34 neuronas, una capa de maxpooling en 1D de 33 neuronas de entrada y 16 de salida, una capa de aplanamiento flatten de 16 neuronas de entrada y 512 de salía y finalmente una capa densamente conectada con 512 neuronas de entrada y 3 de salida las cuales corresponden al número de categorías definidas en la columna Target del conjunto de datos. Para el modelo se definió que el checkpoint evaluara el valor máximo del accuracy en validación y que guardara los pesos del mejor modelo obtenido. En el caso del stopping se estableció también el acuraccy en validación y se definió que se cargaran de forma automática los pesos del mejor modelo una vez finalizado el proceso de refinamiento.

En relación con el desempeño del modelo, la siguiente gráfica muestra la tasa de error en función del número de épocas (sección en la que el modelo procesa todos los datos), se obtiene un comportamiento decreciente tanto en entrenamiento como en validación, alcanzando una tasa de perdida máxima cercana al 37% en validación.

<img src="https://i.postimg.cc/C13y5s87/grafica-error.png" />

En relación con la exactitud, se puede evidenciar en la gráfica de abajo que dicha métrica exhibe un comportamiento creciente por lo que el número de aciertos de clasificación incrementa conforme aumenta el número de épocas, esto sugiere, que la red va ajustando sus parámetros internos de forma favorable logrando abstraer las relaciones entre los descriptores de los datos, por lo tanto, la red aprende conforme aumenta el número de épocas.

<img src="https://i.postimg.cc/7h0F2JMk/grafica-accuracy.png" />


Para reconocer el desempeño de la red de una forma más concreta, se calcularon métricas adicionales al accuracy, como son: precision, f1_score y el recall_score, estas últimas con average macro. En la imagen de abajo, se pueden apreciar los valores obtenidos por el modelo en las métricas referidas.

<img src="https://i.postimg.cc/6QpzgcHY/A1.png" />

Lo anterior, permite advertir que el modelo tiene una precisión, es decir, una tasa de clasificaciones positivas que corresponden en efecto a la clase positiva de un 78,15%, de igual forma, exhibe un recall, clasificaciones correctas de la clase positiva, de 60,3% y un f1_score, del 56,91%, dicha métrica denota la media armonía entre la precisión y el recall, se obtiene un valor discreto, lo que sugiere que tanto la precisión y el recall están en un margen medio y finalmente un accuracy del del 75%, que indica que esta proporción de instancias se clasificó de manera correcta. Cabe señalar, que en esta implementación de la red se empleó una función de pérdida “binary_cross_entropy”, sin embargo, el problema que se tiene es un problema multiclase, lo cual indica que el modelo podría estar segmentando las clases en dos grupos para realizar las respectivas clasificaciones. 

Probando con la función de pérdida “categorical_cross_entropy”:

Esta función de perdida es la recomendada para un problema multiclase. Para el caso de esta variante se muestran las métricas obtenidas en la imagen de abajo.

<img src="https://i.postimg.cc/ZqsjSN2Z/A2.png" />

El comportamiento del error en función del número de épocas, muestra una tendencia decreciente, con picos marcados, lo cual indica que al modelo le cuesta reconocer los patrones entre los datos y tarda en ajustarse alcanzando márgenes de rendimiento más bajos en comparación con la implementación anterior.

<img src="https://i.postimg.cc/qM2m3zhX/A3.png" />

Respecto a la exactitud se advierte un comportamiento creciente, lo cual sugiere que el modelo está aprendiendo conforme se va refinando época a época, sin embargo, los picos pronunciados muestran que el modelo es volátil y no logra descifrar en un margen amplio las relaciones entre las variables descriptoras de los datos.

<img src="https://i.postimg.cc/RZ9gQ4wr/A4.png" />

Comparativa entre los modelos:

En la siguiente imagen se muestra la comparativa en el rendimiento de los modelo en el dashboard de ML Flow, siendo la primer columna la correspondiente a la SVM.

<img src="https://i.postimg.cc/Hk2B6Rtr/A5.png" />

## Conclusiones y Recomendaciones

El modelo con mejor desempeño fue la red neuronal convolucional, estando sus dos variantes muy cercanas en desempeño, pues el rendimiento de la svm muestra un rezago amplio, en buena medida por la definición de los atributos de su arquitectura.

