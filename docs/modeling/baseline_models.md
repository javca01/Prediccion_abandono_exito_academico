# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

El modelo baseline es el primer modelo construido y se utiliza para establecer una línea base para el rendimiento de los modelos posteriores.
En este caso el modelo baseline se definión a través de una red neuronal convolucional, la cual contaba con una capa convolucional en 1D, otra de maxpooling, otra de aplanamiento y una de salida densamente conectada.

## Variables de entrada

Las variables de entrada son las 34 primeros campos del conjunto de datos que contemplan variables de tipo demográfico, socio-económico y de rendimiento académico.

## Variable objetivo

La variable de salida es nominada como Target y contiene el conjunto de etiquetas que catalogan a un individuo en función de su probabilidad de éxito o deserción académica, las etiquetas originales fueron codificadas de la siguiente forma: 'Dropout': 0, 'Graduate': 1, 'Enrolled': 2

## Evaluación del modelo

### Métricas de evaluación

Dentro de las métricas consideradas para medir el desempeño del sistema, se resaltan: precission, recall, f1_score. A continuación, se muestra el reporte de clasificación con los valores obtenidos para las diferentes métricas.

<img src="https://i.postimg.cc/dtNBjxhW/reporte-clasificacion.png" />

Para el análisis del reporte anterior, conviene recordar que:

- Precisión: Porcentaje de predicciones positivas correctas respecto al total de predicciones positivas.

- Recall: Porcentaje de predicciones positivas correctas en relación con el total de predicciones positivas reales.

- Puntuación F1: Media armónica ponderada de precisión y recuperación. Cuanto más se acerque a 1, mejor será el modelo.

Se puede indicar que las clasificaciones de las clases 0 y 1 tuvieron porcentajes importantes de clasificaciones correctas, y teniendo en cuenta que dichas clases son las más numerosas, se puede referir que el sistema aprendió a reconocer que estudiantes podrían eventualmente tener éxito académico o abandonar sus estudios, en función de las variables demográficas, socio-económicas y de rendimiento académico que constituian la entrada al sistema.

### Arquitectura de la red
Se trabajón con una red convolucinoal cuya arquitectura se resumen en la siguiente imagen.

<img src="https://i.postimg.cc/JzcFmWKp/arquitectura-red.png" />

### Resultados de evaluación

El desempeño del sistema en cuanto a su nivel de aprendizaje y exactitud se muestra en las siguiente gráficas.

En la imagen de abajo se muestra la evalución del error por las epochs de entrenamiento.
<img src="https://i.postimg.cc/7h0F2JMk/grafica-accuracy.png" />

En la siguiente gráfica se muestra el desempeño del accuracy en el tiempo.

<img src="https://i.postimg.cc/7h0F2JMk/grafica-accuracy.png" />


## Análisis de los resultados

El modelo tuvo un buen desempeño, sin embargo es conveniene experiementar más con la arquitectura de la red.

## Conclusiones

- Contrario a lo esperado el modelo multiclase, tiene un mejor rendimiento con una función de perdida binary_crossentropy, cuando debiese ser modelado con una función categorical_crossentropy, lo cual podría sugerir, que el sistema aprende a distinguir mejor un conjunto de clase como abandono (Dropout) y Graduate (Graduado).
- 

## Script de python
El script de python que contiene todas las operaciones referidas arriba se puede encontrar en el directorio modeling/modeling.py en este repositorio.
