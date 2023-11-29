# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

El conjunto de datos cuenta con 35 columnas y 4424 registros, la última colummna posee las etiquetas de clasificación entre 4 posibles categorías, las cuales se enuncian a continuación: Dropout [Abandonó], Graduated [Graduado], Enrolled [matriculado], estás etiquetas posibilitan la implementación de un sistema de clasificación supervisado basado en técnicas de Machine Learning.

## Resumen de calidad de los datos

En esta sección se presenta un resumen de la calidad de los datos. Se describe la cantidad y porcentaje de valores faltantes, valores extremos, errores y duplicados. También se muestran las acciones tomadas para abordar estos problemas.

## Variable objetivo

La varianble objetivo de este conjunto de datos, está nominada como "Target", pues, cuenta con las etiquetas entre las cuales se clasifica la población según sus probabilidades de éxito o abandono académico.

A continuación, se muestra el histograma de la variable de salida.
<img src="https://i.postimg.cc/28n3PJkS/histograma-var-salida.png" />

## Variables individuales

En esta sección se presenta un análisis detallado de cada variable individual. Se muestran estadísticas descriptivas, gráficos de distribución y de relación con la variable objetivo (si aplica). Además, se describen posibles transformaciones que se pueden aplicar a la variable.

Obtenemos la descripción estadística de todas las variables del conjunto de datos.

La siguiente imagen muestra las estadísticas de los 8 primeros campos:

<img src="https://i.postimg.cc/5yjhvjg2/descrip-PF-1.png" />


La imagen de abajo muestra las estadísticas de los campos del 9 al 16:

<img src="https://i.postimg.cc/J0g9QmYc/descrip-PF-2.png" />


La imagen de abajo muestra las estadísticas de los campos del 17 al 24:

<img src="https://i.postimg.cc/hG1RRkhc/descrip-PF-3.png" />


La imagen de abajo muestra las estadísticas de los campos del 25 al 32:

<img src="https://i.postimg.cc/zBDZzxRW/descrip-PF-4.png" />


La imagen de abajo muestra las estadísticas de los campos del 33 al 35:

<img src="https://i.postimg.cc/tTr0YFhc/descrip-PF-5.png" />

A continuación, se muestran la cantidad de valores presentes por cada etiqueta de salida.

|Etiqueta| Cantidad|
|---|---|
|Graduate|    2209|
|Dropout |   1421|
|Enrolled|     794|

## Ranking de variables

En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo. Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.

## Relación entre variables explicativas y variable objetivo

En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.

Obtenemos la matriz de correlación del conjunto de datos y la representamos como un mapa de calor.

<img src="https://i.postimg.cc/GpxxRCYL/matriz-correl-mapheat.png" />

Se encuentra que varias variables suscriben una correlación alta, por ejemplo, international y las diferentes denominaciones de unidades curriculares.

La matriz de dispersión obtenida para el conjunto de datos se muestra abajo, cabe resaltar que existe sobreposición entre los nombres de la variables debido a la gran cantidad de campos con los que cuenta el conjunto de datos.

<img src="https://i.postimg.cc/4N4mW2z9/matriz-de-dispersion.png" />

## Script de python
El script de python con las instrucciones necesarias para aplicar las operaciones descritas previamente sobre los datos, se encuentran en el directorio: data/clean_data.py de este repositoro.
