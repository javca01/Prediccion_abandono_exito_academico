# Predicción del abandono y  éxito académico con ML - Entendimiento del Negocio

## Nombre del proyecto:

Predicción del abandono y el éxito académico con ML.

## Objetivo del proyecto:

Desarrollar un modelo de Machine Learning que prediga las probabilidades de que un estudiante abandone o se gradúe de un programa en función de la valoración de diferentes características sociodemográficas, académicas y económicas.

## Transfondo del negocio:
- ¿Quién es el cliente o los beneficiarios del proyecto? ¿En qué dominio se encuentran (marketing, medicina, entre otros)?

Las instituciones educativas en todos los niveles, las entidades gubernamentales del sector educación y las empresas privadas que brindan servicios educativas, mostrarían un gran interés por advertir las características, económicas, académicas y sociodemográficas que influyen en mayor medida sobre la retención de los estudiantes en el sistema educativo y sobre aquellas que los conducen a su abandono, lo cual, puede contribuir al diseño de estrategias pedagógicas o la creación de políticas públicas que propendan por programas que mejoren las condiciones de los programas educativas y posibiliten las condiciones para la permanencia en las instituciones educativas y la culminación de los estudios.

- ¿Qué problemas del negocio o del dominio estamos tratando de solucionar?

Determinación de características familiares, económicas de la región o académicas que inciden sobre la permanencia de los estudiantes en los programas de formación, para generar un modelo de Machine Learning que permita predecir con base en estos factores, la probabilidades de que un estudiante culmine sus estudios.

## Alcance del proyecto:

- ¿Qué solución basada en Deep Learning queremos implementar?
Dada las características del problema a tratar y considerando que se cuenta con las etiquetas de salida, se adoptará una estrategia de aprendizaje supervisado orientado hacia la clasificación, por lo anterior, se aplicará un algoritmo KNN (K-vecinos cercanos) y una SVM (Máquina de Soporte Vectorial) con el ánimo de advertir cuál de los dos algoritmos presenta un mejor rendimiento.
  
- ¿Qué se hará?
  El conjunto de datos con el que se trabajará lleva por nombre, Predict students' dropout and academic success, tomado del repositorio de Kaggle, en la siguiente dirección: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data , este conjunto de datos está compuesto por 35 columnas y 4424 registros. Las características con las que cuenta el dataset se muestran a continuación:

|#|Campo|Descripción|Tipo|
|--|--|--|--|
|1.	|Marital status [Estado civil]:| El estado civil del estudiante. |(Categórico)|
|2.	|Application mode [Modo de solicitud]:| El método de solicitud utilizado por el estudiante. |(Categórico)|
|3.|	Application order [Orden de solicitud]:| El orden en que el estudiante presentó la solicitud. |(Numérico)|
|4.|	Course [Curso]:| El curso realizado por el alumno.| (Categórico)|
|5.|	Daytime/evening attendance [Asistencia diurna/noche]:| Si el alumno asiste a clase durante el día o por la tarde. |(Categórica)|
|6.|	Previous qualification [ Titulación previa]:| La titulación obtenida por el estudiante antes de matricularse en la enseñanza superior. |(Categórica)|
|7.	|Nacionality [ Nacionalidad]:| La nacionalidad del estudiante. |(Categórico)|
|8.	|Mother's qualification [Titulación de la madre]:| La cualificación de la madre del estudiante. |(Categórico)|
|9.|	Father's qualification [Titulación del padre]:| La cualificación del padre del estudiante.| (Categórico)|
|10.	|Mother's occupation [Profesión de la madre]:| La ocupación de la madre del estudiante.| (Categórico)|
|11.|	Father's occupation [Profesión del padre]:| La ocupación del padre del estudiante.| (Categórico)|
|12.|	Displaced [Desplazado]:| Si el estudiante es una persona desplazada. |(Categórico)|
|13.	|Educational special needs [Necesidades Educativas Especiales]: |Si el alumno tiene necesidades educativas especiales. |(Categórico)|
|14.	|Debtor:[Deudor] |Si el alumno es deudor. |(Categórico)|
|15.	|Tuition fees up to date [Colegiaturas]:| Si las tasas de matrícula del estudiante están al día.|(Categórico)|
|16.|	Gender [Genero]: |El sexo del estudiante. |(Categórico)|
|17.	|Scholarship holder [Becario]:| Si el alumno es becario.| (Categórico)|
|18.	|Age at enrollment [Edad en el momento de la inscripción]:| La edad del alumno en el momento de la matriculación.|(Numérico)|
|19.	|International [Internacional]: |Si el estudiante es internacional. |(Categórico)|
|20.	|Curricular units 1st sem (credited) [Unidades curriculares 1er sem (acreditadas)]:| El número de unidades curriculares acreditadas por el estudiante en el primer semestre.| (Numérico)|
|21.	|Curricular units 1st sem (enrolled) [Unidades curriculares 1er sem (matriculado)]:| El número de unidades curriculares matriculadas por el estudiante en el primer semestre.| (Numérico)|
|22.|	Curricular units 1st sem (evaluations) [Unidades curriculares 1er sem (evaluadas)]:| El número de unidades curriculares evaluadas por el estudiante en el primer semestre. |(Numérico)|
|23.	|Curricular units 1st sem (approved) [Unidades curriculares 1er sem (aprobadas)]: |El número de unidades curriculares aprobadas por el estudiante en el primer semestre.| (Numérico)|
|24.	|Curricular units 1st sem (grade)|--|--|
|25.|	Curricular units 1st sem (without evaluations)|--|--|
|26.	|Curricular units 2st sem (credited) [Unidades curriculares 2do sem (acreditadas)]: |El número de unidades curriculares acreditadas por el estudiante en el primer semestre.| (Numérico)|
|27.	|Curricular units 2st sem (enrolled) [Unidades curriculares 2do sem (matriculado)]:| El número de unidades curriculares matriculadas por el estudiante en el primer semestre. |(Numérico)|
|28.	|Curricular units 2st sem (evaluations) [Unidades curriculares 2do sem (evaluadas)]:| El número de unidades curriculares evaluadas por el estudiante en el primer semestre. |(Numérico)|
|29.	|Curricular units 2st sem (approved) [Unidades curriculares 2do sem (aprobadas)]: |El número de unidades curriculares aprobadas por el estudiante en el primer semestre. |(Numérico)|
|30.	|Curricular units 2st sem (grade)|--|--|
|31.	|Curricular units 2st sem (without evaluations)|--|--|
|32.	|Unemployment rate [Tasa de desempleo]:|--|--|
|33.	|Inflation rate [Tasa de inflación]:|--|--|
|34.	|GDP [PIB]:|--|--|
|35.	|Target [Objetivo]: |Clasificación del registro entre Dropout [Abandonó], Graduated [Graduado], Enrolled [matriculado].|--|

## Metodología:

El proyecto se llevará a cabo a través de avances semanales donde se irán implementando las diferentes etapas de la metodología CRISP, las cuales se descomponen a grandes rasgos en: Entendimiento del Negocio, Entendimiento de los Datos, Preprocesamiento de los datos, Modelamiento, Evaluación e Implementación (Colocar el modelo en Producción)

## Cronograma:

Las diferentes etapas del proyecto se implementarán de acuerdo al diagrama que se muestra abajo.

<img src="https://i.postimg.cc/JhfnQPMt/cronograma-MLDS-M3.png)https://i.postimg.cc/JhfnQPMt/cronograma-MLDS-M3.png" />

## Autor del proyecto:

Javier Caicedo Pedrozo

## Datos:

- ¿De dónde vienen los datos?

  Los datos fueron tomados del repositorio de Kaggle, concretamente del proyecto Predict students' dropout and academic success, el conjunto posee 35 características en 4424 registros, condensados en un archivo en formato csv con un tamaño de 460KB.
  
- ¿Se usa alguna herramienta o proceso para la descarga de la información?

  El archivo con el conjunto de datos fue descargado el repositorio de Kaggle, para posteriormente alojarlo en el Drive de una cuenta personal, desde el cual se generó un archivo de descarga.
  
- ¿Qué tipo de datos estamos manejando?

  El conjunto de datos está compuesto principalmente por datos numéricos y categóricos.

- Carga de los datos:

  El script de python para cargar los datos desde el sitio donde están alojados se encuentra en la ruta scripts/data_acquisition y lleva por nombre carga_data.py.
