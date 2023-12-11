# Informe de salida

## Resumen Ejecutivo

En este proyecto se aplicaron tres modelos con características diferentes para la predicción del abandono y el éxito académico usando variables sociodemográficas, económicas y de rendimiento escolar, la evaluación del desempeño de los modelos, se realizó a través del cálculo de métricas como accuracy, precision, recall_score, y f1_score, las cuales permitieron efectuar una comparativa del desempeño de los modelos.

## Lecciones aprendidas

A lo largo de este proyecto se aprendieron diferentes elementos a lo largo de las diferentes etapas, por ejemplo, el preprocesamiento de los datos debe hacerse de manera meticulosa con el ánimo de hallar registros defectuosos, atípicos o incongruentes, también es importantes, realizar las normalizaciones o codificaciones de los datos según el tipo de variable que representen, de este trabajo dependerán que los análisis posteriores tengan validez y que puedan tomarse decisiones apropiadas a partir de estos. Posteriormente en la definición de los modelos, es necesario tomarse el tiempo para maniobrar con los parámetros internos de dichos modelos, jugando un poco con las arquitecturas, lo cual, va de la mano con el conocimiento profundo del problema y el contexto sobre el que este está situado, adicional, deben complementarse estos esfuerzos con técnicas de refinamiento para optimizar la elección de hiper-parámetros y dar con ello, con el conjunto de características con los que los modelos generan mejores resultados. Finalmente, están las herramientas de gestión del código, de los datos y los modelos, las cuales permiten y monitoreando la evolución de los proyectos, gestionando de manera eficiente el ciclo de vida del software. Los modelos fueron puestos es producción para poder gestionarlos desde la plataforma MLFlow e invocarlos desde las aplicaciones que figuran como clientes.

## Impacto del proyecto

La implementación de los modelos tiene un gran impacto en instituciones educativas de todos los niveles, tanto para aquellas donde se generan las políticas públicas como aquellas adscritas al marco del escenario de la aplicación directa de estrategias de enseñanza-aprendizaje (centros de estudio, universidades, colegios, instituciones de educación)

## Conclusiones

Los tres modelos implementados, muestran desempeños interesantes, dados los ajustes en sus arquitecturas e hiperparámetros, sin embargo, la red neuronal convolucional muestra un mejor desempeño, dados los valores de las métricas obtenidas, siendo la svm el modelo con peor desempeño.
