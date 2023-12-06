# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:**
  En el desarrollo del proyecto, se plantearon dos modelos el primero lleva por nombre model_conv_network, pues fue definido a partir de una red convolucional, cuya arquitectura está integrada por una capa convolucional 1D, una de max pooling 1D, una de aplanamiento (flatten) y finalmente una capa de salida densamente conectada, tal como se ilustra en la imagen de abajo.

<img src="https://i.postimg.cc/K8gSRZc6/desp-1.png" />

  
- **Plataforma de despliegue:**

La plataforma seleccionada para realizar el despliegue fue MLflow, la cual es una plataforma de código abierto, que permite gestionar de forma sencilla el ciclo de vida de un proyecto de machine learning, entre sus ventajas se cuentan: a) permite realizar un seguimiento estricto de los atributos de los experimentos de aprendizaje de máquina, como pueden ser parámetros característicos, métricas de desempeño, entre otros, facilitando la labora de compara modelos, lo cual propenden por la búsqueda de consistencia y calidad de los modelos en producción, b) es compatible con múltiples lenguajes y frameworks dentro del ecosistema de machine learning, posibilitando la integración de modelos entrenados con diferentes tecnologías, c) brinda herramientas que hacen flexible la implementación de modelos en producción, por medio de su compatibilidad con plataformas en la nube que permiten que los despliegues sean escalables y distribuidos, d) brinda la posibilidad de monitorear el rendimiento de los modelos que se encuentran en producción al permitir el seguimiento de las métricas de desempeño, desplegando alertas cuando el rendimiento del modelo no es favorable o no cumple con los estándares definidos para la aplicación.


   
- **Requisitos técnicos:**

Entre los requisitos técnicos necesarios para el despliegue de la aplicación se cuentan las versiones de las librerías empleadas, así como los requisitos de hardware que deben tener las máquinas sobre las cuales se monten los modelos. A continuación, se mencionan las versiones de las librerías empleadas.
|Libreria|Version|
|--|--|
|Python| 3.10.12|
|NumPy |1.23.5|
|pandas| 1.5.3|
|tensorflow |2.14.0|
|matplotlib| 3.7.1|
|sklearn |1.5.3|
|seaborn |0.12.2|
|pyngrok| 7.0.3|
|mlflow |2.1.0|



- **Requisitos de seguridad:**

Al implementar aplicaciones de machine learning en la etapa de producción es indispensable el uso de estrategias de seguridad que mantengan fiable la aplicación y no comprometan la integridad de los datos de los clientes, algunas de las estrategias que se pueden aplicar son: a)autenticación y autorización: emplear un sistema de gestión de identidades para asignar roles que determinen los permisos de accesos a los recursos de la aplicación, b) encriptación de datos: se puede implementar el protocolo HTTPS para cifrar la comunicación de extremo a extremo (cliente-servidor), empleando herramientas de encriptación de disco, c) Monitoreo y registro de la actividad de la aplicación, por ejemplo, errores, actividad de los usuarios, tipos de solicitudes, entre otras, d) Protección contra ataques maliciosos, por medio de pruebas de adversarios para evitar la inyección de información que pueda provocar anomalías o fallos, así como practicar las respectivas pruebas de penetración que diluciden fallos en las técnicas de ciberseguridad.
  

 - **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

La arquitectura definida para el despliegue de la aplicación se muestra a continuación:


<img src="https://i.postimg.cc/jj0VfGLL/desp-2.png" />


## Código de despliegue

- **Archivo principal:** (nombre del archivo principal que contiene el código de despliegue)
- **Rutas de acceso a los archivos:** (lista de rutas de acceso a los archivos necesarios para el despliegue)
- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)

## Documentación del despliegue

- **Instrucciones de instalación:**

Proceso de creación del modelo:

En la pestaña Experiments ubicamos el experimento que creamos, en el caso del primer modelo desarrollado en este proyecto se trata de conv_network.

<img src="https://i.postimg.cc/R0ckLXHG/desp-4.png" />

Luego daremos clic en el nombre del run, que en este caso es default_logistic.

<img src="https://i.postimg.cc/yxN2BKrr/desp-5.png" />

Lo anterior, nos permitirá acceder a los atributos del run principal de nuestro experimento.
Al interior de este run podemos ver los valores de las métricas generadas para el modelo y cargadas en MLFlow.

<img src="https://i.postimg.cc/B6pRdbNK/desp-6.png" />


- **Instrucciones de configuración:**

También podremos apreciar nuestro modelos, en la pestaña artifacts, donde aparecen todas las dependencias vinculadas al proyecto, por ejemplo, los archivos conda.yaml, model.pkl, Python_env.yaml y requirements.txt.
En esta sección podremos registrar nuestro modelo para posteriormente llevarlo a producción.



<img src="https://i.postimg.cc/jSGTXDkq/desp-12.png" />

Luego seleccionamos nuestro experimento, en este caso, conv_network, y después, elegimos el run principal, que para este proyecto es default_logistic. En la nueva pantalla, en la pestaña Artifacts elegimos la carpeta confusion_matrix y finalmente seleccionamos la imagen que cargamos desde la aplicación cliente.

<img src="https://i.postimg.cc/fLshG6N6/desp-13.png" />


- **Instrucciones de uso:**
Proceso para llevar a producción el modelo:

En la pestaña artifacts, nos cercioramos que la carpeta raíz model se encuentre seleccionada, en este caso, en la parte derecha de dicha sección aparecerá en azul el botón Register Model, sobre el cual deberemos dar clic.

<img src="https://i.postimg.cc/BbwknzrZ/desp-7.png" />

Posteriormente, emergerá una nueva ventana, la cual cuenta con dos espacios para rellenar, el primero es Model, donde deberemos elegir la opción Create New Model, y la segunda es Model Name, donde deberemos colocar el nombre que le asignaremos a nuestro modelo en la plataforma.

<img src="https://i.postimg.cc/2jHPGq1p/desp-8.png" />

Al dar clic en el botón register, se cerrará la ventana y en la ventana principal en la esquina superior derecha de la sección artifacts se podrá ver la confirmación del registro de nuestro modelo.

<img src="https://i.postimg.cc/xTpnXxvz/desp-9.png" />

Luego nos dirigimos a la pestaña principal Models, en la esquina superior izquierda de la interfaz y en la nueva pantalla, podremos observar que nuestro modelo ya se encuentra registrado.

<img src="https://i.postimg.cc/6p7wnrc1/desp-10.png" />

Posteriormente, daremos clic en el nombre de nuestra última versión, para el caso de este proyecto se trata de Version 1. Al hacerlo saltaremos a otra pantalla, donde se encuentran las secciones: Description, Tags y Schema y en la que encontraremos la pestaña Stage, en la cual seleccionaremos la opción Production.

<img src="https://i.postimg.cc/MGVSrzh7/desp-11.png" />

Finalmente, al dar clic de nuevo en la pestaña Models, veremos listado muestro modelo con la designación de la versión 1 en producción.
En este punto nuestro modelo ya se encuentra disponible para realizar las predicciones que necesitemos, haciendo las peticiones desde cualquier aplicación cliente a nuestro servicio en MLFlow.

Matriz de Confusión del sistema:

La matriz de confusión es una herramienta de representación gráfica que permite sintetizar la efectividad del sistema, a través del registro de los aciertos y errores en las predicciones del modelo.

MLFlow asume la matriz de confusión como un artefacto, una vez que hayamos generado y cargado la matriz de confusión de nuestro modelo desde nuestra aplicación cliente podremos subirla a nuestro servicio en MLFlow. Para observar la matriz de confusión en MLFlow debemos seguir los siguientes pasos.

En la ventana principal nos cercioramos que la pestaña Experiments esté seleccionada.
- **Instrucciones de mantenimiento:**
Montaje del segundo modelo:

En el caso del segundo modelo se optó por una máquina de soporte vectorial (SVM) por sus siglas en inglés. Para este modelo el procedimiento para subir los datos a MLFlow fue examente igual al primer modelo, por lo que se comentarán los elementos que adjuntaron al modelo en este servicio.

<img src="https://i.postimg.cc/fLshG6N6/desp-13.png" />

Parámetros:

Se implementó una búsqueda de parámetros a través de la estrategia GridSearch, la cual arrojó que los mejores valores para C y gamma, eran respectivamente 2 y 0,03125.

<img src="https://i.postimg.cc/pTBtRqVy/desp-14.png" />

Métricas:

Para este modelo se obtuvieron las representaciones micro del f1_score, precision y recall, así como el accuracy, cuyos valores se muestran a continuación.

<img src="https://i.postimg.cc/VvGQbvC8/desp-15.png" />

Registro del modelo:

Se registra el modelo como model_svm_rbf y se coloca en producción así como se hizo con el primer modelo.

<img src="https://i.postimg.cc/T1Yvwgh0/desp-16.png" />

Se define el stage de la versión 1 del modelo en producción.

<img src="https://i.postimg.cc/HWBCNZPh/desp-17.png" />

A continuación, se muestra el recuento de los dos modelos del proyecto en producción.

<img src="https://i.postimg.cc/kXGdPQ0X/desp-18.png" />

Matriz de confusión:

En la pestaña Experimentos, se debe seleccionar el experimento marco en el que se alojaron todos los atributos de nuestro modelo; una vez el experimento se ha seleccionado, debemos seleccionar la pestaña Artifacts para posteriormente elegir la carpeta confusion_matrix, en esta se aloja la imagen de la matriz de confusión que condensa la efectividad de nuestro modelo.

<img src="https://i.postimg.cc/kXHPJVCN/desp-19.png" />

