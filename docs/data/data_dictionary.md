# Diccionario de datos

## Base de datos data_edu

**La base de datos data_edu está compuesta por 35 campos y 4424 registros, cuenta con descriptores demográfico, socio-económicos y de rendimiento académico que permiten predecir la probabilidad de éxito o de abandono de un estudiante.

|#| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
|---| --- | --- | --- | --- | --- |
| 1|	Marital status [Estado civil] | El estado civil del estudiante. (Categórico)| Numerico | 1-6 | data_educacion.csv |
| 2|	Application mode [Modo de solicitud]  | El método de solicitud utilizado por el estudiante. (Categórico) | Numerico | 1-18 | data_educacion.csv |
| 3|	Application order [Orden de solicitud] | El orden en que el estudiante presentó la solicitud. (Numérico) | Numerico | 0-9 | data_educacion.csv |
| 4|	Course [Curso] | El curso realizado por el alumno. (Categórico) | Numerico | 1-17 | data_educacion.csv |
| 5|	Daytime/evening attendance [Asistencia diurna/noche] | Si el alumno asiste a clase durante el día o por la tarde. (Categórica) | Numerico| 1-17 | data_educacion.csv |
| 7|	Nacionality [ Nacionalidad]   | La nacionalidad del estudiante.  (Categórica)  | Numerico| 1-21 | data_educacion.csv |
| 8|	Mother's qualification [Titulación de la madre]:    | La cualificación de la madre del estudiante. (Categórico)  | Numerico | 1-29| data_educacion.csv |
| 9|	Father's qualification [Titulación del padre]:     | La cualificación del padre del estudiante. (Categórico) | Numerico| 1-34 | data_educacion.csv |
| 10|	Mother's occupation [Profesión de la madre]:      | La ocupación de la madre del estudiante. (Categórico) | Numerico| 1-32 | data_educacion.csv |
| 11|	Father's occupation [Profesión del padre]      | La ocupación del padre del estudiante. (Categórico) | Numerico| 1-46 | data_educacion.csv |
| 12|	Displaced [Desplazado]: | Si el estudiante es una persona desplazada. (Categórico)  | Numerico| 0-1 | data_educacion.csv |
| 13| Educational special needs | Si el alumno tiene necesidades educativas especiales. (Categórico)|Numerico|0-1| data_educacion.csv |
| 14| Debtor | Si el alumno es deudor. Categórico |Numerico|0-1| data_educacion.csv |
| 15| Tuition fees up to date | Si las tasas de matrícula del estudiante están al día. ( Categórico) |Numerico|0-1| data_educacion.csv |
| 16| Gender | El sexo del estudiante. (Categórico)|Numerico|0-1| data_educacion.csv |
| 18| Age at enrollment | La edad del alumno en el momento de la matriculación. | Numérico |17-70 | data_educacion.csv |
| 19| International | Si el estudiante es internacional. (Categórico) |Numerico|17-70| data_educacion.csv |
| 20     | Curricular units 1st sem (credited) | El número de unidades curriculares acreditadas por el estudiante en el primer semestre. |Numérico| 0-20 | data_educacion.csv |
| 21     | Curricular units 1st sem (enrolled) | El número de unidades curriculares matriculadas por el estudiante en el primer semestre. | Numérico|0-26| data_educacion.csv |
| 22     | Curricular units 1st sem (evaluations) | El número de unidades curriculares evaluadas por el estudiante en el primer semestre. |Numérico| 0-45 | data_educacion.csv |
| 23     | Curricular units 1st sem (approved) | El número de unidades curriculares aprobadas por el estudiante en el primer semestre.| Numérico| 0-26 | data_educacion.csv |
| 24     | Curricular units 1st sem (grade)(*) | -- |Numerico | 0-1,16e16 | data_educacion.csv |
| 25     | Curricular units 1st sem (without evaluations)| -- | Numerico| 0-12  | data_educacion.csv |
| 26     | Curricular units 2nd sem (credited) | El número de unidades curriculares acreditadas por el estudiante en el segundo semestre. | Numérico|0-19 | data_educacion.csv |
| 27     | Curricular units 2nd sem (enrolled) | El número de unidades curriculares matriculadas por el estudiante en el segundo semestre.| Numérico| 0-23 | data_educacion.csv |
| 28     | Curricular units 2nd sem (evaluations) | El número de unidades curriculares evaluadas por el estudiante en el segundo semestre. | Numérico|0-33 | data_educacion.csv |
| 29| Curricular units 2nd sem (approved) | El número de unidades curriculares aprobadas por el estudiante en el segundo semestre. |Numérico| 0-20 | data_educacion.csv |
| 30| Curricular units 2nd sem (grade)(*)|-- | Numerico| 0-1,85e16  | data_educacion.csv |
| 31| Curricular units 2nd sem (without evaluations)|--   | Numerico|0-12  |data_educacion.csv|
| 32| Unemployment rate | Tasa de desempleo. |Numerico| 76-162 | data_economic.csv |
| 33| Inflation rate | Tasa de inflación.  |Numerico| -8-37  | data_economic.csv |
| 34| GDP GDP de la población. |--| Numérico|-406 - 351 | data_educacion.csv |
| 35| Target |Clasificación del registro (Categórico) |Dropout [Abandonó], Graduated [Graduado], Enrolled [matriculado] |Numerico |data_educacion.csv |





- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.



- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.
