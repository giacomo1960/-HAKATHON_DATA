 **1. DATASET IRANI BASE PREDICION**

# **1.1. MODELO CON EL NUEVO DATASET IRANI**

En este bloque de código importa las librerías necesarias para el preprocesamiento de datos, el modelado y la evaluación.

**ConfusionMatrixDisplay**:
Herramienta de sklearn para visualizar matrices de confusión, útil para evaluar el rendimiento de modelos de clasificación.

**warnings**: 
Módulo para gestionar advertencias en Python, configurado para ignorar todas las advertencias en este script.

**imblearn.pipeline.Pipeline as imbpipeline**:
 Una extensión de sklearn.pipeline.Pipeline que permite integrar pasos de balanceo de clases (como oversampling y undersampling) dentro de un flujo de trabajo de preprocesamiento y modelado.

 **RandomOverSampler, SMOTE (de imblearn.over_sampling)**: 
Métodos para el sobremuestreo de la clase minoritaria, ayudando a equilibrar el conjunto de datos para el entrenamiento de modelos.

**StratifiedKFold, KFold, cross_validate (de sklearn.model_selection)**: 
Herramientas para la validación cruzada de modelos, permitiendo una evaluación robusta del rendimiento.

**RandomUnderSampler (de imblearn.under_sampling):**
 Método para el submuestreo de la clase mayoritaria, otra técnica para equilibrar el conjunto de datos.

**plotly.express as px**:
Librería para crear visualizaciones interactivas de manera sencilla.

**LogisticRegression (de sklearn.linear_model)**: 
El modelo de regresión logística, un algoritmo fundamental de clasificación.


# **1.2. VALIDACION INICIAL**

Este bloque de código se encarga de la carga inicial y preparación de los datos:

**Carga de Datos:**
Utiliza pandas para leer un archivo CSV directamente desde una URL de UCI Machine Learning Repository, cargando el *dataset iraní* en un DataFrame llamado **dfIranian**.

**Renombrado de Columnas**:
 Las columnas del DataFrame se renombran a nombres más descriptivos y en español (ej. 'Call Failure' a 'Fallos_Llamadas', 'Churn' a 'Abandono') para facilitar su comprensión y manejo. El DataFrame resultante se guarda como dfEsp.

**Inferencia de Tipos de Datos**: 
Se crea una copia de dfEsp llamada sample. Luego, se itera sobre las columnas de sample para inferir los tipos de datos óptimos para cada una (Int64 para enteros, float32 para flotantes, boolean para booleanos y string para el resto), almacenando esta información en el diccionario dtypes_map. 

Esto ayuda a optimizar el uso de memoria y asegura la correcta manipulación de los datos en fases posteriores.

**DataFrame dfEsp**
contiene 3150 entradas (filas) y 14 columnas.
https://github.com/giacomo1960/-HAKATHON_DATA/blob/main/1.2.VALIDACION%20INICIAL%20DATAFRAME%20ORIGINAL.png
Una observación crucial es que todas las columnas tienen 3150 valores no nulos, lo que indica que no hay valores faltantes en el conjunto de datos. 

Los tipos de datos predominantes son enteros (int64), con una columna (Valor_cliente) de tipo decimal (float64). 

La cantidad de memoria utilizada por el DataFrame es de 344.7 KB.


# **1.3.CARGA DATOS**

## **1.3.1. Carga completa con tipos optimizados**

**DataFrame dfEsp** contiene las siguientes columnas, con sus respectivas descripciones:

**Fallos_Llamadas**:Número de fallos en las llamadas realizadas por el cliente.

**Quejas**: Indica si el cliente ha presentado alguna queja (generalmente 0 para no, 1 para sí).

**Meses_permanencia**: Duración de la suscripción del cliente en meses.

**Cargo**: Cantidad de dinero cargada al cliente.

**Total_segundos**: Total de segundos de uso del servicio por parte del cliente.

**Total_llamadas**: Frecuencia total de uso del servicio (número de llamadas).

**Total_mensajes**: Frecuencia de envío de mensajes SMS.

**Llamadas_numeros_distintos**: Cantidad de números distintos a los que el cliente ha llamado.

**Grupo_edades**: Grupo de edad al que pertenece el cliente (categorizado).

**Plan_tarifa**: Tipo de plan tarifario contratado por el cliente.

**Estado_clientes**: Estado actual del cliente (por ejemplo, activo, inactivo).

**Edad**: Edad del cliente.

**Valor_cliente**: Valor del cliente, que puede ser una medida de su rentabilidad o importancia.

**Abandono**: Variable objetivo que indica si el cliente ha abandonado el servicio (Churn) (generalmente 0 para no, 1 para sí).

## **1.3.2. Normalizar Nombres de Columnas**

**DataFrame df** (con sus nombres de columna normalizados) contiene las siguientes columnas y sus descripciones:

**fallos_llamadas**: Número de fallos en las llamadas realizadas por el cliente.

**quejas**: Indica si el cliente ha presentado alguna queja (0 para no, 1 para sí).

**meses_permanencia**: Duración de la suscripción del cliente en meses.

**cargo**: Cantidad de dinero cargada al cliente.

**total_segundos**: Total de segundos de uso del servicio por parte del cliente.

**total_llamadas**: Frecuencia total de uso del servicio (número de llamadas).

**total_mensajes**: Frecuencia de envío de mensajes SMS.

**llamadas_numeros_distintos**: Cantidad de números distintos a los que el cliente ha llamado.

**grupo_edades**: Grupo de edad al que pertenece el cliente (categorizado).

**plan_tarifa**: Tipo de plan tarifario contratado por el cliente.

**estado_clientes**: Estado actual del cliente (por ejemplo, activo, inactivo).

**edad**: Edad del cliente.

**valor_cliente**: Valor del cliente, que puede ser una medida de su rentabilidad o importancia.

**abandono**: Variable objetivo que indica si el cliente ha abandonado el servicio (Churn) (0 para no, 1 para sí).

## **1.3.3. Validación de columnas clave**

Asegura que el DataFrame df contenga todas las columnas esperadas. 

a. Se capturan los nombres actuales de las columnas en expected_cols. 

b. Luego, se crea una lista missing para identificar cualquier columna de expected_cols que no se encuentre en df.columns. 

c. La salida Columnas faltantes respecto al esquema esperado: [] confirma que todas las columnas definidas al inicio del proceso están presentes en el DataFrame df, garantizando la integridad de la estructura de los datos para análisis posteriores.


# **1.4. AUDITORIA DE LA VARIABLE OBJETIVO Y CONSISTENCIA**

## **1.4.1. AUDITORIA DE LA VARIABLE OBJETIVO**

Este código audita y normaliza la columna 'abandono'. Lo que hace es:

a. Verifica su existencia, lanzando un error si no está. 

b. Si el tipo de dato es 'string', la limpia (str.strip().str.lower()) y 

c. la transforma a binaria (1 o 0) usando un mapeo (mapping), asegurando el tipo Int64 y

d. Finalmente, muestra la distribución de 'abandono' con value_counts().

**Resultado y Comentario**: La salida muestra 2565 'No abandono' (0) y 585 'Abandono' (1). 

Esta distribución desbalanceada es crítica, ya que sugiere la necesidad de **estrategias de balanceo de clases** para modelos predictivos efectivos.

# **1.4.2. Aplicacion de EDA**

Este código realiza un **Análisis Exploratorio de Datos (EDA)**, como sigue:

a. Muestra las primeras 10 filas de df con display(df.head(10)).

b. Imprime con print(df.info()) y entrega un resumen conciso de la estructura del DataFrame, incluyendo tipos de datos y conteo de no nulos. 

c. Calcula con df.isnull().sum() y muestra con display(nulls.head(30)) los valores nulos por columna. 

d. Usa display(df.describe()) para presentar la estadísticas descriptivas para las columnas numéricas.

**Resultado y Comentario**: El código revela que el **DataFrame df** tiene **3150 entradas** y **14 columnas**, sin valores nulos. 

Esto es crucial ya que confirma la integridad de los datos antes de cualquier modelado, evitando pasos de imputación.

## **1.4.3. Control de Tipo de Variables y Tratamiento de Nulos**

El código **df.columns** es una propiedad de los **DataFrames de Pandas** que permite acceder y visualizar los nombres de todas las columnas presentes en el DataFrame df.

**Resultado y Comentario**: La salida es un objeto **Index** que contiene la lista ordenada de todos los nombres de las columnas: ['fallos_llamadas', 'quejas', ..., 'abandono']. 

Esto es fundamental para verificar la estructura del DataFrame, confirmar que las columnas se han renombrado correctamente y referenciarlas en operaciones posteriores. 

Es una validación rápida y esencial de la integridad de los datos.

## **1.4.4.Definición Variables Numéricas y Categóricas**

a. Este código **clasifica las columnas** de df en **numéricas** (num_cols) y **categóricas** (cat_cols), excluyendo la variable objetivo. 

b. Realiza el **tratamiento de nulos**: imputa los valores faltantes en las columnas numéricas con la **mediana** (df[c].median()) y en las categóricas con la **moda** (df[c].mode()).

**Resultado y Comentario**: La validación post-imputación (df.isnull().sum()) muestra que **no quedan nulos**. 

Esto asegura que el dataset está limpio y listo para el preprocesamiento y modelado, sin interrupciones por datos faltantes.

## **1.5.1. Preparación Datos Modelado**

a. Este código es un paso **fundamental** en la preparación de datos para el modelado. Con **y = df['abandono'].astype('Int64')**, se extrae la columna **'abandono'** (la variable objetivo que queremos predecir) y se asegura que su tipo de dato sea Int64.

b. Luego, con **X = df.drop(columns=['abandono']),** se crea el DataFrame X que contiene todas las características (features) del dataset, excluyendo la variable objetivo **'abandono'.** 

Esto prepara los datos para que X sea el conjunto de entrada para el modelo y y la salida esperada.

**Resultado y Comentario**: En **X** ahora contiene **13 columnas** e **y** es una Serie con la variable objetivo. 

Esta separación es crucial antes de entrenar cualquier modelo de aprendizaje automático.

## **1.5.2. Inspeccion de Columnas Transformadas**

a.	Este código configura un **preprocesador** utilizando **ColumnTransformer** para transformar las características. 

b.	Se aplica **OneHotEncoder** a las **columnas categóricas** (cat_cols) para convertirlas en un formato numérico binario, evitando la **multicolinealidad** con drop='first'. 

c.	Las demás columnas son descartadas ** con(remainder='drop')**. 

d.	La transformación se aplica a **X**, creando **X_prepared**.

**Resultado y Comentario**: El **shape** original **(3150, 13)** se reduce a **(3150, 6)**. Esta transformación es clave para preparar los **datos categóricos** para modelos de aprendizaje automático que requieren entrada numérica, como la **regresión logística**. 

## **1.5.3. Recuperar Nombre Columnas Transformadas**

La reducción del número de columnas sugiere que solo las codificadas están siendo retenidas.

Este código se dearrolla de la siguiente manera:

a. Recupera los nombres de las columnas transformadas usando **preprocessor.get_feature_names_out()**.

b.  Luego, **convierte** el array resultante **X_prepared** en un **DataFrame de Pandas** (X_prepared_df). 

c. Se **limpian los prefijos** como cat__ de los nombres de las columnas para mayor claridad. 

d. Finalmente, se **imprimen las columnas finales** y se muestra una **muestra aleatoria** (X_prepared_df.sample(10)) del DataFrame transformado.

**Resultado y Comentario**: Muestra las columnas codificadas (grupo_edades_2, plan_tarifa_2, etc.) y un extracto de X_prepared_df.  

Es crucial para **verificar** que la codificación **OneHot** se aplicó correctamente y entender las nuevas características.

## **1.5.4. Deteccion de Multicolinealidad**

Este código busca **multicolinealidad** entre las variables numéricas usando el **Factor de Inflación de la Varianza (VIF)**. 

a.	Se seleccionan las **columnas numéricas (num_cols)**, 

b.	se eliminan las constantes y 

c.	se calcula el **VIF** para cada una con **variance_inflation_factor**. 

d.	Los resultados se ordenan y se muestran las 3 variables con el **VIF más alto** en **df_filt**.

Resultado y Comentario: El display(df_filt) muestra las variables con mayor VIF. 

Valores altos (usualmente >5 o >10) sugieren **fuerte multicolinealidad**, lo cual es crítico para la estabilidad y la interpretabilidad de los coeficientes en modelos lineales.

## **1.5.5. Extracción variables con VIF**

El código **df_filt['Variable'].tolist()** se utiliza para **extraer** los nombres de las variables con el **VIF (Factor de Inflación de la Varianza)** más alto (en este caso, el top 3) y convertirlos en una **lista de Python.**

**Resultado y Comentario**: La salida es **['valor_cliente', 'total_mensajes', 'total_llamadas']**.
Esta lista contiene las columnas identificadas con mayor multicolinealidad.  

Este paso es **crucial** para identificar qué variables tienen una fuerte relación lineal entre sí. 

Estas variables podrían ser consideradas para ser **excluidas o transformadas** en modelos predictivos, especialmente en aquellos sensibles a la multicolinealidad, para **mejorar la estabilidad y la interpretabilidad** del modelo.

## **1.5.6. Redefinición Dataframe segun variables con VIF**

El código **X = resultado[df_filt['Variable'].tolist()] redefine** el DataFrame **X** para que contenga **únicamente** las variables identificadas con **alta multicolinealidad** a través del análisis **VIF (df_filt['Variable'].tolist())**. 

Es decir, **X** ahora incluye solo **'valor_cliente', 'total_mensajes' y 'total_llamadas'** del DataFrame resultado.

**Resultado y Comentario**: **X** se convierte en un **DataFrame con 3 columnas** (las tres variables con mayor VIF). 
Este paso es **crucial** antes de entrenar modelos lineales, ya que el objetivo es **mitigar los efectos negativos** de la multicolinealidad, aunque también implica **descartar** otras variables relevantes que no tenían VIF alto.


# 1.6. ENTRENAMIENTO Y EVALUACION DE VARIABLES Y PROCESADAS

## 1.6.1. Entrenamiento y Evaluacion de Regresion Logistica

Este código prepara los datos para el **entrenamiento y evaluación de modelos** de aprendizaje automático. 

Se importan librerías clave como:

 **sklern.model_selection** para dividir los datos, 

**sklearn.linear_model.LogisticRegression** para el modelo, y 

**sklearn.metrics** junto con **seaborn y matplotlib.pyplot** para la evaluación y visualización de resultados. 

**train_test_split** esta función  divide **X** (características) e **y** (objetivo) en conjuntos de entrenamiento (80%) y prueba (20%), asegurando la **estratificación** (stratify=y) para mantener la proporción de clases en ambos conjuntos. 

**random_state** garantiza la reproducibilidad. 

Esto es crucial para evaluar objetivamente el rendimiento del modelo.

# **1.6. ENTRENAMIENTO Y EVALUACION DE VARIABLES Y PROCESADAS**

## **1.6.1. Entrenamiento y Evaluacion de Regresion Logistica**

Este código prepara los datos para el **entrenamiento y evaluación de modelos** de aprendizaje automático. 
Se importan librerías clave como:
** sklearn.model_selection** para dividir los datos, 

**sklearn.linear_model.LogisticRegression** para el modelo, y 

**sklearn.metrics, seaborn y matplotlib.pyplot** para la evaluación y visualización de resultados. 

**train_test_split** divide **X**(características) e **y** (objetivo) en conjuntos de entrenamiento (80%) y prueba (20%), asegurando la **estratificación (stratify=y)** para mantener la proporción de clases en ambos conjuntos. 

**random_state** garantiza la reproducibilidad. 
Esto es crucial para evaluar objetivamente el rendimiento del modelo.

## **1.6.2. Modelo De Regresión Logistica**

Este bloque de código realiza la **normalización de datos** utilizando **MinMaxScaler de sklearn.preprocessing**. 

a.	Se importa la clase y se crea una instancia del **escalador (normalización = MinMaxScaler())**.

b.	Se aplica el **ajuste y la transformación (fit_transform)** a **X_train**, que escala las características al rango [0, 1]. 
c.	El código convierte el array resultante **X_train** en un **DataFrame de Pandas** y muestra **5 muestras aleatorias (sample(5))** para inspeccionar los datos normalizados.
**Resultado y Comentario**: Las variables **valor_cliente, total_mensajes** y **total_llamadas** han sido escaladas a valores entre 0 y 1.
| valor_cliente | total_mensajes | total_llamadas |
|---------------|----------------|----------------|
| 1479          | 0.087361       | 0.01341        |
| 1115          | 0.105236       | 0.00000        |
| 297           | 0.422430       | 0.13410        |
| 1991          | 0.077902       | 0.04023        |
| 1247          | 0.014813       | 0.00000        |
 Este paso es crucial para algoritmos sensibles a la escala de las características, como la regresión logística, mejorando la convergencia y el rendimiento del modelo.

## **1.6.2. Modelo De Regresión Logistica**

Este bloque de código realiza la **normalización de datos** utilizando **MinMaxScaler de sklearn.preprocessing**. 

a.	Se importa la clase y se crea una instancia del **escalador (normalización = MinMaxScaler())**.

b.	Se aplica el **ajuste y la transformación (fit_transform)** a **X_train**, que escala las características al rango [0, 1]. 

c.	El código convierte el array resultante **X_train** en un **DataFrame de Pandas** y muestra **5 muestras aleatorias (sample(5))** para inspeccionar los datos normalizados.

**Resultado y Comentario**: Las variables **valor_cliente, total_mensajes** y **total_llamadas** han sido escaladas a valores entre 0 y 1.
| valor_cliente | total_mensajes | total_llamadas |
|---------------|----------------|----------------|
| 1479          | 0.087361       | 0.01341        |
| 1115          | 0.105236       | 0.00000        |
| 297           | 0.422430       | 0.13410        |
| 1991          | 0.077902       | 0.04023        |
| 1247          | 0.014813       | 0.00000        |

Este paso es crucial para algoritmos sensibles a la escala de las características, como la regresión logística, mejorando la convergencia y el rendimiento del modelo.

## **1.6.2. Entrenamiento Modelo con Regularizacion**

Este código entrena un modelo de **Regresión Logística (LogisticRegression)**. 

Se configura con **max_iter=1000**, **regularización l2 (penalty='l2')**, **optimizador lbfgs**, y **class_weight='balanced'** para manejar el desbalance de clases. 

a.	Tras el entrenamiento con **X_train** y **y_train**, se realizan predicciones **(yPrevisto) sobre X_test**. 

b.	Se evalúa el rendimiento del modelo mostrando un **classification_report (precisión, recall, f1-score)** y una **ConfusionMatrixDisplay** para visualizar los verdaderos positivos/negativos y falsos positivos/negativos.

**Resultado Comentario**: El modelo obtiene una **precisión** de 0.94 para la clase 0 y 0.29 para la clase 1, con un **recall** de 0.64 y 0.79 respectivamente. La **exactitud** general es del 0.66.
| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.94      | 0.64   | 0.76     | 531     |
| 1.0   | 0.29      | 0.79   | 0.42     | 99      |

| Métrica       | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      |           |        | 0.66     | 630     |
| Macro Avg     | 0.61      | 0.71   | 0.59     | 630     |
| Weighted Avg  | 0.84      | 0.66   | 0.71     | 630     |

Los resultados sugieren que el modelo es mejor identificando la **clase mayoritaria ('No abandono')** que la minoritaria **('Abandono')**, lo cual es común en datasets desbalanceados, a pesar de usar **class_weight='balanced'**.

### **1.6.2.1. Modelo Regresión Logistica Ridge**

Este código visualiza la importancia de las características de un modelo de **regresión logística Ridge**. 

a.	Calcula la **importancia (importanciamdRgLgRige)** a partir de los coeficientes del modelo y la normaliza. 

b.	Crea un DataFrame para ordenarlas y las 

c.	Muestra en un **gráfico de barras (sns.barplot)**. El gráfico tiene un título, etiquetas de ejes y los valores porcentuales de importancia sobre cada barra, facilitando la interpretación.

**Resultado y Comentario**: El gráfico de barras muestra la importancia relativa de cada característica **(valor_cliente, total_mensajes, total_llamadas)** en el **modelo Ridge**, con sus porcentajes.

INSERTAR IMAGEN

Esto ayuda a identificar qué variables tienen mayor impacto en la predicción del abandono, siendo crucial para la interpretabilidad del modelo.

**NOTA** : 
La razón por la que el **'Modelo de Regresión Logística'** y el **'Modelo de Regresión Logística Ridge'** muestran los mismos resultados de ¨precisión, recall y exactitud** es porque, al revisar el código, ambos modelos se inicializaron con la misma configuración: **LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs', class_weight='balanced')**.

En esencia, son el mismo modelo siendo entrenado y evaluado con los mismos datos, lo que lleva a métricas idénticas.

### **1.6.2.2. Modelo Regresión Logaritmica Lasso**

Este código entrena un modelo de **Regresión Logística Lasso(LogisticRegression)**. 

a.	Se configura con **regularización l1 (penalty='l1')**, el **solucionador liblinear** y **class_weight='balanced'** para manejar el desbalance de clases. 

b.	Tras el entrenamiento, se realizan predicciones **(yPrevisto)** sobre **X_test**. 

c.	El rendimiento se evalúa con un **classification_report (precisión, recall, f1-score)** y una **ConfusionMatrixDisplay**.

**Resultado y Comentario**: La **exactitud** general es del 0.67, con una **precisión** de 0.95 para la clase 0 y 0.30 para la clase 1. El **recall** es 0.65 (clase 0) y 0.81 (clase 1).

El **modelo Lasso** mejora ligeramente la **exactitud** y el **recall** para la **clase minoritaria ('Abandono')** en comparación con **Ridge**, lo que sugiere que la **regularización L1** puede ser más efectiva para la selección de características en este caso.


# **1.7. AJUSTE DE HIPERPARAMETROS**

## **1.7.1. HIPERPARAMETROS**

Este código realiza una **búsqueda de hiperparámetros (GridSearchCV)** para optimizar un modelo de **Regresión Logística**. 

a.	Define una **grilla de parámetros (param_grid)** para **C, penalty y solver**. 

b.	Configura **GridSearchCV** con **validación cruzada (cv=5), scoring='roc_auc'** y **class_weight='balanced'**. 

c.	Entrena la búsqueda con **grid.fit()**, imprime los mejores hiperparámetros y el ROC-AUC en validación. 

d.	Finalmente, evalúa el **best_model** en el conjunto de entrenamiento balanceado.

**Resultado y Comentario**:  El resultado esperado mostraría los mejores **hiperparámetros** encontrados y las métricas de rendimiento **(ROC-AUC, classification_report)** para el modelo óptimo.

Es una técnica clave para **afinar modelos** y mejorar su rendimiento general, especialmente importante en datasets desbalanceados.

## **1.7.1. Balanceo de datos con undersampling**

Este código implementa una técnica de **balanceo de clases** mediante **RandomUnderSampler** para abordar **datasets desbalanceados**. 
a.	Se inicializa el **undersampling** con **random_state=42** para reproducibilidad

b.	Se aplica **sampling_strategy='auto'** para igualar el número de muestras de la clase mayoritaria a la minoritaria. 


c.	Luego, se aplica **fit_resample** a **X_train** y **y_train** para crear **xTrainBal** y **yTrainBal**, los cuales contienen las clases balanceadas. 

d.	Finalmente, se imprimen las **distribuciones** de las clases balanceadas.

**Resultado y Comentario**: Se observa una distribución equilibrada con 396 muestras para cada clase (0 y 1), representando un 50% cada una.
Este balanceo es crucial para evitar que los modelos se sesguen hacia la clase mayoritaria y mejoren su capacidad para predecir la clase minoritaria.

## **1.7.2. Modelo Regresión Logistica Lasso Con Undersampling**

Este código entrena un modelo de **Regresión Logística** (`LogisticRegression`) con **undersampling**. 

a. Se configura con **regularización `l2` (`penalty='l2'`)**, el **solucionador `liblinear`** y **`class_weight='balanced'`**, además de un parámetro `C=1`. 

b. Tras el entrenamiento en los datos balanceados (`xTrainBal`, `yTrainBal`), se realizan **predicciones** (`y_pred_best`) sobre `X_test` usando el modelo `log_reg_lasso`.  

c. El rendimiento se evalúa con un **`classification_report`** y una **`ConfusionMatrixDisplay`**.

**Resultado y Comentario**:La **exactitud** general es del 0.67, con una **precisión** de 0.95 (clase 0) y 0.30 (clase 1). El **recall** es 0.65 (clase 0) y 0.81 (clase 1).

Aunque la variable del modelo se llama **`log_reg_lassoUnder`**, la **penalización utilizada es `l2` (Ridge)**, lo que resulta en un comportamiento similar al modelo Ridge, pero con datos balanceados por undersampling.

## **1.7.3. Modelo Regresión Logística Ridge Con Undersampling**

Este código entrena un modelo de **Regresión Logística Ridge (LogisticRegression)** con datos balanceados por **undersampling**. 

a. Se configura con **regularización l2 (penalty='l2')**, el **optimizador lbfgs**, y **class_weight='balanced'** para manejar el desbalance de clases. 

b. Tras el entrenamiento en los **datos balanceados (xTrainBal, yTrainBal)**, se realizan predicciones **(yPrevisto)** sobre **X_test**. 

c. El rendimiento se evalúa con un **classification_report y una ConfusionMatrixDisplay**

**Resultado y Comentario**: El modelo obtiene una **precisión** de 0.94 para la clase 0 y 0.29 para la clase 1, con un **recall** de 0.64 y 0.79 respectivamente. La **exactitud** general es del 0.66.

Los resultados son idénticos a los del modelo de **Regresión Logística original con class_weight='balanced'**, lo que refuerza que la **regularización L2** ya estaba aplicada y el balanceo por **undersampling** no alteró significativamente las métricas en este caso particular, o el efecto de **class_weight** en el modelo original ya mitigaba el desbalance.


## **1.8.*Evalúación  exactitud de varios modelos de regresión logística entrenados** 

Este código **evalúa la exactitud** de varios modelos de regresión logística entrenados. 

a. Se define una **lista** que contiene tuplas, cada una con el **nombre del modelo**, el **objeto modelo entrenado** y los **datos de prueba (X_test)**. 

b. Un bucle **for** itera sobre esta lista, y para cada modelo, calcula y 

c. **muestra la exactitud (model.score(X_test, y_test))** en el conjunto de prueba.

**Resultado y Comentario**: La salida muestra la exactitud de cada uno de los seis modelos evaluados, por ejemplo, 'Modelo de regresión logistica es: 0.6603'.
 Este bloque es crucial para comparar rápidamente el rendimiento de los diferentes modelos en términos de exactitud, lo que ayuda a identificar cuál es el más efectivo para predecir el abandono de clientes.

## Documentación


