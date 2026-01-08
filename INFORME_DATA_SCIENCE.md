# Informe para el Equipo de Data Science

## Análisis del Dataset `TelecomX_Data.json`

Este informe resume el análisis técnico del dataset proporcionado, confirmando su viabilidad para el proyecto "ChurnInsight" y detallando los pasos necesarios para su procesamiento.

### 1. Viabilidad del Dataset

El dataset es **altamente viable y recomendado** para el objetivo del MVP. Cumple con todos los requisitos fundamentales:

*   **Sector:** Telecomunicaciones (coincide con el ejemplo del negocio).
*   **Target:** Variable `Churn` ("Yes"/"No") presente y lista para clasificación binaria.
*   **Riqueza de Datos:** Contiene variables predictivas clave en múltiples dimensiones:
    *   **Demografía:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    *   **Comportamiento:** `tenure` (antigüedad), `Contract` (tipo de contrato).
    *   **Servicios:** `PhoneService`, `InternetService`, `TechSupport`, `StreamingTV`, etc.
    *   **Financiero:** `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, `PaperlessBilling`.

*Nota:* Este dataset sigue la estructura del famoso "Telco Customer Churn", un estándar en la industria para este tipo de problemas.

### 2. Acciones Requeridas (Data Cleaning & Preprocessing)

El equipo de Data Science debe prestar atención a los siguientes puntos críticos detectados durante la validación:

#### A. Limpieza de Datos
*   **Valores Faltantes en Target:** Se ha reportado la posible existencia de registros donde `"Churn": ""`.
    *   *Acción:* Verificar y eliminar estos registros, ya que no sirven para el entrenamiento supervisado.
*   **Tipos de Datos (Casting):**
    *   El campo `TotalCharges` (dentro del objeto `account.Charges`) viene como **string** (ej: `"367.55"`).
    *   *Acción:* Convertir explícitamente a **float/numeric** antes del análisis. Manejar posibles cadenas vacías que representen valores nulos.

#### B. Transformación de Estructura
*   **Aplanamiento (Flattening):**
    *   El JSON actual tiene una estructura anidada (ej: `customer.gender`, `account.Charges.Total`).
    *   *Acción:* Para entrenar modelos (scikit-learn), es necesario "aplanar" el dataset a una estructura tabular (DataFrame).
    *   *Ejemplo:* `customer.gender` -> `customer_gender`.

### 3. Recomendación de Trabajo

1.  Cargar el JSON usando `pandas.read_json` y aplicar `json_normalize` si es necesario para el aplanamiento inicial.
2.  Realizar el casting de `TotalCharges` a numérico (`pd.to_numeric`, con `errors='coerce'`).
3.  Eliminar filas con `Churn` nulo o vacío.
4.  Proceder con el EDA (Análisis Exploratorio de Datos) y la ingeniería de características habitual (One-Hot Encoding para variables categóricas).

---
**Estado del Backend:**
El proyecto Spring Boot ha sido inicializado en la carpeta `backend` y está listo para recibir el modelo (o conectarse a él) una vez que esté disponible.
