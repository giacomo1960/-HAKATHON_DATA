# 🚀 ChurnInsight (MVP)

## 📌 Descripción General

**ChurnInsight** es un MVP desarrollado durante el *Hackathon Oracle ONE* cuyo objetivo es **predecir la probabilidad de cancelación (churn) de clientes** en negocios basados en suscripción (fintech, telecomunicaciones, streaming, e‑commerce, etc.).

La solución completa combina:
* 📊 **Ciencia de Datos** (Python) para entrenar un modelo predictivo.
* ⚙️ **Backend** (Java + Spring Boot) para exponer la predicción mediante una API REST.
* 🎨 **Frontend** (Next.js) para visualizar las métricas y probar el modelo (Mock).

https://github.com/giacomo1960/-HAKATHON_DATA/blob/main/GRAFICO%20N%C2%B0%201%20VENTAS%20TOTALES.pdf

## 📂 Estructura del Repositorio

```
churn-insight/
├── app/               # Código fuente del Frontend (Next.js App Router)
├── backend/           # Código fuente del Backend (Java Spring Boot)
├── DataScience/       # Notebooks, Scripts Python y Modelo serializado
├── components/        # Componentes UI reutilizables
├── ...
└── README.md          # Documentación del proyecto
```

---

## 🎯 Problema que resolvemos

Las empresas pierden dinero cuando los clientes cancelan sus servicios. **ChurnInsight** ayuda a:
1.  **Identificar clientes en riesgo** antes de que se vayan.
2.  **Entender por qué** podrían cancelar.
3.  **Actuar a tiempo** para retenerlos.

---

## ⚙️ Backend (API REST)

El Backend está desarrollado en Java 17 con Spring Boot 3.2.3. Expone una API REST que integra el modelo de Machine Learning desarrollado en Python.

### 🔹 Requisitos Previos

*   Java 17 JDK
*   Maven
*   Python 3.12+
*   Dependencias de Python (instalar con `pip install -r DataScience/requirements.txt`)

### 🔹 Ejecución Local

1.  **Instalar dependencias de Python:**
    ```bash
    pip install -r DataScience/requirements.txt
    ```

2.  **Compilar y ejecutar el Backend:**
    Desde la carpeta raíz o `backend/`:
    ```bash
    cd backend
    ./mvnw spring-boot:run
    ```
    La API estará disponible en `http://localhost:8080`.

### 🔹 Contrato de API: Endpoint `/predict`

**Método:** `POST`
**URL:** `http://localhost:8080/predict`

#### 📥 Input (Request JSON)
```json
{
    "tiempo_contrato_meses": 12,
    "retrasos_pago": 2,
    "uso_mensual": 14.5,
    "plan": "Premium"
}
```

#### 📤 Output (Response JSON)
```json
{
    "prevision": "Va a cancelar",
    "probabilidad": 0.8838
}
```

---

## 📊 Ciencia de Datos

El modelo de predicción se encuentra en la carpeta `DataScience`.
*   **Modelo:** `joblib.dump` (Random Forest Classifier).
*   **Script de Integración:** `api.py` (Usado por el backend de Java).

---

## 🎨 Frontend

El frontend es una aplicación Next.js.
*Nota: Actualmente el frontend utiliza un mock para las predicciones y no está conectado al backend real debido a diferencias en el contrato de datos.*

Para ejecutarlo:
```bash
npm install
npm run dev
```
Acceder a `http://localhost:3000`.

---

## 👥 Equipo
Proyecto para el **Oracle ONE Hackathon**.
*   **Frontend**: Next.js Team
*   **Backend**: Java/Spring Team
*   **Data Science**: Python/ML Team

