# Plan de Trabajo y Distribución de Tareas - ChurnInsight

Este documento guía el desarrollo del MVP de **ChurnInsight**, adaptado a la estructura del equipo: **Data Science** (Puro) y **Full Stack** (Backend + Frontend).

---

## 📋 1. Análisis de Brechas (Gap Analysis)

| Área | Estado Actual | Tareas Faltantes (Críticas) |
| :--- | :--- | :--- |
| **Data Science** | ❌ **Pendiente**<br>Solo existe el dataset. | 1. **EDA & Training:** Limpieza, Análisis y Entrenamiento.<br>2. **Modelo Final:** Exportación del artefacto (`.joblib`/`.pkl`) probado.<br>3. **Script de Inferencia:** Función Python limpia para predecir nuevos datos. |
| **Ingeniería (Full Stack)** | ⚠️ **Parcial**<br>Next.js iniciado. Spring Boot iniciado (Mock). | 1. **Integración:** Conectar Frontend <-> Backend Java <-> Modelo Python.<br>2. **API Serving:** Crear el servicio que expone el modelo (ya que DS no hace backend).<br>3. **UX/UI:** Finalizar interfaz y manejo de errores. |

---

## 👥 2. Distribución de Tareas (Equipo de 10)

### 🧠 Equipo Data Science (5 Personas)
*Misión: Crear el modelo predictivo más preciso posible y entregar el artefacto listo para usar.*
*Nota: Este equipo NO toca código de API ni servidores, se enfoca 100% en los datos.*

1.  **DS-1: Data Engineering & Cleaning**
    *   **Tarea:** Cargar `TelecomX_Data.json`. Tratar valores nulos (ej: `TotalCharges` vacíos), convertir tipos de datos y aplanar el JSON a formato tabular.
    *   **Entregable:** Un DataFrame limpio guardado como CSV para que el resto trabaje.
2.  **DS-2: Feature Engineering**
    *   **Tarea:** Analizar correlaciones. Crear nuevas variables (ej: "Tenure en años", "Ratio cobro/servicio"). Seleccionar las 10-15 columnas más importantes.
    *   **Entregable:** Lista definitiva de *inputs* requeridos.
3.  **DS-3: Entrenamiento de Modelos**
    *   **Tarea:** Probar algoritmos (Logistic Regression, Random Forest, XGBoost). Usar *Cross-Validation* para asegurar que el modelo no memorice los datos.
    *   **Entregable:** El modelo con mejor métrica seleccionado.
4.  **DS-4: Evaluación y Métricas**
    *   **Tarea:** Generar matriz de confusión, curvas ROC y reporte de métricas (Accuracy, Recall, F1). Explicar qué variables pesan más.
    *   **Entregable:** Reporte de rendimiento para el README.
5.  **DS-5: Serialización y Entrega (Handover)**
    *   **Tarea:** Empaquetar el modelo final (`model.joblib`) y el pipeline de preprocesamiento (escaladores, encoders).
    *   **Crítico:** Escribir un script simple `predict.py` que reciba un diccionario y devuelva la predicción. Esto es lo que usará el equipo Full Stack.

---

### 💻 Equipo Full Stack (5 Personas)
*Misión: Construir la plataforma Web (Next.js) y la API (Java) que utiliza el modelo.*

1.  **FS-1: Arquitecto de Solución y API Python (Serving)**
    *   **Rol:** Puente entre DS y Web.
    *   **Tarea:** Tomar el script de DS-5 y envolverlo en una micro-API rápida (FastAPI/Flask) o investigar cómo cargar el modelo ONNX directamente en Java.
    *   **Objetivo:** Que el backend Java tenga a quién preguntarle la predicción.
2.  **FS-2: Backend Java - Lógica de Negocio**
    *   **Tarea:** Implementar el `ChurnService.java` en Spring Boot. Consumir la API de predicción (creada por FS-1).
    *   **Objetivo:** Orquestar la llamada: Recibe de Frontend -> Valida -> Llama Modelo -> Retorna resultado.
3.  **FS-3: Backend Java - Validación y Seguridad**
    *   **Tarea:** Definir los DTOs (`CustomerRequest`) basándose estrictamente en los inputs definidos por DS-2. Implementar validaciones `@NotNull`, `@Min`.
    *   **Objetivo:** Proteger el sistema de datos basura.
4.  **FS-4: Frontend - Formulario e Integración**
    *   **Tarea:** Construir el formulario en Next.js. Crear el servicio de conexión con la API Java (`lib/api.ts`).
    *   **Objetivo:** Que el botón "Predecir" funcione realmente.
5.  **FS-5: Frontend - UI/UX y Dashboard**
    *   **Tarea:** Diseñar la visualización de la respuesta (Medidor de riesgo). Si hay tiempo, hacer el Dashboard de estadísticas (`/stats`).
    *   **Objetivo:** Que la aplicación se vea profesional y amigable.

---

## 🔄 Flujo de Trabajo Recomendado

1.  **Día 1 (Definición):**
    *   **DS-2 y FS-3** se reúnen para definir el JSON de entrada ("Contrato").
    *   *Ejemplo:* `{"age": int, "salary": float, ...}`.
    *   Si esto cambia después, rompe todo. ¡Definirlo bien al principio!

2.  **Día 2-3 (Desarrollo Paralelo):**
    *   **Equipo DS:** Trabaja en sus Notebooks.
    *   **Equipo FS:** Crea el Frontend y el Backend Java usando un "Mock" (datos falsos) mientras espera el modelo real.

3.  **Día 4 (Integración):**
    *   **DS** entrega el archivo `.joblib` y el script `predict.py`.
    *   **FS-1** crea el contenedor con el modelo.
    *   **FS-2** conecta Java al contenedor del modelo.

4.  **Día 5 (Pruebas):**
    *   Probar el flujo completo: Frontend -> Java -> Python Model -> Java -> Frontend.

## 🛠️ Stack Tecnológico Final

*   **Frontend:** Next.js (React), Tailwind CSS.
*   **Backend Principal:** Java 17, Spring Boot 3.
*   **Motor IA:** Python 3.10, Scikit-learn, FastAPI (gestionado por FS-1).
*   **Base de Datos (Opcional):** H2 (Embebida) o PostgreSQL.
