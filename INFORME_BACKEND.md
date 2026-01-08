# Informe para el Equipo de Backend

## Estado Actual y Hoja de Ruta (Roadmap)

Este documento detalla las tareas pendientes y la guía técnica para completar el desarrollo del backend de **ChurnInsight**, basándose en los requerimientos del MVP y las funcionalidades opcionales.

### 1. Estado Actual
*   **Proyecto Inicializado:** Java 17 + Spring Boot 3.2.3.
*   **Estructura Base:** Controlador `ChurnController` creado.
*   **Endpoint:** `POST /predict` implementado como *Mock* (devuelve datos fijos).

### 2. Tareas Críticas para el MVP (Prioridad Alta)

Para cumplir con los entregables mínimos, el equipo debe enfocarse en:

#### A. Integración con Data Science
El modelo predictivo correrá en un entorno de Python. El backend Java debe comunicarse con él.
*   **Estrategia Recomendada:** Arquitectura de Microservicios.
    *   *Python:* Expondrá el modelo vía API (FastAPI/Flask) en un puerto (ej: 5000).
    *   *Java:* Consumirá ese servicio usando `RestTemplate` o `WebClient`.
*   **Tarea:** Implementar un servicio (`ChurnService`) que haga la petición HTTP al microservicio de Python enviando los datos del cliente y recibiendo la predicción.

#### B. Validación de Datos (Input Validation)
Actualmente el endpoint acepta cualquier JSON. Se debe asegurar la calidad de los datos.
*   **Tarea:**
    *   Crear un DTO (Data Transfer Object) para la entrada (ej: `CustomerRequest.java`).
    *   Usar anotaciones de validación (`@NotNull`, `@Min`, `@NotBlank`) en los campos.
    *   Ejemplo: `tiempo_contrato_meses` no puede ser negativo.

#### C. Manejo de Errores
La API debe responder con códigos HTTP adecuados y mensajes claros.
*   **Tarea:** Implementar un `GlobalExceptionHandler` (`@ControllerAdvice`).
    *   400 Bad Request: Cuando faltan datos o son inválidos.
    *   500 Internal Server Error: Si el servicio de Python no responde.

#### D. Configuración
*   **Tarea:** Crear `application.properties` (o `.yml`) para configurar la URL del servicio de Data Science, permitiendo cambiarla fácilmente entre entornos (local/prod).

### 3. Funcionalidades Opcionales (Mejora del Proyecto)

Una vez asegurado el MVP, se pueden agregar estas características para destacar en el Hackathon:

*   **Persistencia (Base de Datos):**
    *   Guardar cada predicción realizada para análisis futuro.
    *   *Tecnología:* H2 (memoria) o PostgreSQL.
    *   *Entidad:* `PredictionRecord` (id, fecha, input_data, resultado, probabilidad).
*   **Endpoint de Estadísticas (`GET /stats`):**
    *   Devolver métricas simples consultando la base de datos (ej: "¿Cuántos clientes hemos analizado hoy?", "Tasa de churn predicha promedio").
*   **Documentación API:**
    *   Integrar **Swagger/OpenAPI** (`springdoc-openapi`) para tener documentación interactiva automática.
*   **Containerización:**
    *   Crear un `Dockerfile` para empaquetar la aplicación Java.

### 4. Resumen de Endpoints a Desarrollar

| Verbo | Endpoint | Estado Actual | Acción Necesaria |
| :--- | :--- | :--- | :--- |
| `POST` | `/predict` | **Mock** | Conectar con lógica real y validaciones. |
| `GET` | `/stats` | *No existe* | Crear (Opcional). |
| `GET` | `/health` | *No existe* | Crear (Actuator) para verificar estado. |

---
**Siguiente Paso Sugerido:**
Definir junto al equipo de Data Science el contrato exacto (JSON Schema) que tendrá el microservicio de Python para empezar a programar el cliente HTTP en Java.
