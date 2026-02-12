# Math to Image: Sistema Web Bidireccional de Arte Matemático

Este proyecto es un sistema interactivo que explora la relación entre imágenes y fórmulas matemáticas. Permite generar arte visual a partir de ecuaciones paramétricas y, a la inversa, aproximar imágenes existentes mediante fórmulas matemáticas explícitas de alta precisión.

## Características

- **Fórmula ⇄ Imagen**: Generación determinista y reconstrucción algorítmica.
- **Arte Paramétrico**: Basado en curvas paramétricas 2D y series trigonométricas.
- **Optimización de Error**: Minimización del error de reconstrucción mediante algoritmos evolutivos y regresión no lineal.
- **Interfaz Técnica**: Visualización comparativa de original vs. reconstruido con métricas de error.

## Estructura del Proyecto

- `backend/`: API construida con FastAPI, lógica matemática (NumPy, SymPy, SciPy) y procesamiento de imagen (scikit-image, Pillow).
- `frontend/`: Interfaz web minimalista (HTML5 Canvas/JS).
- `docs/`: Documentación adicional y recursos.

## Instalación y Uso

### Backend

1. Navega al directorio del backend:
   ```bash
   cd backend
   ```
2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Inicia el servidor:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend

Simplemente abre `frontend/index.html` en tu navegador para interactuar con el sistema (requiere que el backend esté en ejecución para las funcionalidades de ajuste).

## Pruebas y Benchmarks

El proyecto incluye una suite completa de pruebas unitarias y de integración, así como benchmarks de rendimiento.

Para ejecutar todo:
```bash
cd backend
bash run_all_tests.sh
```

### Despliegue

El proyecto se puede desplegar en cualquier servidor web que soporte Python y FastAPI.

Para desplegar el proyecto en Render.com, sigue los siguientes pasos:

1. Crea una cuenta en Render.com si no tienes una.
2. Clona el repositorio en tu computadora.
3. Ve al panel de control de Render.com y selecciona "New" -> "Web Service".
4. En la configuración, selecciona "GitHub" como fuente de código.
5. Selecciona el repositorio que clonaste.
6. Configura el entorno de despliegue:
   - Lenguaje: Python
   - Rama: main
   - Comando de despliegue: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
7. Guarda y despliega.

## 🌐 Demo en Producción

- 🖼️ Frontend:
  https://math-to-image-frontend.onrender.com

## 📚 Documentación

- 📝 API (Swagger):
  https://math-to-image.onrender.com/docs

- 📚 Documentación de la API:
  https://math-to-image.onrender.com/docs

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
