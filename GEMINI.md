# Math to Image: Instructional Context

Este proyecto es un sistema interactivo bidireccional que explora la relación entre imágenes y fórmulas matemáticas, permitiendo la generación de arte paramétrico y la reconstrucción de imágenes mediante series de Fourier.

## 🚀 Vista General del Proyecto

- **Propósito**: Generar imágenes a partir de fórmulas paramétricas $x(t), y(t)$ y, a la inversa, aproximar contornos de imágenes mediante fórmulas explícitas.
- **Arquitectura**: 
  - **Backend**: FastAPI (Python) para procesamiento matemático pesado y servicios de API.
  - **Frontend**: Interfaz web minimalista basada en HTML5 Canvas y Vanilla JavaScript.
- **Tecnologías Clave**:
  - **NumPy**: Evaluación vectorial de alto rendimiento.
  - **SciPy**: Optimización no lineal (Differential Evolution + L-BFGS-B).
  - **SymPy**: Manipulación simbólica, validación de fórmulas y conversión a LaTeX.
  - **scikit-image & Pillow**: Procesamiento y extracción de contornos de imágenes.

## 🛠 Comandos de Construcción y Ejecución

### Backend
1. **Instalación**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Ejecución (Desarrollo)**:
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Pruebas**:
   ```bash
   bash backend/run_all_tests.sh
   # O individualmente:
   pytest backend/tests/
   ```

### Frontend
- No requiere compilación. Abrir `frontend/index.html` directamente en el navegador. Asegurarse de que el backend esté corriendo en `http://localhost:8000`.

## 📐 Convenciones de Desarrollo

- **Rendimiento**: Se prioriza la vectorización con NumPy sobre bucles de Python. El renderizado utiliza *adaptive sampling* para mejorar la calidad en zonas de alta curvatura.
- **Seguridad**: Todas las fórmulas ingresadas por el usuario pasan por `FormulaValidator` (en `backend/app/models/validation.py`) que usa SymPy para evitar inyección de código y limitar la complejidad del AST.
- **Algoritmos de Ajuste**: 
  1. Extracción de contorno principal.
  2. Ajuste inicial mediante Coeficientes de Fourier (FFT).
  3. Refinamiento mediante optimización no lineal para minimizar el error L2.
- **Documentación**: Consultar `docs/mathematical_limits.md` para entender las limitaciones teóricas (ej. curvas paramétricas no pueden rellenar áreas ni representar formas no conexas).

## 📂 Estructura de Archivos Clave

- `backend/app/main.py`: Punto de entrada de la API y definición de endpoints.
- `backend/app/services/renderer.py`: Motor de renderizado de curvas paramétricas.
- `backend/app/services/contour_fitter.py`: Lógica de ajuste de imagen a fórmula.
- `backend/app/models/validation.py`: Validación y seguridad de expresiones matemáticas.
- `frontend/js/api_client.js`: Cliente para comunicación con el backend.
- `frontend/js/canvas_renderer.js`: Manejo de la visualización en el cliente.
