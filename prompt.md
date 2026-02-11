# PROMPT

## **Título**

Sistema Web Bidireccional de Arte Matemático
**Imagen ⇄ Fórmula Matemática Generativa (explicable, optimizada y de error mínimo)**

---

## **Contexto actual**

Se requiere diseñar y documentar un **sistema web interactivo** de arte matemático algorítmico, inspirado en enfoques como Hamid Naderi Yeganeh, donde:

* **Toda imagen es generada exclusivamente por fórmulas matemáticas explícitas**.
* El sistema opera en **dos direcciones**:

  1. **Fórmula → Imagen**
  2. **Imagen → Fórmula matemática aproximada**
* La prioridad absoluta es **minimizar el error de reconstrucción**, incluso si las fórmulas resultantes son largas o complejas.
* La legibilidad humana es deseable pero **secundaria** frente a la precisión matemática.
* Se permite y fomenta el uso de **aceleración por hardware** (CPU vectorizada, multihilo, GPU si está disponible), siempre que:

  * El resultado final sea una **fórmula explícita**.
  * No se usen modelos generativos tipo “caja negra” para producir la imagen final.

### Fundamentos matemáticos permitidos

* Trigonometría (`sin`, `cos`, `tan`, funciones compuestas).
* Curvas paramétricas 2D.
* Series finitas (Fourier truncado, combinaciones trigonométricas).
* Transformaciones geométricas explícitas.
* Composición funcional profunda (sin límite rígido de términos).

### Stack técnico requerido

* **Frontend**: HTML5 + Canvas y/o SVG (JavaScript moderno, sin frameworks pesados).
* **Backend**: Python (FastAPI o Flask).
* **Procesamiento matemático**: NumPy, SymPy, SciPy.
* **Procesamiento de imagen**: scikit-image, Pillow.
* **Optimización**:

  * Regresión no lineal (LM, BFGS).
  * Algoritmos evolutivos (Differential Evolution).
  * Uso opcional de GPU (CUDA / OpenCL) si está disponible.
* **Licencias**: solo dependencias con licencias permisivas (MIT, BSD, Apache 2.0).

---

## **Objetivos y alcance de la tarea**

### **1. Módulo Fórmula → Imagen**

**Entrada**

* Fórmulas paramétricas 2D:
  `x(t), y(t)` en formato **SymPy evaluable**.
* Dominio de `t`.
* Resolución de salida.
* Soporte completo para **color** (RGB, funciones dependientes de `t`).

**Proceso**

* Validación sintáctica y semántica (dominios, singularidades).
* Evaluación eficiente (vectorización, paralelismo si procede).
* Render matemático determinista.

**Salida**

* Imagen PNG y/o SVG.
* Fórmula equivalente en **LaTeX + SymPy**.
* Metadatos matemáticos (dominio, parámetros, hash reproducible).

---

### **2. Módulo Imagen → Fórmula (aproximación de alta precisión)**

**Entrada**

* Imagen raster (PNG/JPG), a color.
* Presupuesto computacional configurable (tiempo, iteraciones).
* Métrica objetivo prioritaria (L2, IoU, u otra).

**Pipeline obligatorio**

1. Normalización y separación de canales de color.
2. Detección de bordes y contornos por canal.
3. Vectorización de curvas dominantes.
4. Ajuste de funciones paramétricas usando:

   * Series trigonométricas truncadas.
   * Composición funcional profunda si reduce el error.
   * Optimización híbrida (global + local).
5. Uso de hardware acelerado si está disponible.

**Criterio rector**

> Minimizar el error **a toda costa**, incluso si la fórmula final es larga o poco “elegante”.

**Salida**

* Fórmula matemática aproximada (LaTeX + SymPy).
* Imagen reconstruida.
* Métricas cuantitativas claras (error absoluto y relativo).
* Reporte de complejidad matemática (número de términos, profundidad).

---

### **3. Arquitectura del sistema**

* Separación estricta frontend / backend.
* API REST documentada:

  * `POST /render-formula`
  * `POST /fit-image`
* Contratos JSON claros, reproducibles y versionados.
* Código documentado con **énfasis matemático**, no solo técnico.

---

### **4. Interfaz de usuario**

* UI técnica, minimalista.
* Entrada clara de fórmulas (texto SymPy).
* Carga de imágenes a color.
* Visualización comparativa:

  * Imagen original
  * Imagen reconstruida
  * Diferencia visual (error)

---

## **Entregables esperados**

1. **Arquitectura general del sistema**

   * Diagrama lógico de componentes.
2. **Documento matemático**

   * Qué clases de imágenes son reconstruibles.
   * Qué no lo son y por qué (límites teóricos).
3. **Backend completo**

   * Procesamiento de imagen.
   * Ajuste matemático.
   * Generación de fórmulas.
   * Soporte opcional de GPU.
4. **Frontend funcional**

   * Render matemático Canvas/SVG.
   * UI comparativa.
5. **Ejemplos reproducibles**

   * Fórmulas → imágenes complejas.
   * Imágenes a color → fórmulas aproximadas.
6. **Pruebas automáticas**

   * Unitarias y de integración (pytest).
   * Casos de referencia con métricas objetivo.
7. **Limitaciones técnicas**

   * Coste computacional.
   * Casos patológicos.
8. **Ideas de extensión**

   * 3D.
   * Animación paramétrica.
   * Exportación SVG matemático puro.

---

## **Formato requerido de la respuesta**

* Lenguaje técnico, preciso y sin ambigüedades.
* Secciones numeradas y viñetas claras.
* Código **real y funcional** (no pseudocódigo).
* Comentarios que expliquen decisiones matemáticas.
* Advertencias explícitas cuando algo sea una aproximación.
* Incluir scripts de prueba y benchmark reproducibles.

---

## **Criterio de éxito**

El sistema se considera exitoso si:

* Reconstruye imágenes a color con **error mínimo cuantificado**.
* Produce **fórmulas matemáticas explícitas**, aunque sean largas.
* Es reproducible, explicable y computacionalmente optimizable.

---