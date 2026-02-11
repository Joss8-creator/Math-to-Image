# backend/app/main.py
"""
API REST para sistema de arte matemático.

ENDPOINTS:
- POST /api/render: Fórmula → Imagen
- POST /api/validate: Validar fórmula sin renderizar
- GET /api/examples: Fórmulas de ejemplo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from app.models.validation import FormulaValidator
from app.services.renderer import ParametricRenderer, RenderConfig
from app.services.contour_fitter import ContourFitter

app = FastAPI(
    title="Mathematical Art API",
    description="Sistema bidireccional de arte matemático",
    version="1.0.0"
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servicios globales
validator = FormulaValidator()
renderer = ParametricRenderer()
fitter = ContourFitter()

# Modelos de datos
class FormulaInput(BaseModel):
    x_formula: str = Field(..., description="Fórmula paramétrica x(t)")
    y_formula: str = Field(..., description="Fórmula paramétrica y(t)")
    t_min: float = Field(0.0, description="Valor mínimo de t")
    t_max: float = Field(6.28318, description="Valor máximo de t (2π por defecto)")
    resolution: int = Field(800, ge=100, le=4096, description="Resolución de imagen")
    num_points: int = Field(10000, ge=100, le=100000, description="Número de puntos")
    color: Optional[List[int]] = Field([0, 0, 0], description="Color RGB")

class FitInput(BaseModel):
    image_base64: str
    num_terms: int = 10
    iterations: int = 500

class FormulaOutput(BaseModel):
    success: bool
    image_base64: Optional[str] = None
    formula_latex: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

@app.post("/api/render", response_model=FormulaOutput)
async def render_formula(input: FormulaInput):
    """
    Renderiza fórmulas paramétricas a imagen.
    
    Ejemplo:
    ```json
    {
        "x_formula": "cos(5*t) * cos(t)",
        "y_formula": "cos(5*t) * sin(t)",
        "t_min": 0,
        "t_max": 6.28318,
        "resolution": 800
    }
    ```
    """
    try:
        # Validar fórmulas
        x_expr, x_valid, x_error = validator.validate_and_parse(input.x_formula)
        if not x_valid:
            raise HTTPException(status_code=400, detail=f"Error en x(t): {x_error}")
        
        y_expr, y_valid, y_error = validator.validate_and_parse(input.y_formula)
        if not y_valid:
            raise HTTPException(status_code=400, detail=f"Error en y(t): {y_error}")
        
        # Compilar a funciones NumPy
        x_func = validator.to_numpy_function(x_expr)
        y_func = validator.to_numpy_function(y_expr)
        
        # Configurar renderizado
        config = RenderConfig(
            width=input.resolution,
            height=input.resolution,
            t_min=input.t_min,
            t_max=input.t_max,
            num_points=input.num_points,
            color=tuple(input.color)
        )
        
        # Renderizar
        image_array = renderer.render_curve(x_func, y_func, config)
        
        # Convertir a base64
        pil_image = Image.fromarray(image_array)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # LaTeX
        import sympy as sp
        latex_x = sp.latex(x_expr)
        latex_y = sp.latex(y_expr)
        
        return FormulaOutput(
            success=True,
            image_base64=image_base64,
            formula_latex=f"x(t) = {latex_x}\\\\y(t) = {latex_y}",
            metadata={
                "points_rendered": input.num_points,
                "resolution": f"{input.resolution}x{input.resolution}",
                "domain": f"t ∈ [{input.t_min}, {input.t_max}]"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/examples")
async def get_examples():
    """Retorna fórmulas de ejemplo (estilo Yeganeh)."""
    return {
        "examples": [
            {
                "name": "Rosa de 5 pétalos",
                "x": "cos(5*t) * cos(t)",
                "y": "cos(5*t) * sin(t)",
                "t_min": 0,
                "t_max": 6.28318
            },
            {
                "name": "Espiral de Arquímedes",
                "x": "t * cos(t)",
                "y": "t * sin(t)",
                "t_min": 0,
                "t_max": 12.56637
            },
            {
                "name": "Lissajous 3:2",
                "x": "sin(3*t)",
                "y": "sin(2*t)",
                "t_min": 0,
                "t_max": 6.28318
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
