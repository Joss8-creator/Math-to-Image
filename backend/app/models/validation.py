# backend/app/models/validation.py
"""
Validador de fórmulas matemáticas con límites de seguridad.
Objetivo: Prevenir inyección de código y validar dominio matemático.
"""

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Tuple, Optional
import numpy as np

class FormulaValidator:
    """
    Valida y prepara fórmulas paramétricas para evaluación segura.
    
    Restricciones de seguridad:
    - Solo funciones matemáticas permitidas (sin eval/exec)
    - Límite de complejidad (profundidad AST < 100)
    - Sin importaciones ni llamadas a sistema
    """
    
    ALLOWED_FUNCTIONS = {
        'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
        'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan',
        'floor', 'ceil', 'sign', 'pi', 'E'
    }
    
    MAX_COMPLEXITY = 100  # Profundidad máxima del árbol de expresión
    
    def __init__(self):
        self.t = sp.Symbol('t', real=True)
    
    def validate_and_parse(
        self, 
        formula_str: str,
        param_name: str = 't'
    ) -> Tuple[sp.Expr, bool, Optional[str]]:
        """
        Valida y parsea una fórmula.
        
        Returns:
            (expresion_sympy, es_valida, mensaje_error)
        """
        try:
            # Parsing con transformaciones seguras
            expr = parse_expr(
                formula_str,
                local_dict={param_name: self.t},
                transformations='all'
            )
            
            # Verificar funciones permitidas
            funcs_used = expr.atoms(sp.Function)
            func_names = {f.func.__name__ for f in funcs_used}
            
            forbidden = func_names - self.ALLOWED_FUNCTIONS
            if forbidden:
                return None, False, f"Funciones no permitidas: {forbidden}"
            
            # Verificar complejidad
            complexity = self._measure_complexity(expr)
            if complexity > self.MAX_COMPLEXITY:
                return None, False, f"Fórmula demasiado compleja ({complexity} > {self.MAX_COMPLEXITY})"
            
            return expr, True, None
            
        except Exception as e:
            return None, False, f"Error de sintaxis: {str(e)}"
    
    def _measure_complexity(self, expr: sp.Expr) -> int:
        """Mide profundidad del árbol de expresión."""
        if not expr.args:
            return 1
        return 1 + max(self._measure_complexity(arg) for arg in expr.args)
    
    def to_numpy_function(self, expr: sp.Expr):
        """
        Convierte expresión SymPy a función NumPy vectorizada.
        
        JUSTIFICACIÓN: SymPy es 100-1000x más lento que NumPy.
        Compilamos una vez, evaluamos miles de veces.
        """
        return sp.lambdify(self.t, expr, modules=['numpy'])
