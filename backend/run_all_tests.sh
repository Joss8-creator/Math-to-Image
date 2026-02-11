# backend/run_all_tests.sh
#!/bin/bash

# Script para ejecutar todos los tests y benchmarks

echo "=================================================="
echo "SISTEMA DE ARTE MATEMÁTICO - SUITE DE TESTS"
echo "=================================================="

# Activar entorno virtual
# Adaptado para Windows PowerShell
if [ -d "venv" ]; then
    . venv/Scripts/activate
elif [ -d ".venv" ]; then
    . .venv/Scripts/activate
else
    echo "No se encontró un entorno virtual (venv o .venv). Por favor, créalo e instale las dependencias."
    echo "Ej: python -m venv venv && ./venv/Scripts/pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "[1/4] Tests unitarios..."
pytest tests/test_validation.py -v --tb=short

echo ""
echo "[2/4] Tests de integración..."
pytest tests/test_renderer.py -v --tb=short
pytest tests/test_contour_fitter.py -v --tb=short

echo ""
echo "[3/4] Benchmarks de rendimiento..."
python tests/benchmark.py

echo ""
echo "[4/4] Cobertura de código..."
pytest --cov=app --cov-report=html tests/

echo ""
echo "=================================================="
echo "✓ Tests completados"
echo "Reporte de cobertura: htmlcov/index.html"
echo "Gráficos de benchmarks: *.png"
echo "=================================================="
