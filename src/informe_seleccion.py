"""Muestra el informe vigente de selección sin ejecutar experimentos."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
INFORME = RAIZ / "docs/informes/seleccion_fase23_estado_actual.md"

if __name__ == "__main__":
    print(INFORME.read_text(encoding="utf-8"))
