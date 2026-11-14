"""Mantiene reservados los datos reales de Test durante la revisión de selección."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid == "tests/test_preprocessing_pipeline.py::test_el_punto_de_entrada_da_la_tabla_con_todas_las_variables":
            item.add_marker(pytest.mark.skip(reason="Reserva de Test: esta prueba compartida procesa los tres tramos reales; no se ejecuta en la revisión de Fase 2.3"))
