import os
import sys
import ast
import inspect
import textwrap
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from syntx.syn import SyNTo
from syntx.tvf import TVFModel
from syntx.syngs import GeodesicShootingModel
from syntx.syn_jax import SyNJAX
from syntx.tvf_jax import TVFModelJAX
from syntx.syngs_jax import GeodesicShootingModelJAX


class SobolevClampASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found_clamp_calls = []
        self.found_magic_clamp_keywords = []

    def visit_Call(self, node):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in ['clamp', 'clip']:
            self.found_clamp_calls.append(func_name)

        for kw in node.keywords:
            if kw.arg in ['min', 'max'] and isinstance(kw.value, (ast.Constant, ast.UnaryOp, ast.Num)):
                val = None
                if isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.operand, ast.Constant):
                    val = -kw.value.operand.value if isinstance(kw.value.op, ast.USub) else kw.value.operand.value
                self.found_magic_clamp_keywords.append((kw.arg, val))

        self.generic_visit(node)


def check_method_ast_for_clamps(cls, method_name):
    method = getattr(cls, method_name, None)
    if method is None:
        return
    source = inspect.getsource(method)
    source_dedented = textwrap.dedent(source)
    parsed = ast.parse(source_dedented)
    visitor = SobolevClampASTVisitor()
    visitor.visit(parsed)

    assert len(visitor.found_clamp_calls) == 0, (
        f"{cls.__name__}.{method_name} AST contains forbidden clamp/clip call: {visitor.found_clamp_calls}"
    )
    assert len(visitor.found_magic_clamp_keywords) == 0, (
        f"{cls.__name__}.{method_name} AST contains magic clamp keywords: {visitor.found_magic_clamp_keywords}"
    )
    assert "float(dim)" not in source, (
        f"{cls.__name__}.{method_name} source contains unexposed division '/ float(dim)' on alpha!"
    )


def test_static_ast_sobolev_clamp_inspection():
    classes_and_methods = [
        (SyNTo, '_apply_sobolev_green_operator'),
        (SyNTo, '_apply_dsti_green_operator'),
        (TVFModel, '_apply_sobolev_green_operator'),
        (TVFModel, '_apply_dsti_green_operator'),
        (GeodesicShootingModel, 'apply_green_operator'),
        (GeodesicShootingModel, '_apply_sobolev_green_operator'),
        (SyNJAX, '_apply_sobolev_green_operator'),
        (SyNJAX, '_apply_dsti_green_operator'),
        (TVFModelJAX, '_apply_sobolev_green_operator'),
        (TVFModelJAX, '_apply_dsti_green_operator'),
        (GeodesicShootingModelJAX, 'apply_green_operator'),
        (GeodesicShootingModelJAX, '_apply_sobolev_green_operator'),
    ]

    for cls, method_name in classes_and_methods:
        check_method_ast_for_clamps(cls, method_name)
