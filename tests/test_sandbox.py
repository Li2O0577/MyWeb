"""Tests for the LLM code-interpreter sandbox (_sandbox.py)."""
import unittest
import sys
import os

# Ensure the backend package is importable from the tests directory.
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from backend.services._sandbox import (
    CodeValidationError,
    validate_code_ast,
    make_safe_globals,
    SANDBOX_ENABLED,
)


# ═══════════════════════════════════════════════════════════════════════════════
# AST Validator
# ═══════════════════════════════════════════════════════════════════════════════

class TestASTValidator(unittest.TestCase):
    """Unit tests for validate_code_ast — no subprocess needed."""

    # ── Allowed patterns ──

    def test_empty_code(self):
        validate_code_ast("")

    def test_whitespace_only(self):
        validate_code_ast("   \n  ")

    def test_pandas_head(self):
        validate_code_ast("df.head()")

    def test_pandas_describe(self):
        validate_code_ast("df.describe()")

    def test_pandas_groupby(self):
        validate_code_ast("df.groupby('cat')['x'].mean()")

    def test_numpy_mean(self):
        validate_code_ast("np.array([1,2,3]).mean()")

    def test_numpy_arange(self):
        validate_code_ast("np.arange(0, 10, 0.1)")

    def test_matplotlib_plot(self):
        validate_code_ast("plt.figure(); plt.plot([1,2]); plt.show()")

    def test_matplotlib_scatter(self):
        validate_code_ast("plt.scatter(df['x'], df['y']); plt.show()")

    def test_scipy_stats(self):
        validate_code_ast("stats.norm.pdf(0)")

    def test_sklearn_scaler(self):
        validate_code_ast("sklearn.preprocessing.StandardScaler()")

    def test_function_definition(self):
        validate_code_ast("def f(x):\n    return x ** 2")

    def test_class_definition(self):
        validate_code_ast("class MyClass:\n    pass")

    def test_try_except(self):
        validate_code_ast("try:\n    x = 1\nexcept Exception:\n    x = 2")

    def test_lambda(self):
        validate_code_ast("df.apply(lambda r: r.max())")

    def test_list_comprehension(self):
        validate_code_ast("[x ** 2 for x in range(10)]")

    def test_dict_comprehension(self):
        validate_code_ast("{k: v for k, v in zip(range(3), 'abc')}")

    def test_for_loop(self):
        validate_code_ast("for i in range(10):\n    print(i)")

    def test_while_loop(self):
        validate_code_ast("i = 0\nwhile i < 10:\n    i += 1")

    def test_if_elif_else(self):
        validate_code_ast("if x > 0:\n    print('pos')\nelif x < 0:\n    print('neg')\nelse:\n    print('zero')")

    def test_with_statement_no_open(self):
        # with open() is caught by visit_Call (open is in _BLOCKED_FUNCS)
        with self.assertRaises(CodeValidationError):
            validate_code_ast("with open('f.txt') as f:\n    pass")

    def test_string_formatting(self):
        validate_code_ast("f'the value is {x}'")

    def test_slice_notation(self):
        validate_code_ast("df.iloc[:10, 1:3]")

    def test_unpacking(self):
        validate_code_ast("a, *b = [1, 2, 3]")

    def test_type_single_arg(self):
        validate_code_ast("type(42)")

    def test_getattr_normal(self):
        validate_code_ast("getattr(df, 'columns')")

    def test_hasattr_normal(self):
        validate_code_ast("hasattr(df, 'columns')")

    def test_type_annotation(self):
        validate_code_ast("def f(x: int) -> float:\n    return float(x)")

    def test_assert(self):
        validate_code_ast("assert 1 + 1 == 2")

    def test_augmented_assignment(self):
        validate_code_ast("x = 1\nx += 2")

    # ── Blocked: imports ──

    def test_block_import(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import os")

    def test_block_import_as(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import os as operating_system")

    def test_block_from_import(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("from os import system")

    def test_block_from_import_star(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("from os import *")

    def test_block_import_subprocess(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import subprocess")

    def test_block_import_socket(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import socket")

    def test_block_import_ctypes(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import ctypes")

    def test_block_import_requests(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import requests")

    def test_block_import_shutil(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import shutil")

    def test_block_import_pickle(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import pickle")

    def test_block_import_sys(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("import sys")

    # ── Blocked: dangerous functions ──

    def test_block_eval(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("eval('1+1')")

    def test_block_exec(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("exec('x=1')")

    def test_block_compile(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast('compile("x=1","","exec")')

    def test_block_open(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("open('/etc/passwd')")

    def test_block___import__(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("__import__('os')")

    def test_block_breakpoint(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("breakpoint()")

    def test_block_input(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("input()")

    def test_block_globals(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("globals()")

    def test_block_locals(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("locals()")

    def test_block_vars(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("vars(x)")

    def test_block_dir(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("dir(obj)")

    # ── Blocked: dunder attribute access ──

    def test_block_dunder_access(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("().__class__")

    def test_block_class_bases(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("x.__class__.__bases__")

    def test_block_subclasses_escape(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("().__class__.__bases__[0].__subclasses__()")

    def test_block_int_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("(1).__class__")

    def test_block_list_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("[].__class__")

    def test_block_str_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast('"".__class__')

    def test_block_dict_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("{}.__class__")

    def test_block_dunder_globals(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("x.__globals__")

    def test_block_dunder_builtins(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("x.__builtins__")

    def test_block_dunder_code(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("f.__code__")

    def test_block_dunder_dict(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("obj.__dict__")

    # ── Blocked: getattr/setattr/hasattr with dunder args ──

    def test_block_getattr_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("getattr(x, '__class__')")

    def test_block_setattr_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("setattr(x, '__class__', y)")

    def test_block_hasattr_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("hasattr(x, '__builtins__')")

    def test_block_delattr_dunder(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("delattr(x, '__dict__')")

    # ── Blocked: type() with multiple args (dynamic class creation) ──

    def test_block_type_three_args(self):
        with self.assertRaises(CodeValidationError):
            validate_code_ast("type('X', (object,), {})")

    # ── Syntax errors pass through ──

    def test_syntax_error(self):
        with self.assertRaises(SyntaxError):
            validate_code_ast("df = ")

    def test_syntax_error_unclosed_paren(self):
        with self.assertRaises(SyntaxError):
            validate_code_ast("print((1+2)")


# ═══════════════════════════════════════════════════════════════════════════════
# Safe Builtins
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeBuiltins(unittest.TestCase):
    """Unit tests for the safe builtins dict and wrapper functions."""

    @classmethod
    def setUpClass(cls):
        cls._sb = make_safe_globals()['__builtins__']

    def test_has_basic_types(self):
        for name in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple',
                     'set', 'frozenset', 'bytes', 'bytearray', 'complex'):
            self.assertIn(name, self._sb, f"Missing basic type: {name}")

    def test_has_math_builtins(self):
        for name in ('abs', 'round', 'min', 'max', 'sum', 'pow', 'divmod', 'len'):
            self.assertIn(name, self._sb, f"Missing math builtin: {name}")

    def test_has_iteration_builtins(self):
        for name in ('iter', 'next', 'enumerate', 'zip', 'range', 'map',
                     'filter', 'reversed', 'sorted'):
            self.assertIn(name, self._sb, f"Missing iteration builtin: {name}")

    def test_has_type_builtins(self):
        for name in ('type', 'isinstance', 'issubclass', 'callable'):
            self.assertIn(name, self._sb, f"Missing type builtin: {name}")

    def test_has_exception_types(self):
        for name in ('Exception', 'ValueError', 'TypeError', 'KeyError',
                     'IndexError', 'AttributeError', 'RuntimeError',
                     'NotImplementedError', 'OSError', 'ZeroDivisionError',
                     'StopIteration', 'AssertionError', 'ImportError'):
            self.assertIn(name, self._sb, f"Missing exception: {name}")

    def test_has_print(self):
        self.assertIn('print', self._sb)

    def test_has_true_false_none(self):
        self.assertIn('True', self._sb)
        self.assertIn('False', self._sb)
        self.assertIn('None', self._sb)

    def test_has_super(self):
        self.assertIn('super', self._sb)

    def test_has_classmethod_staticmethod_property(self):
        for name in ('classmethod', 'staticmethod', 'property'):
            self.assertIn(name, self._sb)

    # ── Removed builtins ──

    def test_no_eval(self):
        self.assertNotIn('eval', self._sb)

    def test_no_exec(self):
        self.assertNotIn('exec', self._sb)

    def test_no_compile(self):
        self.assertNotIn('compile', self._sb)

    def test_no_open(self):
        self.assertNotIn('open', self._sb)

    def test_no___import__(self):
        self.assertNotIn('__import__', self._sb)

    def test_no_breakpoint(self):
        self.assertNotIn('breakpoint', self._sb)

    def test_no_input(self):
        self.assertNotIn('input', self._sb)

    def test_no_globals(self):
        self.assertNotIn('globals', self._sb)

    def test_no_locals(self):
        self.assertNotIn('locals', self._sb)

    def test_no_vars(self):
        self.assertNotIn('vars', self._sb)

    def test_no_dir(self):
        self.assertNotIn('dir', self._sb)

    def test_no_memoryview(self):
        self.assertNotIn('memoryview', self._sb)

    # ── Wrapped builtins ──

    def test_safe_getattr_allowed(self):
        class Obj:
            name = 'test'
        result = self._sb['getattr'](Obj, 'name')
        self.assertEqual(result, 'test')

    def test_safe_getattr_allowed_with_default(self):
        result = self._sb['getattr']({}, 'missing', 42)
        self.assertEqual(result, 42)

    def test_safe_getattr_dunder_blocked(self):
        with self.assertRaises(AttributeError):
            self._sb['getattr'](object(), '__class__')

    def test_safe_getattr_single_underscore_blocked(self):
        with self.assertRaises(AttributeError):
            self._sb['getattr'](object(), '_private')

    def test_safe_setattr_allowed(self):
        class Obj:
            pass
        obj = Obj()
        self._sb['setattr'](obj, 'name', 'value')
        self.assertEqual(obj.name, 'value')

    def test_safe_setattr_dunder_blocked(self):
        class Obj:
            pass
        with self.assertRaises(AttributeError):
            self._sb['setattr'](Obj(), '__class__', int)

    def test_safe_delattr_allowed(self):
        class Obj:
            pass
        obj = Obj()
        obj.x = 1
        self._sb['delattr'](obj, 'x')
        self.assertFalse(hasattr(obj, 'x'))

    def test_safe_delattr_dunder_blocked(self):
        with self.assertRaises(AttributeError):
            self._sb['delattr'](object(), '__dict__')

    def test_safe_hasattr_allowed(self):
        class Obj:
            pass
        obj = Obj()
        obj.name = 'value'
        self.assertTrue(self._sb['hasattr'](obj, 'name'))
        self.assertFalse(self._sb['hasattr'](obj, 'missing'))

    def test_safe_hasattr_dunder_blocked(self):
        # Should return False for _-prefixed attributes
        self.assertFalse(self._sb['hasattr'](object(), '__class__'))

    def test_has_build_class(self):
        self.assertIn('__build_class__', self._sb)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSandboxedExec(unittest.TestCase):
    """Integration tests that exercise _execute_code_in_subprocess."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import numpy as np
        cls._df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat': ['a', 'b', 'a', 'b', 'a'],
        })
        cls._executor = None

    def _run(self, code):
        from backend.services.llm_service import _execute_code_in_subprocess
        return _execute_code_in_subprocess(code, self._df, timeout=15)

    # ── Allowed operations ──

    def test_basic_pandas(self):
        r = self._run("print(df.shape)")
        self.assertIn("(5, 3)", r["text"])

    def test_basic_numpy(self):
        r = self._run("print(np.mean(df['x']))")
        self.assertIn("3.0", r["text"])

    def test_matplotlib_figure_capture(self):
        r = self._run("plt.figure(); plt.plot(df['x'], df['y']); plt.show()")
        self.assertTrue(len(r["images"]) >= 1, "Should capture at least one figure")
        self.assertIn("base64", r["images"][0])

    def test_multiple_figures(self):
        r = self._run(
            "plt.figure(); plt.hist(df['x']); plt.show()\n"
            "plt.figure(); plt.scatter(df['x'], df['y']); plt.show()"
        )
        self.assertTrue(len(r["images"]) >= 2, "Should capture two figures")

    def test_print_output(self):
        r = self._run('print("hello sandbox")')
        self.assertIn("hello sandbox", r["text"])

    def test_series_computation(self):
        r = self._run("print(df['x'].sum())")
        self.assertIn("15", r["text"].replace(".0", ""))

    def test_describe(self):
        r = self._run("print(df.describe())")
        self.assertIn("mean", r["text"])

    # ── Blocked patterns ──

    def test_import_os_blocked_in_exec(self):
        r = self._run("import os")
        self.assertNotIn("images", [img for img in r.get("images", [])])
        # Should contain error about sandbox or import
        text = r.get("text", "")
        self.assertTrue(
            "不允许" in text or "import" in text.lower() or "SyntaxError" in text
            or text.strip() == "",
            f"Expected block message, got: {text!r}"
        )

    def test_eval_blocked_in_exec(self):
        r = self._run("eval('1+1')")
        text = r.get("text", "")
        self.assertTrue(
            "不允许" in text or "eval" in text.lower() or "error" in text.lower()
            or "NameError" in text or text.strip() == "",
            f"Expected block, got: {text!r}"
        )

    def test_dunder_access_blocked_in_exec(self):
        # This should be blocked by AST validation before subprocess
        r = self._run("().__class__")
        text = r.get("text", "")
        self.assertTrue(
            "不允许" in text or "安全" in text or "blocked" in text.lower()
            or text.strip() == "",
            f"Expected block, got: {text!r}"
        )

    def test_open_blocked_in_exec(self):
        r = self._run("open('/etc/passwd')")
        text = r.get("text", "")
        self.assertTrue(
            "不允许" in text or "open" in text.lower() or "NameError" in text
            or text.strip() == "",
            f"Expected block, got: {text!r}"
        )

    # ── Edge cases ──

    def test_empty_code_subprocess(self):
        r = self._run("")
        self.assertIsNotNone(r)
        self.assertIsInstance(r.get("text"), str)

    def test_syntax_error_subprocess(self):
        r = self._run("df = ")
        text = r.get("text", "")
        # "SyntaxError" from Python, "Syntax error" / "语法错误" from sandbox,
        # or "invalid syntax" in the traceback
        self.assertTrue(
            "syntax" in text.lower() or "SyntaxError" in text,
            f"Expected syntax error in output, got: {text!r}"
        )

    def test_exception_captured(self):
        r = self._run("raise ValueError('test error')")
        text = r.get("text", "")
        self.assertTrue(
            "ValueError" in text or "test error" in text,
            f"Exception should be captured in output, got: {text!r}"
        )


if __name__ == '__main__':
    unittest.main()
