"""Multi-layer sandbox for LLM-generated code execution.

Layer 1: AST pre-validation — parses code, rejects dangerous patterns.
Layer 2: exec() with restricted globals — blocks imports, dangerous builtins,
         and dunder attribute access at runtime.

Usage:
    from services._sandbox import validate_code_ast, make_safe_globals, CodeValidationError

    try:
        validate_code_ast(user_code)
    except CodeValidationError as e:
        return f"Blocked: {e}"

    safe_globals = make_safe_globals()
    safe_globals['df'] = dataframe   # add domain objects
    exec(user_code, safe_globals)
"""

import ast
import builtins as _builtins_mod
import functools

SANDBOX_ENABLED = True

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: AST Validator
# ═══════════════════════════════════════════════════════════════════════════════

class CodeValidationError(Exception):
    """Raised when user code contains a pattern blocked by the sandbox."""


# Functions whose direct-name calls are always blocked.
_BLOCKED_FUNCS = frozenset({
    'eval', 'exec', 'compile', 'open', '__import__',
    'breakpoint', 'input',
    'globals', 'locals', 'vars', 'dir',
    'help', 'exit', 'quit',
})

# Dunder attribute names blocked in attribute-access nodes.
_BLOCKED_DUNDERS = frozenset({
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__globals__', '__builtins__', '__code__', '__closure__',
    '__dict__', '__module__', '__qualname__',
    '__init__', '__new__', '__del__',
    '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
    '__getattribute__', '__setattr__', '__delattr__',
    '__dir__', '__format__', '__sizeof__', '__subclasshook__',
    '__init_subclass__', '__prepare__',
    '__import__',
})


class _SandboxASTValidator(ast.NodeVisitor):
    """Walk code AST and raise CodeValidationError on prohibited patterns."""

    def _check_arg_const(self, node, arg_idx, block_prefix='_'):
        """Raise if positional arg at *arg_idx* is a str constant starting
        with *block_prefix*."""
        if len(node.args) <= arg_idx:
            return
        arg = node.args[arg_idx]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value.startswith(block_prefix):
                raise CodeValidationError(
                    f"禁止使用 {arg.value!r} 访问内部属性。"
                )

    def visit_Import(self, node):
        raise CodeValidationError(
            "不允许使用 import 语句。"
            "所有必要的库（numpy, pandas, matplotlib, scipy, sklearn）已预加载。"
        )

    def visit_ImportFrom(self, node):
        raise CodeValidationError(
            "不允许使用 from ... import 语句。"
            "所有必要的库已预加载，可直接使用。"
        )

    def visit_Call(self, node):
        # Block calls to dangerous name-referenced functions
        if isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_FUNCS:
                raise CodeValidationError(
                    f"沙箱中不允许使用 {node.func.id}()。"
                )
        # Block getattr/setattr/delattr/hasattr with _-prefixed string
        if isinstance(node.func, ast.Name):
            if node.func.id in {'getattr', 'hasattr'}:
                self._check_arg_const(node, 1)
            elif node.func.id in {'setattr'}:
                self._check_arg_const(node, 1)
            elif node.func.id in {'delattr'}:
                self._check_arg_const(node, 1)
        # Block type() with 2+ args (dynamic class creation)
        if isinstance(node.func, ast.Name) and node.func.id == 'type':
            if len(node.args) >= 2:
                raise CodeValidationError(
                    "沙箱中不允许动态创建类（type() 多参数调用）。"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in _BLOCKED_DUNDERS:
            raise CodeValidationError(
                f"沙箱中不允许访问 {node.attr}（双下划线属性）。"
            )
        self.generic_visit(node)

    def visit_Delete(self, node):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in _BLOCKED_DUNDERS:
                raise CodeValidationError(
                    f"沙箱中不允许删除 {target.attr}。"
                )
        self.generic_visit(node)


def validate_code_ast(code):
    """Parse *code* and validate its AST against sandbox rules.

    Raises:
        CodeValidationError — a security-relevant pattern was found.
        SyntaxError           — the code is not syntactically valid Python.

    Returns None on success.
    """
    if not SANDBOX_ENABLED:
        return
    tree = ast.parse(code)
    _SandboxASTValidator().visit(tree)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Safe Builtins
# ═══════════════════════════════════════════════════════════════════════════════

# Names we strip from builtins entirely.
_REMOVED_BUILTIN_NAMES = frozenset({
    'eval', 'exec', 'compile', 'open', '__import__',
    'breakpoint', 'input', 'memoryview',
    'globals', 'locals', 'vars', 'dir',
    'help', 'exit', 'quit',
    'license', 'credits', 'copyright',
})

# Names we replace with safe wrappers.
_WRAPPED_BUILTIN_NAMES = frozenset({
    'getattr', 'setattr', 'delattr', 'hasattr',
})


def _safe_getattr(obj, name, *default):
    if isinstance(name, str) and name.startswith('_'):
        raise AttributeError(f"Access to '{name}' is restricted in sandbox.")
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _safe_setattr(obj, name, value):
    if isinstance(name, str) and name.startswith('_'):
        raise AttributeError(f"Setting '{name}' is restricted in sandbox.")
    return setattr(obj, name, value)


def _safe_delattr(obj, name):
    if isinstance(name, str) and name.startswith('_'):
        raise AttributeError(f"Deleting '{name}' is restricted in sandbox.")
    return delattr(obj, name)


def _safe_hasattr(obj, name):
    if isinstance(name, str) and name.startswith('_'):
        return False
    return hasattr(obj, name)


def _make_safe_builtins():
    """Build a dict of restricted builtins with dangerous names removed and
    attribute-access names wrapped."""
    safe = {}
    for name, obj in vars(_builtins_mod).items():
        if name in _REMOVED_BUILTIN_NAMES:
            continue
        if name in _WRAPPED_BUILTIN_NAMES:
            continue  # added back below with safe wrappers
        safe[name] = obj

    safe['getattr'] = _safe_getattr
    safe['setattr'] = _safe_setattr
    safe['delattr'] = _safe_delattr
    safe['hasattr'] = _safe_hasattr
    safe['__build_class__'] = _builtins_mod.__build_class__

    return safe


# Pre-built safe builtins dict (immutable from caller's perspective).
_SAFE_BUILTINS = _make_safe_builtins()


def make_safe_globals():
    """Return a minimal globals dict suitable for ``exec(code, globals)``.

    Caller must add domain objects (df, np, pd, plt, etc.) before calling
    exec().  The returned dict already has a restricted ``__builtins__``.
    """
    return {
        '__builtins__': _SAFE_BUILTINS,
        '__name__': '__sandboxed__',
        '__doc__': None,
        '__package__': None,
        '__loader__': None,
        '__spec__': None,
    }
