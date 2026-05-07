"""
Calculator tool: Safe mathematical expression evaluation.
Uses Python's ast module to avoid eval() security risks.
"""
import ast
import math
import operator
from typing import Union

from backend.tools.registry import BaseTool, ToolParameter, ToolResult
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Safe operations map
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial, "gcd": math.gcd,
    "pi": math.pi, "e": math.e, "inf": math.inf,
}


def safe_eval(expr: str) -> Union[int, float]:
    """Safely evaluate a mathematical expression using AST parsing."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op = SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if type(node.op) == ast.Pow and abs(right) > 100:
                raise ValueError("Exponent too large")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            op = SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported unary operator")
            return op(_eval(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func = SAFE_FUNCTIONS.get(node.func.id)
                if func is None:
                    raise ValueError(f"Unknown function: {node.func.id}")
                args = [_eval(a) for a in node.args]
                return func(*args)
            elif isinstance(node.func, ast.Attribute):
                # Support math.sqrt style calls
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                    func = SAFE_FUNCTIONS.get(node.func.attr)
                    if func:
                        args = [_eval(a) for a in node.args]
                        return func(*args)
            raise ValueError(f"Unsupported function call")
        elif isinstance(node, ast.Name):
            if node.id in SAFE_FUNCTIONS:
                return SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    return _eval(tree)


class CalculatorTool(BaseTool):
    """
    Safe mathematical calculator using AST-based expression evaluation.
    Supports arithmetic, trigonometry, logarithms, and common math functions.
    """

    name = "calculator"
    description = "Evaluate mathematical expressions safely. Supports arithmetic, trigonometry, logarithms (sqrt, sin, cos, tan, log, exp, etc.)."
    category = "data"
    parameters = [
        ToolParameter("expression", "str", "Mathematical expression to evaluate (e.g. '2 + 3 * sqrt(16)')", required=True),
    ]

    async def run(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression."""
        try:
            # Clean input
            expr = expression.strip()
            if len(expr) > 500:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error="Expression too long (max 500 chars)",
                )

            result = safe_eval(expr)

            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    formatted = str(int(result))
                else:
                    formatted = f"{result:.10g}"
            else:
                formatted = str(result)

            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"expression": expr, "result": result, "formatted": formatted},
                metadata={"expression": expr},
            )
        except ZeroDivisionError:
            return ToolResult(tool_name=self.name, success=False, output=None, error="Division by zero")
        except ValueError as e:
            return ToolResult(tool_name=self.name, success=False, output=None, error=f"Math error: {e}")
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output=None, error=f"Calculation failed: {e}")
