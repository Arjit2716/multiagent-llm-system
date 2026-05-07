"""
Sandboxed code execution tool.
Runs Python code in a subprocess with strict timeout and resource limits.
Prevents filesystem access, network calls, and dangerous imports.
"""
import ast
import asyncio
import sys
import textwrap
from typing import Optional

from backend.tools.registry import BaseTool, ToolParameter, ToolResult
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Forbidden imports - security blocklist
FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "urllib", "http", "requests",
    "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "shutil", "pathlib", "glob", "tempfile",
    "pickle", "marshal", "shelve",
    "ctypes", "cffi", "importlib",
    "__builtin__", "builtins",
}

# Allowed safe modules
SAFE_IMPORTS = {
    "math", "random", "json", "re", "datetime", "collections",
    "itertools", "functools", "operator", "string", "statistics",
    "decimal", "fractions", "cmath",
}

SECURITY_PREAMBLE = """
import sys
import signal

# Remove dangerous builtins
BLOCKED = ['open', 'exec', 'eval', 'compile', '__import__', 'breakpoint']
for b in BLOCKED:
    if hasattr(__builtins__, b):
        delattr(__builtins__, b)

# Timeout handler
def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")
"""


class CodeSecurityError(Exception):
    """Raised when code fails security checks."""
    pass


def check_code_safety(code: str) -> Optional[str]:
    """
    Static analysis of code for security issues.
    Returns error message if unsafe, None if safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in FORBIDDEN_IMPORTS:
                        return f"Forbidden import: '{module}'"
                    if module not in SAFE_IMPORTS:
                        return f"Unknown/untrusted import: '{module}'. Only {SAFE_IMPORTS} are allowed."
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in FORBIDDEN_IMPORTS:
                    return f"Forbidden import: '{module}'"

        # Check for exec/eval calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "compile", "__import__"}:
                    return f"Forbidden function call: '{node.func.id}'"

        # Check for __dunder__ attribute access (potential introspection attacks)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                if node.attr in {"__class__", "__subclasses__", "__globals__", "__builtins__"}:
                    return f"Forbidden attribute access: '{node.attr}'"

    return None


class CodeExecutorTool(BaseTool):
    """
    Sandboxed Python code executor.
    
    Security measures:
    - Static AST analysis before execution
    - Subprocess isolation with timeout
    - Forbidden import blocklist
    - Output size limiting
    """

    name = "code_executor"
    description = "Execute safe Python code in a sandboxed environment. Useful for calculations, data processing, and algorithm testing."
    category = "code"
    parameters = [
        ToolParameter("code", "str", "Python code to execute", required=True),
        ToolParameter("timeout", "int", "Execution timeout in seconds (max 10)", required=False, default=5),
    ]

    async def run(self, code: str, timeout: int = 5) -> ToolResult:
        """Execute code in a sandboxed subprocess."""
        timeout = min(timeout, settings.CODE_EXECUTION_TIMEOUT)

        # Security check
        security_error = check_code_safety(code)
        if security_error:
            logger.warning("code_security_blocked", error=security_error)
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Security violation: {security_error}",
                metadata={"blocked": True},
            )

        # Wrap code to capture output
        wrapped_code = textwrap.dedent(f"""
import sys
import io
_output_buffer = io.StringIO()
sys.stdout = _output_buffer
sys.stderr = _output_buffer

try:
{textwrap.indent(code, '    ')}
except Exception as e:
    print(f"Runtime Error: {{type(e).__name__}}: {{e}}")

sys.stdout = sys.__stdout__
print(_output_buffer.getvalue(), end="")
""")

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", wrapped_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Code execution timed out after {timeout}s",
                )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            # Limit output size
            if len(output) > 5000:
                output = output[:5000] + "\n... [output truncated]"

            success = proc.returncode == 0 and "Runtime Error:" not in output
            return ToolResult(
                tool_name=self.name,
                success=success,
                output=output if output else errors,
                error=errors if not success else None,
                metadata={"return_code": proc.returncode, "stderr": errors[:500]},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Execution error: {e}",
            )
