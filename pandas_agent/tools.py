"""
Tools for the Pandas Agent.
"""

import subprocess
import litellm


def execute_python(code: str) -> str:
    """Execute Python code and return the output.

    Parameters
    ----------
    code : str
        Python code to execute.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip() if result.stdout.strip() else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"


# Tool schema
AVAILABLE_TOOLS = [
    {"type": "function", "function": litellm.utils.function_to_dict(execute_python)},
]

FUNCTION_MAP = {
    "execute_python": execute_python,
}
