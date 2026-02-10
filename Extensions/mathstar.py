import wimdex
import re
from rich.console import Console
console = Console()


# Define the allowed characters for a safe mathematical expression
ALLOWED_CHARS_PATTERN = re.compile(r"^[\d\s\+\-\*/\(\)\*\*]+$")


def solve_math_expression(expression: str):
    if not ALLOWED_CHARS_PATTERN.match(expression):
        return None
    if not expression.strip():
        return None

    try:
        result = eval(expression)
        return f"Result: {result}"
    except (SyntaxError, ZeroDivisionError):
        return None
    except Exception:
        return None


# --- wimdex integration ---
input_data = wimdex.check()

if input_data:
    output = solve_math_expression(input_data.strip())
    if output:
        console.print(f"[cyan]│  [/cyan]{output}")
