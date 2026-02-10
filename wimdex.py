import os
import sys
import importlib.util


from rich.console import Console
console = Console()


_current_message = None  # holds the shared message


def wimdsend(value):
    """Send input value and store globally for extensions"""
    global _current_message
    _current_message = value
    return value


def check(value=None):
    """Check for the current stored message in extensions."""
    # if caller wants to verify, return stored message
    return _current_message


def to(value, folder_path):
    """Send input to all .py files inside a folder and execute them."""
    global _current_message
    _current_message = value

    if not os.path.exists(folder_path):
        console.print(f"[red]|  [/red]  [orange][/orange] Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    if not files:
        print(f"[red]|  [/red] No Python files found in '{folder_path}'.")
        return
    console.print(f'[green]| [/green] Running command of {value} ...')

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            # dynamic import
            spec = importlib.util.spec_from_file_location(file[:-3], file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[file[:-3]] = module
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error found in {file}: {e}")