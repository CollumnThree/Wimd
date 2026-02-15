import random
from rich.console import Console
from rich.live import Live
from rich.text import Text
from time import sleep


console = Console()


def lerp(a, b, t):
    return int(a + (b - a) * t)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

def darken(rgb, factor):
    # factor in [0,1], 0 = original, 1 = black
    return tuple(lerp(c, 0, factor) for c in rgb)
def type_with_gradient(
    text: str,
    base_color: str = "#56b6c2",      # primary hue (cyan-ish)
    tail_length: int = 12,            # number of recent chars with gradient tail
    speed: float = 0.05,              # seconds per char
    dim_factor: float = 0.75,         # how dark older text becomes
    bold_current: bool = True,        # emphasis on the newest char
):
    base_rgb = hex_to_rgb(base_color)
    dark_base_rgb = darken(base_rgb, dim_factor)

    with Live(Text(), console=console, refresh_per_second=30) as live:
        typed = ""
        for i, ch in enumerate(text):
            typed += ch

            # Build a Text object for the full line
            t = Text(typed)

            # 1) Dim the older part (everything except the tail)
            start_tail = max(0, len(typed) - tail_length)
            if start_tail > 0:
                dim_hex = rgb_to_hex(dark_base_rgb)
                t.stylize(f"rgb({dark_base_rgb[0]},{dark_base_rgb[1]},{dark_base_rgb[2]})", 0, start_tail)

            # 2) Apply gradient on the recent tail (from dark to bright)
            tail = typed[start_tail:]
            for j in range(len(tail)):
                # position along the tail [0..1]
                pos = j / max(1, len(tail) - 1)
                r = lerp(dark_base_rgb[0], base_rgb[0], pos)
                g = lerp(dark_base_rgb[1], base_rgb[1], pos)
                b = lerp(dark_base_rgb[2], base_rgb[2], pos)
                t.stylize(f"rgb({r},{g},{b})", start_tail + j, start_tail + j + 1)

            # 3) Emphasize the newest char
            if bold_current and len(typed) > 0:
                t.stylize("bold", len(typed) - 1, len(typed))
            live.update(t)
            sleep(speed)
def random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))
