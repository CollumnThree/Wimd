# pip install rich
from time import sleep
from rich.console import Console
from rich.live import Live
from rich.text import Text
import re
import random
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import Lexer
from rich.console import Console
import getpass
import dbcake

username = getpass.getuser()

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

dbcake.db.reconfigure("comlist.dbce")
words = dbcake.db.get('commands-list')
completer = WordCompleter(words, ignore_case=True)

def random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

class PersistentColorWordLexer(Lexer):
    def __init__(self, words):
        self.words = words
        pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
        self._regex = re.compile(pattern, re.IGNORECASE)
        self._color_map = {}  # maps lowercased word -> hex color

    def lex_document(self, document):
        def get_line(lineno):
            text = document.lines[lineno]
            if not text:
                return []

            tokens = []
            last = 0
            for m in self._regex.finditer(text):
                start, end = m.span()
                if start > last:
                    tokens.append(("", text[last:start]))

                matched_text = text[start:end]
                key = matched_text.lower()
                # Assign a color the first time this word is seen
                if key not in self._color_map:
                    self._color_map[key] = random_hex_color()
                color = self._color_map[key]

                style_str = f"fg:{color} bold"
                tokens.append((style_str, matched_text))
                last = end

            if last < len(text):
                tokens.append(("", text[last:]))
            return tokens

        return get_line

session = PromptSession(completer=completer)

def main():
    import pyfiglet

    theme_char=random.randrange(1,5)
    if theme_char == 1:
        console.print(pyfiglet.figlet_format('\\o_o/', font='starwars'))
    elif theme_char == 2:
        console.print(pyfiglet.figlet_format('\\o-o/', font='starwars'))
    elif theme_char == 3:
        console.print(pyfiglet.figlet_format('\\^_^/', font='starwars'))

    elif theme_char == 4:
        console.print(pyfiglet.figlet_format('\\-_-/', font='starwars'))

    elif theme_char == 5:
        console.print(pyfiglet.figlet_format('\\x_x/', font='starwars'))
    try:
        lexer = PersistentColorWordLexer(words)
    except:
        import time
        console.rule(f"[bold black on green ]   Error {time.strftime('%H:%M', time.localtime())} [/bold black on green]", characters="█")
        console.print (' [red]Error Code[/red]: [blue bold]LRpCwLwords[/blue bold]')
        dbcake.db.list ['commands-list'] = ['clear','cls','exit']
        console.print (' [yellow] Error is fixed. now you need to restart Wimd and try to install extentions again.[/yellow]')
        console.rule()
        input()
    exit=False
    while exit == False:
        console.print(f'[blue]│ {username} [yellow]   [/yellow]Workspace')
        import time
        text = session.prompt(f"│   {time.strftime('%H:%M', time.localtime())}  ", lexer=lexer)

        #----------------------| Converter System |-------------------------#
        textinp = "".join(text.split()).lower()
        #-------------------------------------------------------------------#
        import wimdex
        textinp = wimdex.wimdsend(textinp)
        wimdex.to(textinp, "extensions")
        if textinp == 'exit':
            exit=True
if __name__ == "__main__":
    main()