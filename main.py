# pip install rich
from sys import exit
import subprocess
from rich.console import Console
import re
import random
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
import getpass
import dbcake
import time
import colors_local as colors
import color_lexer

username = getpass.getuser()

console = Console()

dbcake.db.reconfigure("comlist.dbce")
words = dbcake.db.get('commands-list')
completer = WordCompleter(words, ignore_case=True)


session = PromptSession(completer=completer)
import pyfiglet

theme_char=random.randrange(1,5)
match theme_char:
    case 1:
        console.print(pyfiglet.figlet_format('\\o_o/', font='starwars'))
    case 2:
        console.print(pyfiglet.figlet_format('\\o-o/', font='starwars'))
    case 3:
        console.print(pyfiglet.figlet_format('\\^_^/', font='starwars'))
    case 4:
        console.print(pyfiglet.figlet_format('\\-_-/', font='starwars'))
    case 5:
        console.print(pyfiglet.figlet_format('\\x_x/', font='starwars'))
try:
        lexer = color_lexer.PersistentColorWordLexer(words)
except:
        console.rule(f"[bold black on green ]   Error {time.strftime('%H:%M', time.localtime())} [/bold black on green]", characters="█")
        console.print (' [red]Error Code[/red]: [blue bold]LRpCwLwords[/blue bold]')
        dbcake.db.list ['commands-list'] = ['clear','cls','exit']
        console.print (' [yellow] Error is fixed. now you need to restart Wimd and try to install extentions again.[/yellow]')
        console.rule()
        input()
while True:
    console.print(f'[blue]│ {username} [yellow]   [/yellow]Workspace')
    # Build the prompt
    text = session.prompt(f"│   {time.strftime('%H:%M', time.localtime())}  ", lexer=lexer)
    # Parse the command
    text = text.split()
    subprocess.run(text)
    if text == 'exit':
        exit(0)
