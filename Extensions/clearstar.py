import wimdex

if wimdex.check() == 'clear' or wimdex.check() == 'cls':
    import os
    import platform
    os_name = platform.system().lower()

    try:
        if os_name == "windows":
            # Windows
            os.system('cls')
        elif os_name in ["linux", "darwin", "freebsd", "openbsd"]:
            # Unix-like systems (Linux, macOS, BSD)
            os.system('clear')
        else:
            # Try ANSI codes as fallback
            print('\033[H\033[2J', end='')
    except:
            # Final fallback
            print('\n' * 100)