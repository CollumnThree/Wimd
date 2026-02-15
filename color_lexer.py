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
                    self._color_map[key] = colors.random_hex_color()
                color = self._color_map[key]

                style_str = f"fg:{color} bold"
                tokens.append((style_str, matched_text))
                last = end

            if last < len(text):
                tokens.append(("", text[last:]))
            return tokens

        return get_line
