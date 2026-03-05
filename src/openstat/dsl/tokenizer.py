"""Safe tokenizer for OpenStat expressions.

Produces a flat list of typed tokens from a string expression.
No Python eval is ever used.

Supports:
- Backtick-quoted identifiers: `Column Name`, `income ($)`
- Function calls: log(x), sqrt(x), is_null(x)
- Standard operators and boolean keywords
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token types."""

    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    OP = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TT
    value: str

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


# Order matters: longer operators first.
_TOKEN_SPEC: list[tuple[TT | None, str]] = [
    (None, r"\s+"),  # skip whitespace
    (TT.NUMBER, r"\d+(?:\.\d+)?"),
    (TT.STRING, r'"[^"]*"|\'[^\']*\''),
    (TT.IDENT, r"`[^`]+`"),  # backtick-quoted identifiers
    (TT.OP, r">=|<=|!=|==|>|<|\+|-|\*\*|\*|/|%"),
    (TT.LPAREN, r"\("),
    (TT.RPAREN, r"\)"),
    (TT.COMMA, r","),
    (TT.IDENT, r"[A-Za-z_][A-Za-z0-9_]*"),
]

_KEYWORDS = {"and": TT.AND, "or": TT.OR, "not": TT.NOT}

_PATTERN = re.compile(
    "|".join(f"(?P<G{i}>{pat})" for i, (_, pat) in enumerate(_TOKEN_SPEC))
)


def tokenize(text: str) -> list[Token]:
    """Tokenize an expression string into a list of Tokens.

    Raises ValueError if the input contains unrecognized characters.
    """
    tokens: list[Token] = []
    last_end = 0
    for m in _PATTERN.finditer(text):
        # Check for unmatched characters between tokens
        if m.start() > last_end:
            bad = text[last_end:m.start()]
            raise ValueError(
                f"Unexpected character(s) at position {last_end}: {bad!r}"
            )
        last_end = m.end()
        for i, (tt, _) in enumerate(_TOKEN_SPEC):
            val = m.group(f"G{i}")
            if val is not None:
                if tt is None:
                    break  # whitespace — skip
                if tt == TT.IDENT:
                    # Strip backticks if present
                    if val.startswith("`") and val.endswith("`"):
                        tokens.append(Token(TT.IDENT, val[1:-1]))
                    elif val.lower() in _KEYWORDS:
                        tokens.append(Token(_KEYWORDS[val.lower()], val.lower()))
                    else:
                        tokens.append(Token(tt, val))
                elif tt == TT.STRING:
                    tokens.append(Token(tt, val[1:-1]))  # strip quotes
                else:
                    tokens.append(Token(tt, val))
                break
    # Check for trailing unmatched characters
    if last_end < len(text):
        bad = text[last_end:]
        if bad.strip():  # ignore trailing whitespace
            raise ValueError(
                f"Unexpected character(s) at position {last_end}: {bad.strip()!r}"
            )
    tokens.append(Token(TT.EOF, ""))
    return tokens
