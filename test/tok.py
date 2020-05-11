import sys
sys.path.append("../")
import lexer
import grammar

input = b"""
if var1 <= var2 then
    var3 := 10
endif
"""

class Token(lexer.TokenEnum):
    IF = 0
    THEN = 1
    ELSE = 2
    ENDIF = 3
    ID = 4
    NUMBER = 5
    ASSN = 6
    CMP = 7
    LT = 8
    LE = 9
    EQ = 10
    NE = 11
    GT = 12
    GE = 13

lex = lexer.Lexer(input, Token)

lex.token(b"\s+")
lex.token(b"if", Token.IF)
lex.token(b"then", Token.THEN)
lex.token(b"else", Token.ELSE)
lex.token(b"endif", Token.ENDIF)
lex.token(b"[a-zA-Z_][a-zA-Z0-9_]*", Token.ID, lambda x: x)
lex.token(b"[0-9_]+", Token.NUMBER, lambda x: int(x))
lex.token(b"<", Token.CMP, Token.LT)
lex.token(b"<=", Token.CMP, Token.LE)
lex.token(b"=", Token.CMP, Token.EQ)
lex.token(b"<>", Token.CMP, Token.NE)
lex.token(b">", Token.CMP, Token.GT)
lex.token(b">=", Token.CMP, Token.GE)
lex.token(b":=", Token.ASSN)
