import sys
sys.path.append("../")
import lexer
import grammar

input = b"""
"""

class Tok(lexer.TokenEnum):
     PLUS   = 0
     TIMES  = 1
     ID     = 2
     LPAREN = 3
     RPAREN = 4

lex = lexer.Lexer(input, Tok)

lex.token(b"\s+")
lex.token(b"[a-zA-Z_][a-zA-Z0-9_]*", Tok.ID, lambda x: x)
lex.token(b"\+", Tok.PLUS)
lex.token(b"\*", Tok.TIMES)
lex.token(b"\(", Tok.LPAREN)
lex.token(b"\)", Tok.RPAREN)

g = grammar.Grammar(Tok)

E = g.add_prod("E")
T = g.add_prod("T")
F = g.add_prod("F")

g.add_rule(E, [E, Tok.PLUS, T])
g.add_rule(E, [T])

g.add_rule(T, [T, Tok.TIMES, F])
g.add_rule(T, [F])

g.add_rule(F, [Tok.LPAREN, E, Tok.RPAREN])
g.add_rule(F, [Tok.ID])