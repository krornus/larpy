import sys
sys.path.append("../")
import lexer
import grammar

input = b"""
"""

class Token(lexer.TokenEnum):
     PLUS   = 0
     NUM    = 1
     LPAREN = 2
     RPAREN = 3

lex = lexer.Lexer(input, Token)

lex.token(b"\s+")
lex.token(b"[0-9]+", Token.NUM, int)
lex.token(b"\+", Token.PLUS)
lex.token(b"\(", Token.LPAREN)
lex.token(b"\)", Token.RPAREN)

g = grammar.Grammar(Token)

Expr   = g.add_prod("Expr")
Factor = g.add_prod("Factor")

g.add_rule(Expr, [Factor])
g.add_rule(Expr, [Token.LPAREN, Expr, Token.RPAREN])

g.add_rule(Factor, [Token.NUM])
g.add_rule(Factor, [Token.PLUS, Factor])
g.add_rule(Factor, [Factor, Token.PLUS, Token.NUM])
