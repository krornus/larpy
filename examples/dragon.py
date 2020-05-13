import sys
sys.path.append("../")

from grammar import Lexer, Grammar, Parser, rule

input = b"""
(1 + 2 * 3 * (1 + 2)) * 0
"""

class DragonGrammar(Grammar):
    E = Grammar.newprod()
    T = Grammar.newprod()
    F = Grammar.newprod()

    NUM    = Grammar.newtok()
    PLUS   = Grammar.newtok()
    TIMES  = Grammar.newtok()
    LPAREN = Grammar.newtok()
    RPAREN = Grammar.newtok()

    @rule(E, [E, PLUS, T])
    def add(self, e, p, t):
        return (e or 0) + t

    @rule(E, [T])
    def product(self, t):
        return t

    @rule(T, [T, TIMES, F])
    def mul(self, t, m, f):
        return t * f

    @rule(T, [F])
    def factor(self, f):
        return f

    @rule(F, [LPAREN, E, RPAREN])
    def pexpr(self, l, e, r):
        return e or 0

    @rule(F, [NUM])
    def num(self, n):
        return n

g = DragonGrammar(DragonGrammar.E)

lex = Lexer(g, input)
lex.token(b"\s+")
lex.token(b"[0-9]+", DragonGrammar.NUM, int)
lex.token(b"\+", DragonGrammar.PLUS)
lex.token(b"\*", DragonGrammar.TIMES)
lex.token(b"\(", DragonGrammar.LPAREN)
lex.token(b"\)", DragonGrammar.RPAREN)

p = Parser(lex, g)
print(p.parse())
