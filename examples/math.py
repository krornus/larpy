import sys
sys.path.append("..")
from larpy import Lexer, Grammar, Parser, rule

input = b"""
(1 + 2 * 3 * (1 + 2)) * 2
"""

class MathGrammar(Grammar):

    # Target production
    E = Grammar.newgoal()
    # Other productions
    T = Grammar.newprod()
    F = Grammar.newprod()

    # Tokens
    NUM    = Grammar.newtok()
    PLUS   = Grammar.newtok()
    TIMES  = Grammar.newtok()
    LPAREN = Grammar.newtok()
    RPAREN = Grammar.newtok()

    # Rules
    @rule(E, [E, PLUS, T])
    def add(self, e, p, t):
        return e + t

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
        return e

    @rule(F, [NUM])
    def num(self, n):
        return n

lex = Lexer(MathGrammar)
lex.token(b"\s+")
lex.token(b"[0-9]+", MathGrammar.NUM, int)
lex.token(b"\+", MathGrammar.PLUS)
lex.token(b"\*", MathGrammar.TIMES)
lex.token(b"\(", MathGrammar.LPAREN)
lex.token(b"\)", MathGrammar.RPAREN)

g = MathGrammar()
p = Parser(lex, g)

if __name__ == "__main__":
    eq = input.decode().strip()
    rv = p.parse(input)
    assert(eval(eq) == rv)
    print(f"{eq} = {rv}")
