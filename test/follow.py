import sys
sys.path.append("../")
import grammar
import dragon
import lrk

p = lrk.g.parser(lrk.Expr)

first = grammar.First(p)
follow = grammar.Follow(p, first)

print(follow(lrk.Expr))
