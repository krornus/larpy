import sys
sys.path.append("../")
import grammar
import dragon
import lrk

p = lrk.g.parser(lrk.Expr)
f = grammar.First(p)
print(f[lrk.Expr])
