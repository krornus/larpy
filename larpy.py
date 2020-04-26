import re
from collections import deque, namedtuple

class intset(list):
    def __init__(self, len):
        super(intset, self).__init__((False for _ in range(len)))
        self._len = 0
        self._cap = len

    def __contains__(self, x):
        if x >= 0 and x < self._cap:
            return self[x]
        else:
            return False

    def add(self, x):
        if x not in self:
            self._len += 1
            self[x] = True

    def __iter__(self):
        n = 0
        for x in range(self._cap):
            if n >= self._len:
                break
            if self[x]:
                n += 1
                yield x

    def __len__(self):
        return self._len

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{{{', '.join(str(x) for x in iter(self))}}}"

class first(list):
    def __init__(self, len):
        super(first, self).__init__((None for x in range(len)))
        self._len = 0
        self._cap = len

    def __contains__(self, x):
        if x >= 0 and x < self._cap:
            return super(first, self).__getitem__(x) is not None
        else:
            return False

    def __setitem__(self, p, v):
        if p not in self:
            self._len += 1
        return super(first, self).__setitem__(p, v)

    def __getitem__(self, p):
        if super(first, self).__getitem__(p) is None:
            self[p] = intset(self._cap)
        return super(first, self).__getitem__(p)

class Grammar:

    Token = namedtuple('Token', 'name creg')
    Production = namedtuple('Production',  'name')
    Rule = namedtuple('Rule', 'lhs rhs')
    Item = namedtuple('Item', 'rule cursor')

    def __init__(self):
        self._eps = 0
        self._tokens = [True]
        self._symbols = [self.Token("EPSILON","")]
        self._rules = [None]
        self._first = None

    def add_tok(self, name, reg):
        creg = re.compile(reg)
        if creg.match(""):
            raise ValueError("Token regex cannot match empty string")

        tok = len(self._symbols)
        self._rules.append(None)
        self._tokens.append(True)
        self._symbols.append(self.Token(name, creg))

        # first set is now dirty
        self._first = None

        return tok

    def add_prod(self, name):
        if name in self._rules:
            raise ValueError("Production already exists")

        prod = len(self._symbols)
        self._rules.append([])
        self._tokens.append(False)
        self._symbols.append(self.Production(name))

        # first set is now dirty
        self._first = None

        return prod

    def add_rule(self, lhs, rhs):
        if not self.is_prod(lhs):
            raise ValueError("Right-hand-side of rule is not a production")
        self._rules[lhs].append(self.Rule(lhs, tuple(rhs)))

        # first set is now dirty
        self._first = None

    def sym(self, x):
        try:
            return self._symbols[x]
        except IndexError:
            raise ValueError("Missing symbol")

    def is_prod(self, s):
        return s >= 0 and s < len(self._symbols) and not self._tokens[s]

    def is_tok(self, s):
        return s >= 0 and s < len(self._symbols) and self._tokens[s]

    def productions(self):
        return (x for x in range(len(self._symbols)) if self.is_prod(x))

    def tokens(self):
        return (x for x in range(len(self._symbols)) if self.is_tok(x))

    def rules(self, p):
        return self._rules[p]

    def name(self, p):
        return self.sym(p).name

    @property
    def epsilon(self):
        return self._eps

    def itemstr(self, i):
        lhs = self.name(i.rule.lhs)
        rhs = [self.name(x) for x in i.rule.rhs]
        lrhs = " ".join(x for x in rhs[:i.cursor])
        rrhs = " ".join(x for x in rhs[i.cursor:])
        return f"{lhs} -> {lrhs} . {rrhs}"

    def _partial_first(self, f, p):

        # first deal with terminals
        for r in self.rules(p):
            # if the rule is of the form X: epsilon
            a = len(r) == 1 and r[0] == self._eps
            # if the rule is of the form X: t B where t is a token != epsilon
            b = self.is_tok(r[0]) and r[0] != self._eps
            if a or b:
                f[p].add(r[0])

        # now deal with productions
        for r in self.rules(p):
            if self.is_prod(r[0]) and r[0] != p:
                # add all values from first(r[0])
                # that arent epsilon
                for s in f[r[0]]:
                    if s != self._eps:
                        f[p].add(s)
            # if epsilon is in first(r[0]),
            # we need to include the next
            # production in line
            if len(r) > 1 and self._eps in f[r[0]]:
                self._partial_first(f, r[1])

    def first(self, sym):
        if self._first is None:
            # only recalculate this if the grammar
            # has changed since last calculation
            self._first = first(len(self._symbols))

            q = deque(self.productions())
            while q:
                p = q.popleft()
                fl = len(self._first[p])
                self._partial_first(self._first, p)
                if fl != len(self._first[p]) or not len(self._first[p]):
                    q.append(p)

        return iter(self._first[sym])

    # get the symbol pointed to by
    # the cursor associated with an item
    # unless cursor is at the end of the item,
    # in which case return None
    def curs(self, i):
        if i.cursor >= 0 and i.cursor < len(i.rule.rhs):
            return i.rule.rhs[i.cursor]

    def nonkernel(self, r):
        return self.Item(r, cursor=0)

    def closure(self, items):
        # intset of items. each thing in this
        # set represents the lhs of a production
        # where each associated nonkernel item
        # is in the closure
        c = intset(len(self._symbols))
        iset = set((x for x in items))

        dirty = True
        while dirty:
            # save the initial length
            n = len(c)
            # get each item in the set
            for i in list(iset):
                # get the symbol pointed to by the item in the rhs
                s = self.curs(i)
                if s is not None and self.is_prod(s) and s not in c:
                    # if its a new production, add each rule of the symbol
                    # as a nonkernel item
                    for r in self.rules(s):
                        # add the new item to the itemset,
                        # mark it as done
                        c.add(s)
                        iset.add(self.nonkernel(r))

            # update the dirty bit to reflect
            # if anything was added
            dirty = n != len(c)

        return frozenset(iset)

    def goto(self, items, sym):
        fwd = set()
        for i in items:
            if self.curs(i) == sym:
                fwd.add(self.Item(i.rule, i.cursor + 1))
        return self.closure(fwd)

    def parser(self, prog):
        # generate G'
        Goal = g.add_prod("S'")
        self.add_rule(Goal, [prog])

        # create i0
        i0 = self.Item(self.rules(Goal)[0], 0)
        c = {self.closure([i0])}

        dirty = True
        while dirty:
            n = len(c)
            # iterate each item in the closure
            for items in list(c):
                for s in range(len(self._symbols)):
                    d = self.goto(items, s)
                    if d and d not in c:
                        c.add(d)
            dirty = n != len(c)

        return c

g = Grammar()

# num = g.add_tok("num", r"[0-9]+")
# plus = g.add_tok("'+'", r"\+")
# lparen = g.add_tok("'('", r"\(")
# rparen = g.add_tok("')'", r"\)")

# Expr = g.add_prod("Expr")
# Factor = g.add_prod("Factor")

# g.add_rule(Expr, [Factor])
# g.add_rule(Expr, [lparen, Factor, rparen])
# g.add_rule(Factor, [num])
# g.add_rule(Factor, [plus, Factor])
# g.add_rule(Factor, [Factor, plus, num])

# Goal = g.add_prod("Goal")
# g.add_rule(Goal, [Expr])

# i = g.Item(g.rules(Goal)[0], 0)

plus = g.add_tok("'+'", r"\+")
times = g.add_tok("'*'", r"\*")
lparen = g.add_tok("'('", r"\(")
rparen = g.add_tok("')'", r"\)")
id = g.add_tok("id", "[a-zA-Z_][a-zA-Z_0-9]*")

E = g.add_prod("E")
T = g.add_prod("T")
F = g.add_prod("F")

g.add_rule(E, [E, plus, T])
g.add_rule(E, [T])

g.add_rule(T, [T, times, F])
g.add_rule(T, [F])

g.add_rule(F, [lparen, E, rparen])
g.add_rule(F, [id])
