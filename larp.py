import re
import copy
from collections import deque, namedtuple

ID = 0
def nextid():
    global ID
    rv = ID
    ID += 1
    return rv

class Symbol:
    def __init__(self, name):
        self.id = nextid()
        self.name = name

    def is_terminal(self):
        raise NotImplementedError

    def is_variable(self):
        return not self.is_terminal()

    def __hash__(self):
        return int(self.id)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name}>"

class MetaToken(type):
    @property
    def EPSILON(cls):
        if not hasattr(cls, "_EPSILON"):
            cls._EPSILON = cls("EPSILON", "")
        return cls._EPSILON

    @property
    def END(cls):
        if not hasattr(cls, "_END"):
            cls._END = cls("$", "$")
        return cls._END


class Token(Symbol, metaclass=MetaToken):
    def __init__(self, name, reg):
        self.creg = re.compile(reg)
        super(Token, self).__init__(name)

    def is_terminal(self):
        return True

class MetaProduction(type):
    @property
    def Epsilon(cls):
        if not hasattr(cls, "_Epsilon"):
            cls._Epsilon = cls("Epsilon")
            cls._Epsilon.rules([[Token.EPSILON]])
        return cls._Epsilon

class Production(Symbol, metaclass=MetaProduction):
    def __init__(self, name):
        super(Production, self).__init__(name)
        self._roflag = False
        self.children = None

    def rules(self, rules):
        if self._roflag:
            raise ValueError("Production is read-only")
        elif not len(rules) or not len(rules[0]):
            raise ValueError("cannot have empty rules, use Production.Epsilon instead")
        self.children = tuple((tuple(x) for x in rules))
        self._roflag = True

    def __len__(self):
        return len(self.children)

    def is_terminal(self):
        return False

class First(dict):
    def __getitem__(self, p):
        if p not in self:
            super(First, self).__setitem__(p, set())
        return super(First, self).__getitem__(p)

class reslist(list):
    def __init__(self, *args, default=None, **kwargs):
        self._default = default
        super(reslist, self).__init__(*args, **kwargs)

    def reserve(self, idx):
        if idx > len(self):
            defaults = (copy.copy(self._default) for _ in range(idx - len(self)))
            super(reslist, self).extend(defaults)

    def __setitem__(self, idx, val):
        if isinstance(idx, int):
            self.reserve(idx + 1)
        return super(reslist, self).__setitem__(idx, val)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            self.reserve(idx + 1)
        return super(reslist, self).__getitem__(idx)

Rule = namedtuple('Rule', 'lhs rhs')
Item = namedtuple('Item', 'rule cursor')

class GrammarTable:
    def __init__(self, goal):
        self._lookup = reslist(default=list())
        self._rules = reslist()
        self._symbols = reslist()

        self._terminals = list()
        self._nonterminals = list()
        self._add_rule(goal)

    def _add_rule(self, sym):
        if sym.is_terminal():
            return self._add_terminal(sym)
        elif sym.is_variable():
            return self._add_nonterminal(sym)

    def _add_terminal(self, sym):
        self._lookup[sym.id] = None
        self._symbols[sym.id] = sym
        self._terminals.append(sym.id)

    def _add_nonterminal(self, sym):
        self._symbols[sym.id] = sym
        for child in sym.children:
            for x in child:
                if self.sym(x.id) == None:
                    self._add_rules(x)
            self._lookup[sym.id].append(len(self._rules))
            self._rules.append(Rule(sym.id, tuple(x.id for x in child)))
        self._nonterminals.append(sym.id)

    def sym(self, id):
        return self._symbols[id]

    def name(self, id):
        return self.sym(id).name

    def is_var(self, id):
        return self._symbols[id].is_variable()

    def is_term(self, id):
        return self._symbols[id].is_terminal()

    def rule(self, id):
        return self._rules[id]

    def lookahead(self, item):
        rule = self._rules[item.rule]
        if item.cursor < len(rule.rhs):
            return rule.rhs[item.cursor]

class Parser:
    def __init__(self, prod):
        self._first = None

        goal = Production("Goal")
        goal.rules([[prod]])

        self._add_rules(goal)
        self._goal = self._lookup[goal.id][0]

    def itemstr(self, i):
        rule = self._rules[i.rule]
        lhs = self._symbols[rule.lhs].name
        rhs = [self._symbols[x].name for x in rule.rhs]
        lrhs = " ".join(x for x in rhs[:i.cursor])
        rrhs = " ".join(x for x in rhs[i.cursor:])
        return f"{lhs} -> {lrhs} . {rrhs}"

    def _sym_rules(self, id):
        try:
            for x in self._lookup[id]:
                yield self._rules[x]
        except KeyError:
            pass

    def rules(self, id=None, filt=None):
        if id:
            if filt:
                for x in self._sym_rules(id):
                    if filt(x):
                        yield x
            else:
                for x in self._sym_rules(id):
                    yield x
        else:
            try:
                if filt:
                    for x in self._rules:
                        if filt(x):
                            yield x
                else:
                    for x in self._rules:
                        yield x
            except KeyError:
                pass

    def _partial(self, first, prod):
        # add all non terminals
        for r in self.rules(id=prod):
            sym = self._symbols[r.rhs[0]]

            eps = len(r) == 1 and sym == Production.Epsilon
            term = sym.is_terminal() and sym != Production.Epsilon

            if eps or term:
                first[prod].add(r.rhs[0])

        for r in self.rules(id=prod):
            sym = self._symbols[r.rhs[0]]

            if sym.is_variable() and r.rhs[0] != prod:
                pf = first[r.rhs[0]]
                if Production.Epsilon not in pf:
                    first[prod] |= pf
                else:
                    sym2 = self._symbols[r.rhs[1]]
                    first[prod] |= pf - {Production.epsilon}
                    if len(r) > 1 and self._sym_rules(r.rhs[1]).is_variable():
                        self._partial(first, r.rhs[1])

    def first(self):
        if self._first == None:
            first = First()
            q = deque(self._nonterminals)
            while q:
                prod = q.popleft()
                lstart = len(first)
                self._partial(first, prod)
                if lstart != len(first) or not len(first):
                    q.append(prod)
            self._first = dict(first)
        return self._first

    def closure(self, items):
        cl = set(items)
        added = [False for _ in range(len(self._rules))]
        done = False

        while not done:
            add = set()
            done = True
            for i in cl:
                ahd = self.lookahead(i)
                print(ahd)
                # if the item has a dot before a nonterminal
                if ahd and self._symbols[ahd].is_variable():
                    for p in self._lookup[ahd]:
                        if not added[p]:
                            added[p] = True
                            add.add(Item(p, 0))
                            done = False
            cl |= add
        return cl

id     = Token("id", r"[a-bA-B_][a-bA-B_0-9]+")
plus   = Token("+", r"\+")
times  = Token("*", r"\*")
lparen = Token("(", r"\(")
rparen = Token(")", r"\)")

E = Production("E")
T = Production("T")
F = Production("F")

E.rules([
  [E, plus, T],
  [T],
])

T.rules([
  [T, times, F],
  [F],
])

F.rules([
  [lparen, E, rparen],
  [id],
])

g = Parser(E)
print(g.first())

for x in g.closure({Item(g._goal, 0)}):
    print(g.itemstr(x))

print()

for x in g.closure({Item(g._goal, 1)}):
    print(g.itemstr(x))
