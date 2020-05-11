import copy
import struct
import functools
from collections import deque

def array(fmt, shape):
    # make an n-D list of known shape in (hopefully) one allocation
    nmemb = functools.reduce(lambda x,y: x*y, shape)
    size = struct.calcsize(fmt)
    ary = memoryview(bytearray(nmemb*size)).cast(fmt, shape=shape)
    return ary.tolist()

class intset(list):
    def __init__(self, len):
        super(intset, self).__init__(array("?", [len]))
        self._len = 0
        self._cap = len
        self._ro = False

    def __contains__(self, x):
        if x >= 0 and x < self._cap:
            return self[x]
        else:
            return False

    def add(self, x):
        if not self._ro:
            if x not in self:
                self._len += 1
                self[x] = True
                return x
            return None
        else:
            raise AttributeError("Set is read only")

    def freeze(self):
        self._ro = True

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

class Grammar:
    def __init__(self, tokens):
        self._tokens = tokens
        self._productions = len(self._tokens)

        self._names = []
        self._rules = []
        self._lookup = []

    def add_prod(self, name):
        id = self._productions
        self._productions += 1
        self._lookup.append([])
        self._names.append(name)
        return id

    def add_rule(self, rhs, lhs):
        id = len(self._rules)
        self._rules.append((rhs, lhs))
        self._lookup[rhs - len(self._tokens)].append(id)

    def parser(self, prod):
        if len(self._tokens) <= prod < self._productions:
            return Parser(self, prod)
        else:
            raise ValueError("Invalid production")

class Parser:
    def __init__(self, grammar, goal):
        self._tokens = grammar._tokens
        self._productions = grammar._productions

        self._names = copy.copy(grammar._names)
        self._rules = copy.copy(grammar._rules)
        self._lookup = copy.copy(grammar._lookup)

    def rules(self, lhs):
        if lhs < len(self._tokens):
            raise IndexError
        return (self._rules[x][1] for x in self._lookup[lhs - len(self._tokens)])

    @property
    def epsilon(self):
        return self._tokens.EPSILON

    @property
    def eof(self):
        return self._tokens.EOF

    def isterm(self, id):
        if id < len(self._tokens):
           return True
        elif id < self._productions:
            return False
        else:
            raise IndexError

    def isprod(self, id):
        return not self.isterm(id)

    def name(self, id):
        if self.isterm(id):
            return self._tokens(id).name
        else:
            return self._names[id - len(self._tokens)]

    def __len__(self):
        return self._productions

    def productions(self):
        return (x for x in range(len(self._tokens), self._productions))

    def tokens(self):
        return (self._tokens(x) for x in range(len(self._tokens)))

    def symbols(self):
        return (x for x in range(self._productions))

class first(list):
    def __init__(self, parser):
        self._cap = len(parser)
        iv = lambda x: None if parser.isterm(x) else intset(self._cap)
        super(first, self).__init__((iv(x) for x in range(self._cap)))
        self._len = 0

        # calculate the first sets
        # for each production
        self._first(parser)

        # now make each set frozen
        for x in parser.productions():
            self[x].freeze()

    def __contains__(self, x):
        if x >= 0 and x < self._cap:
            return super(first, self).__getitem__(x) is not None
        else:
            return False

    def __setitem__(self, k, v):
        raise AttributeError("first set is read-only")

    def _partial(self, prsr, prod):
        added = False
        # first deal with terminals
        for r in prsr.rules(prod):
            # if the rule is of the form X: epsilon
            a = r and all(i == prsr.epsilon for i in r)
            # if the rule is of the form X: t B where t is a token != epsilon
            b = prsr.isterm(r[0]) and r[0] != prsr.epsilon
            if a or b:
                added |= bool(self[prod].add(r[0]))

        # now deal with productions
        for r in prsr.rules(prod):
            # loop productions until epsilon not in rp
            for rp in r:
                if prsr.isprod(rp) and rp != prod:
                    # add all values from first(rp)
                    # that arent epsilon
                    for s in self[rp]:
                        # dont add epsilon here,
                        # only if all rp have epsilon.
                        if s != prsr.epsilon:
                            added |= bool(self[prod].add(s))

                # add each nonterminal's first set to
                # prod's first as long as epsilon is in
                # the nonterminal's first
                if prsr.isterm(rp) or prsr.epsilon not in self[rp]:
                    break
            else:
                # for/else: this only triggers if epsilon
                # was in every rp (break was never hit)
                # this means that every production in r
                # had epsilon in its first set, implying
                # this also has epsilon in its first set
                added |= bool(self[prod].add(prsr.epsilon))

        return added

    def _first(self, prsr):
        q = deque(prsr.productions())
        while q:
            prod = q.popleft()
            fl = len(self[prod])
            if self._partial(prsr, prod) or not self[prod]:
                q.append(prod)
