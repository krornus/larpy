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
            return False
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

        self._goal = self._add_goal(goal)

    # like add_prod/add_rule, but should only be called
    # by __init__ once. converts the grammar into an
    # augmented grammar with a start symbol S' -> S
    def _add_goal(self, goal):
        pid = self._productions
        rid = len(self._rules)
        self._productions += 1
        self._lookup.append([rid])
        self._names.append("Goal'")
        self._rules.append((pid, [goal]))
        return pid

    def rules(self, lhs):
        if lhs < len(self._tokens):
            raise IndexError
        return (self._rules[x][1] for x in self._lookup[lhs - len(self._tokens)])

    @property
    def goal(self):
        return self._goal

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
        return range(len(self._tokens), self._productions)

    def tokens(self):
        return range(len(self._tokens))

    def symbols(self):
        return range(self._productions)

    def closure(self, items):
        c = intset(len(self))
        iset = set((x for x in items))

        dirty = True
        while dirty:
            # save the initial length
            n = len(c)
            # get each item in the set
            for i in list(iset):
                # get the symbol pointed to by the item in the rhs
                s = i.sym()
                if s is not None and self.isprod(s) and s not in c:
                    # if its a new production, add each rule of the symbol
                    # as a nonkernel item
                    for r in self.rules(s):
                        # add the new item to the itemset,
                        # mark it as done
                        c.add(s)
                        iset.add(Item(self, (s, r), 0))

            # update the dirty bit to reflect
            # if anything was added
            dirty = n != len(c)

        return frozenset(iset)

# production table, either a list of FIRST or FOLLOW sets
class SymbolLookup(list):
    def __init__(self, parser, *args, **kwargs):
        # initial max capacity/length
        self._cap = len(parser)
        self._len = 0

        # load the initial list with the default values
        values = (self._default(parser, x) for x in range(self._cap))
        super(SymbolLookup, self).__init__(values)

        # calculate the sets
        # for each production
        self._populate(parser, *args, **kwargs)

        # now make each set frozen (readonly)
        for x in parser.productions():
            self[x].freeze()

    def __contains__(self, x):
        if x >= 0 and x < self._cap:
            return bool(self[x])
        else:
            return False

    def __setitem__(self, k, v):
        raise AttributeError("first set is read-only")

    def __call__(self, x):
        return super(SymbolLookup, self).__getitem__(x)

    def _default(self, parser, sym):
        return None

    def _populate(self, parser, *args, **kwargs):
        # populate each production set as necessary
        raise NotImplementedError

class First(SymbolLookup):
    def __init__(self, prsr):
        # prodtab init takes *args/**kwargs,
        # doing this to generate a better
        # error on invalid arguments
        super(First, self).__init__(prsr)

    def _default(self, prsr, sym):
        if prsr.isterm(sym):
            return frozenset((sym,))
        else:
            return intset(self._cap)

    def _populate(self, prsr):
        # add all productions to queue
        q = deque(prsr.productions())

        while q:
            # consume one value from queue,
            # re-add it to end of queue if it
            # caused a change in the set
            prod = q.popleft()
            fl = len(self[prod])
            if self._partial(prsr, prod) or not self[prod]:
                q.append(prod)

    # calculate the partial first set for one production
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
                # the nonterminal's first set. we break
                # when we have used all leading epsilons
                if (prsr.isterm(rp) and rp != prsr.epsilon) or prsr.epsilon not in self[rp]:
                    break
            else:
                # for/else: this only triggers if epsilon
                # was in every rp (break was never hit)
                # this means that every production in r
                # had epsilon in its first set, implying
                # this also has epsilon in its first set
                added |= bool(self[prod].add(prsr.epsilon))

        return added


class Follow(SymbolLookup):
    def __init__(self, prsr, first):
        # prodtab init takes *args/**kwargs,
        # doing this to generate a better
        # error on invalid arguments
        super(Follow, self).__init__(prsr, first)

    def _default(self, prsr, sym):
        if prsr.isterm(sym):
            return None
        else:
            return intset(self._cap)

    def _populate(self, prsr, first):
        self[prsr.goal].add(prsr.eof)

        # add all productions to queue
        q = deque(prsr.productions())

        while q:
            # consume one value from queue,
            # re-add it to end of queue if it
            # caused a change in the set
            prod = q.popleft()
            fl = len(self[prod])
            if self._partial(prsr, first, prod) or not self[prod]:
                q.append(prod)

    # calculate the partial follow set for one production
    def _partial(self, prsr, first, prod):
        added = False

        # for each rule starting with prod
        for r in prsr.rules(prod):
            rhs = list(r)
            # we want to track how far backward
            # into the rule epsilon is in the first sets.
            # as long as epsilon is in the first set
            # for the current production and all future
            # productions in the rule, then we need to add
            # the follow of the lhs to the follow of the
            # production.
            #
            # e.g.
            #
            # A -> BCDE
            # where epsilon in FIRST(D) and FIRST(E),
            # then FOLLOW(C) has FOLLOW(A)
            #
            # Note: the epsilon flag should always be true
            # for E, because its the end of the rule.
            # epsilon may or may not be in FOLLOW(E) in
            # this example
            #
            # Also note: EPSILON is never added to a FOLLOW
            # set.
            eflag = True

            # iterate backwards so we have an epsilon flag
            for x in reversed(range(len(rhs))):
                if prsr.isterm(rhs[x]):
                    eflag = rhs[x] == prsr.epsilon
                    # FOLLOW is only for non-terminals
                    # add nothing
                    continue
                elif eflag:
                    # update epsilon flag for next iter
                    eflag = prsr.epsilon in first[rhs[x]]
                    # Add FOLLOW(prod) - EPSILON to FOLLOW(rhs[x])
                    for sp in self[prod]:
                        if sp != prsr.epsilon:
                            added |= self[rhs[x]].add(sp)

                # at this point, we are either mid rule
                # a<B>c, or at the end of the rule (ab<C>)
                # if we are at the end of the rule, we're done.
                # otherwise, everything in FIRST(rhs[x+1]) is
                # placed into FOLLOW(rhs[x]) except epsilon
                if x < len(rhs) - 1:
                    # case a<B>X
                    # add first(X)) - EPSILON to FOLLOW(B)
                    for fp in first[rhs[x+1]]:
                        if fp != prsr.epsilon:
                            added |= self[rhs[x]].add(fp)
        return added

class Item:
    def __init__(self, parser, rule, cursor):
        self.rule = rule
        self.cursor = cursor

        lhs = parser.name(rule[0])
        rhs = [parser.name(x) for x in rule[1]]
        lrhs = " ".join(x for x in rhs[:cursor])
        rrhs = " ".join(x for x in rhs[cursor:])

        self.description = f"{lhs} -> {lrhs} . {rrhs}"

    def sym(self):
        if self.cursor >= 0 and self.cursor < len(self.rule[1]):
            return self.rule[1][self.cursor]

    def __str__(self):
        return self.description

    def __repr__(self):
        return f"<{self.description}>"
