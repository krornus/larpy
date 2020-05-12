import copy
import enum
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
        self._names.append("S'")
        self._rules.append((pid, [goal]))
        return pid

    def rules(self, lhs):
        if lhs < len(self._tokens):
            raise IndexError
        return self._lookup[lhs - len(self._tokens)]

    def rule(self, i):
        return self._rules[i]

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

class Item:
    def __init__(self, parser, rule, cursor):
        self.lhs, self.rhs = parser.rule(rule)
        self.rule = rule
        self.cursor = cursor

        lname = parser.name(self.lhs)
        rnames = [parser.name(x) for x in self.rhs]
        lrname = " ".join(x for x in rnames[:cursor])
        rrname = " ".join(x for x in rnames[cursor:])
        self.description = f"{lname} -> {lrname} . {rrname}"

    def sym(self):
        if self.cursor >= 0 and self.cursor < len(self.rhs):
            return self.rhs[self.cursor]

    def __hash__(self):
        return hash((self.rule, self.cursor))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return self.description

    def __repr__(self):
        return f"<{self.description}>"

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
        for rndx in prsr.rules(prod):
            r = prsr.rule(rndx)[1]
            # if the rule is of the form X: epsilon
            a = all(i == prsr.epsilon for i in r)
            # if the rule is of the form X: t B where t is a token != epsilon
            b = prsr.isterm(r[0]) and r[0] != prsr.epsilon
            if a or b:
                added |= bool(self[prod].add(r[0]))

        # now deal with productions
        for rndx in prsr.rules(prod):
            r = prsr.rule(rndx)[1]
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
        for rndx in prsr.rules(prod):
            rhs = prsr.rule(rndx)[1]
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

class ItemSets:
    def __init__(self, parser):
        r0 = list(parser.rules(parser.goal))[0]
        i0 = Item(parser, r0, 0)
        # list of item sets
        self._itemsets = [self._closure(parser, [i0])]
        # lookup table for item sets
        self._lookup = []
        # load the items
        self._items(parser)

    def goto(self, state, sym):
        return self._lookup[state][sym]

    def __getitem__(self, x):
        return self._itemsets[x]

    def _closure(self, parser, items):
        c = intset(len(parser))
        iset = set((x for x in items))

        dirty = True
        while dirty:
            # save the initial length
            n = len(c)
            # get each item in the set
            for i in list(iset):
                # get the symbol pointed to by the item in the rhs
                s = i.sym()
                if s is not None and parser.isprod(s) and s not in c:
                    # if its a new production, add each rule of the symbol
                    # as a nonkernel item
                    for r in parser.rules(s):
                        # add the new item to the itemset,
                        # mark it as done
                        c.add(s)
                        iset.add(Item(parser, r, 0))

            # update the dirty bit to reflect
            # if anything was added
            dirty = n != len(c)

        return tuple(iset)

    def _goto(self, parser, items, sym):
        fwd = set()
        for i in items:
            if i.sym() == sym:
                fwd.add(Item(parser, i.rule, i.cursor + 1))
        return self._closure(parser, fwd)

    def _items(self, parser):
        dirty = True
        while dirty:
            n = len(self._itemsets)
            # iterate current set of items in the closure
            # using an index. we only append to the list,
            # so this is safe to iterate
            for x in range(n):
                for s in parser.symbols():
                    # create the goto set
                    d = self._goto(parser, self._itemsets[x], s)
                    if d:
                        try:
                            j = self._itemsets.index(d)
                            self._add_lookup(parser, x, s, j)
                        except ValueError:
                            self._itemsets.append(d)
                            self._add_lookup(parser, x, s, len(self._itemsets) - 1)
                    else:
                        self._add_lookup(parser, x, s, -1)
            dirty = n != len(self._itemsets)

    def _add_lookup(self, parser, state, sym, trans):
        if len(self._lookup) < len(self._itemsets):
            amt = len(self._itemsets) - len(self._lookup)
            arys = (array("I", [len(parser)]) for _ in range(amt))
            self._lookup.extend(arys)

        if parser.isprod(sym):
            self._lookup[state][sym] = trans
        else:
            self._lookup[state][sym] = trans

    def __len__(self):
        return len(self._itemsets)

    def __iter__(self):
        return iter(self._itemsets)

class ActionEnum(enum.IntEnum):
    ACCEPT = 0
    REJECT = 1
    SHIFT  = 2
    REDUCE = 3

class Action:
    def __init__(self, act, *args):
        self.action = ActionEnum(act)
        if self.action == ActionEnum.ACCEPT:
            if args:
                raise ValueError("Action ACCEPT expected zero arguments")
        elif self.action == ActionEnum.REJECT:
            if args:
                raise ValueError("Action REJECT expected zero arguments")
        elif self.action == ActionEnum.SHIFT:
            if len(args) != 1:
                raise ValueError("Action SHIFT expected one argument")
            (self.state,) = args
        elif self.action == ActionEnum.REDUCE:
            if len(args) != 1:
                raise ValueError("Action REDUCE expected one argument")
            (self.prod,) = args

    def __str__(self):
        if self.action == ActionEnum.ACCEPT:
            return "acc"
        elif self.action == ActionEnum.REJECT:
            return ""
        elif self.action == ActionEnum.SHIFT:
            return f" s{self.state}"
        elif self.action == ActionEnum.REDUCE:
            return f" r{self.prod}"

class ParsingTable:
    def __init__(self, parser):
        self._first = First(parser)
        self._follow = Follow(parser, self._first)
        self._items = ItemSets(parser)
        self._parser = parser

        self._actions = [
                [Action(ActionEnum.REJECT) for _ in range(len(parser.tokens()))]
                for _ in range(len(self._items))]

        self._goto = [
                ["" for _ in range(len(parser.productions()) - 1)]
                for _ in range(len(self._items))]

        minprod = len(parser.tokens())
        for s in range(len(self._items)):
            for p in parser.productions():
                if p == parser.goal:
                    continue
                x = p - minprod
                if self._items.goto(s, p) >= 0:
                    self._goto[s][x] = self._items.goto(s, p)

        self._populate(parser)

    def _populate(self, parser):
        for i, iset in enumerate(self._items):
            for item in iset:
                s = item.sym()
                if s is not None:
                    # terminal -- apply shift rule
                    if parser.isterm(s):
                        j = self._items.goto(i, s)
                        self._actions[i][s] = Action(ActionEnum.SHIFT, j)
                else:
                    if item.lhs != parser.goal:
                        # end of production, add follow set of the lhs
                        # (not including S')
                        last = item.rhs[-1]
                        for a in self._follow(item.lhs):
                            if self._actions[i][a].action != ActionEnum.REJECT:
                                raise ValueError("Grammar is not SLR(1)")
                            self._actions[i][a] = Action(ActionEnum.REDUCE, item.rule)
                    else:
                        if self._actions[i][parser.eof].action != ActionEnum.REJECT:
                            raise ValueError("Grammar is not SLR(1)")
                        self._actions[i][parser.eof] = Action(ActionEnum.ACCEPT)

    def action(self, state, tok):
        return self._actions[state][tok]

    def goto(self, state, tok):
        return self._goto[state][tok]

    def actionstab(self):
        import tabulate
        toks = [self._parser.name(x) for x in self._parser.tokens()]
        return tabulate.tabulate(self._actions, headers=toks)

    def gototab(self):
        import tabulate
        prods = [self._parser.name(x) for x in self._parser.productions()]
        return tabulate.tabulate(self._goto, headers=prods)
