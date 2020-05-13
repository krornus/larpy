import enum
import functools
import re
import struct
import types
import typing
from collections import deque

class LexerError(ValueError):
    """Generic lexer error"""

class UnexpectedCharacter(ValueError):
    """Unexpected character in input"""

class ParsingError(SyntaxError):
    """Generic parsing error"""

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

class TokenMatcher:
    def __init__(self, pat, tok=None, val=None, lookahead=None):
        self._creg = re.compile(pat)
        if self._creg.match(b""):
            raise ValueError("Token regex may not match empty string")
        self._val = val
        self._tok = tok
        self._lookahead = lookahead

    @property
    def ident(self):
        return self._tok

    def match(self, stream):
        m = self._creg.match(stream)
        if m and (self._lookahead is None or self._lookahead.match(stream)):
            n = m.end() - m.start()
            if callable(self._val):
                return n, self._val(m.group(0))
            else:
                return n, self._val

class Lexer:
    def __init__(self, grammar, buf):
        # for this prototype, we read
        # the entire buffer into memory
        self._buf = memoryview(buf)
        self._idx = 0
        self._tokens = []
        self._grammar = grammar

    @property
    def buf(self):
        return self._buf[self._idx:]

    def token(self, pat, tok=None, val=None, lookahead=None):
        if tok is not None:
            if not self._grammar.isterm(tok):
                raise ValueError("Invalid token")
        self._tokens.append(TokenMatcher(pat, tok, val, lookahead))
        return tok

    def nexttok(self):
        ttok = None
        found = True

        while self.buf and found and ttok is None:
            tlen = 0
            tval = None
            found = False
            for tok in self._tokens:
                m = tok.match(self.buf)
                if m:
                    if m[0] > tlen or not found:
                        found = True
                        tlen, tval = m
                        ttok = tok.ident
            if not found:
                raise UnexpectedCharacter(chr(self.buf[0]), self._idx)
            if tlen:
                self._idx += tlen

        if ttok is not None:
            return ttok, tval
        else:
            return self._grammar.EOF, None

    def __iter__(self):
        return self

    def __next__(self):
        return self.nexttok()


class _GrammarBuilder:
    def __init__(self):
        self.symbols = 0
        self.productions = None

        self.names = []
        self.lookup = []
        self.rules = []
        self.actions = []

    def add_tok(self, name):
        # all tokens must be added first
        assert(self.productions is None)

        id = self.symbols
        self.symbols += 1
        self.names.append(name)
        return id


    def add_prod(self, name):
        if self.productions is None:
            self.productions = self.symbols

        id = self.symbols
        self.symbols += 1
        self.lookup.append([])
        self.names.append(name)
        return id

    def add_rule(self, rhs, lhs, action):
        # productions must be added first
        assert(self.productions is not None)

        id = len(self.rules)
        self.rules.append((rhs, lhs))
        self.actions.append(action)
        self.lookup[rhs - self.productions].append(id)

class GrammarMeta(type):
    def __new__(cls, classname, bases, classdict, **kwargs):

        # XXX: Not sure if its faster to loop through
        # the whole list every time (tok, prod, rules)
        # or to seperate everything into lists (allocations)
        # then iter each list. both ways would be O(N).
        # Not going to worry about it until its noticable.
        builder = _GrammarBuilder()
        syms = dict()
        pids = list()

        tokstart = builder.symbols

        # first add each token
        for name, member in classdict.items():
            if isinstance(member, _TokenID):
                id = builder.add_tok(name)
                syms[member.id] = id
                classdict[name] = id

        classdict["EOF"] = builder.add_tok("EOF")
        classdict["EPSILON"] = builder.add_tok("EPSILON")

        tokend = builder.symbols
        prodstart = builder.symbols

        # then each production
        for name, member in classdict.items():
            if isinstance(member, _ProductionID):
                id = builder.add_prod(name)
                syms[member.id] = id
                classdict[name] = id
                pids.append(id)

        prodend = builder.symbols

        # now, add production rules for each _ProductionMethod
        for name, member in classdict.items():
            if isinstance(member, rule):
                # convert the member to a _ProductionMethod first
                member = member(lambda *_: None)
                lhs = syms[member.lhs.id]
                rhs = [syms[x.id] for x in member.rhs]
                builder.add_rule(lhs, rhs, member.fn)
                classdict[name] = member.fn
            elif isinstance(member, _ProductionMethod):
                lhs = syms[member.lhs.id]
                rhs = [syms[x.id] for x in member.rhs]
                builder.add_rule(lhs, rhs, member.fn)
                classdict[name] = member.fn

        # now ensure no empty Productions exist
        for id in pids:
            if not builder.lookup[id - prodstart]:
                raise ValueError(f"Empty production: {builder.names[id]}")

        # now add the builder lists to the new class
        # this is not perfect -- these lists are still
        # mutable, but we have the only reference to them
        # due to the builder reference being dropped after
        # this function. it would be nice to have them
        # as immutable without copying them to tuples
        # or something, but at a certain point, this
        # is python
        classdict["_names"] = builder.names
        classdict["_rules"] = builder.rules
        classdict["_lookup"] = builder.lookup
        classdict["_actions"] = builder.actions
        classdict["_tokstart"] = tokstart
        classdict["_tokend"] = tokend
        classdict["_prodstart"] = prodstart
        classdict["_prodend"] = prodend

        return type.__new__(cls, classname, bases, classdict)

# Fake ID that gets replaced in GrammarMeta
class _TokenID(typing.NamedTuple):
    id: int

# Fake ID that gets replaced in GrammarMeta
class _ProductionID(typing.NamedTuple):
    id: int

# Method wrapper to signify a decorated method for GrammarMeta
class _ProductionMethod(typing.NamedTuple):
    fn: types.MethodType
    lhs: _ProductionID
    rhs: typing.Sequence[_ProductionID]

# adds a new production to the grammar. takes two arguments:
#   lhs: the left hand side of the production, a production id
#   rhs: the right hand side, a list of production and token ids
#
# may be used as a decorator or the return value may be assigned
# as a class attribute
class rule:
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
    def __call__(self, fn):
        return _ProductionMethod(fn, self.lhs, self.rhs)

class Grammar(metaclass=GrammarMeta):
    _sid = 0

    @classmethod
    def newtok(cls):
        id = cls._sid
        cls._sid += 1
        return _TokenID(cls._sid)

    @classmethod
    def newprod(cls):
        id = cls._sid
        cls._sid += 1
        return _ProductionID(cls._sid)

    def __init__(self, goal):
        if not self.isprod(goal):
            raise ValueError("Invalid production")
        self._goal = self._add_goal(goal)

    # like add_prod/add_rule, but should only be called
    # by __init__ once. converts the grammar into an
    # augmented grammar with a start symbol S' -> S
    def _add_goal(self, goal):
        pid = self._prodend
        rid = len(self._rules)

        self._prodend += 1
        self._lookup.append([rid])
        self._names.append("S'")
        self._rules.append((pid, [goal]))
        self._actions.append(lambda x: x)

        return pid

    def rules(self, lhs):
        if lhs < self._prodstart:
            raise IndexError
        return self._lookup[lhs - self._prodstart]

    def action(self, rndx):
        return self._actions[rndx]

    def rule(self, rndx):
        return self._rules[rndx]

    @property
    def goal(self):
        return self._goal

    def isterm(self, id):
        if type(id) != int:
            breakpoint()
        if 0 <= id < self._prodstart:
           return True
        elif id < self._prodend:
            return False
        else:
            raise IndexError

    def isprod(self, id):
        return not self.isterm(id)

    def name(self, id):
        return self._names[id]

    def __len__(self):
        return self._prodend

    def productions(self):
        return range(self._prodstart, self._prodend)

    def tokens(self):
        return range(self._tokstart, self._tokend)

    def symbols(self):
        return range(self._tokstart, self._prodend)

class Item:
    def __init__(self, grammar, rule, cursor):
        self.lhs, self.rhs = grammar.rule(rule)
        self.rule = rule
        self.cursor = cursor

        lname = grammar.name(self.lhs)
        rnames = [grammar.name(x) for x in self.rhs]
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
    def __init__(self, grammar, *args, **kwargs):
        # initial max capacity/length
        self._cap = len(grammar)
        self._len = 0

        # load the initial list with the default values
        values = (self._default(grammar, x) for x in range(self._cap))
        super(SymbolLookup, self).__init__(values)

        # calculate the sets
        # for each production
        self._populate(grammar, *args, **kwargs)

        # now make each set frozen (readonly)
        for x in grammar.productions():
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

    def _default(self, grammar, sym):
        return None

    def _populate(self, grammar, *args, **kwargs):
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
            # if the rule is of the form X: EPSILON
            a = all(i == prsr.EPSILON for i in r)
            # if the rule is of the form X: t B where t is a token != EPSILON
            b = prsr.isterm(r[0]) and r[0] != prsr.EPSILON
            if a or b:
                added |= bool(self[prod].add(r[0]))

        # now deal with productions
        for rndx in prsr.rules(prod):
            r = prsr.rule(rndx)[1]
            # loop productions until EPSILON not in rp
            for rp in r:
                if prsr.isprod(rp) and rp != prod:
                    # add all values from first(rp)
                    # that arent EPSILON
                    for s in self[rp]:
                        # dont add EPSILON here,
                        # only if all rp have EPSILON.
                        if s != prsr.EPSILON:
                            added |= bool(self[prod].add(s))

                # add each nonterminal's first set to
                # prod's first as long as EPSILON is in
                # the nonterminal's first set. we break
                # when we have used all leading EPSILONs
                if (prsr.isterm(rp) and rp != prsr.EPSILON) or prsr.EPSILON not in self[rp]:
                    break
            else:
                # for/else: this only triggers if EPSILON
                # was in every rp (break was never hit)
                # this means that every production in r
                # had EPSILON in its first set, implying
                # this also has EPSILON in its first set
                added |= bool(self[prod].add(prsr.EPSILON))

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
        self[prsr.goal].add(prsr.EOF)

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
            # into the rule EPSILON is in the first sets.
            # as long as EPSILON is in the first set
            # for the current production and all future
            # productions in the rule, then we need to add
            # the follow of the lhs to the follow of the
            # production.
            #
            # e.g.
            #
            # A -> BCDE
            # where EPSILON in FIRST(D) and FIRST(E),
            # then FOLLOW(C) has FOLLOW(A)
            #
            # Note: the EPSILON flag should always be true
            # for E, because its the end of the rule.
            # EPSILON may or may not be in FOLLOW(E) in
            # this example
            #
            # Also note: EPSILON is never added to a FOLLOW
            # set.
            eflag = True

            # iterate backwards so we have an EPSILON flag
            for x in reversed(range(len(rhs))):
                if prsr.isterm(rhs[x]):
                    eflag = rhs[x] == prsr.EPSILON
                    # FOLLOW is only for non-terminals
                    # add nothing
                    continue
                elif eflag:
                    # update EPSILON flag for next iter
                    eflag = prsr.EPSILON in first[rhs[x]]
                    # Add FOLLOW(prod) - EPSILON to FOLLOW(rhs[x])
                    for sp in self[prod]:
                        if sp != prsr.EPSILON:
                            added |= self[rhs[x]].add(sp)

                # at this point, we are either mid rule
                # a<B>c, or at the end of the rule (ab<C>)
                # if we are at the end of the rule, we're done.
                # otherwise, everything in FIRST(rhs[x+1]) is
                # placed into FOLLOW(rhs[x]) except EPSILON
                if x < len(rhs) - 1:
                    # case a<B>X
                    # add first(X)) - EPSILON to FOLLOW(B)
                    for fp in first[rhs[x+1]]:
                        if fp != prsr.EPSILON:
                            added |= self[rhs[x]].add(fp)
        return added

class ItemSets:
    def __init__(self, grammar):
        r0 = list(grammar.rules(grammar.goal))[0]
        i0 = Item(grammar, r0, 0)
        # list of item sets
        self._itemsets = [self._closure(grammar, [i0])]
        # lookup table for item sets
        self._lookup = []
        # load the items
        self._items(grammar)

    def goto(self, state, sym):
        return self._lookup[state][sym]

    def __getitem__(self, x):
        return self._itemsets[x]

    def _closure(self, grammar, items):
        c = intset(len(grammar))
        iset = set((x for x in items))

        dirty = True
        while dirty:
            # save the initial length
            n = len(c)
            # get each item in the set
            for i in list(iset):
                # get the symbol pointed to by the item in the rhs
                s = i.sym()
                if s is not None and grammar.isprod(s) and s not in c:
                    # if its a new production, add each rule of the symbol
                    # as a nonkernel item
                    for r in grammar.rules(s):
                        # add the new item to the itemset,
                        # mark it as done
                        c.add(s)
                        iset.add(Item(grammar, r, 0))

            # update the dirty bit to reflect
            # if anything was added
            dirty = n != len(c)

        return tuple(iset)

    def _goto(self, grammar, items, sym):
        fwd = set()
        for i in items:
            if i.sym() == sym:
                fwd.add(Item(grammar, i.rule, i.cursor + 1))
        return self._closure(grammar, fwd)

    def _items(self, grammar):
        dirty = True
        while dirty:
            n = len(self._itemsets)
            # iterate current set of items in the closure
            # using an index. we only append to the list,
            # so this is safe to iterate
            for x in range(n):
                for s in grammar.symbols():
                    # create the goto set
                    d = self._goto(grammar, self._itemsets[x], s)
                    if d:
                        try:
                            j = self._itemsets.index(d)
                            self._add_lookup(grammar, x, s, j)
                        except ValueError:
                            self._itemsets.append(d)
                            self._add_lookup(grammar, x, s, len(self._itemsets) - 1)
                    else:
                        self._add_lookup(grammar, x, s, -1)
            dirty = n != len(self._itemsets)

    def _add_lookup(self, grammar, state, sym, trans):
        if len(self._lookup) < len(self._itemsets):
            amt = len(self._itemsets) - len(self._lookup)
            arys = (array("I", [len(grammar)]) for _ in range(amt))
            self._lookup.extend(arys)

        if grammar.isprod(sym):
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
            (self.rule,) = args

    def __str__(self):
        if self.action == ActionEnum.ACCEPT:
            return "acc"
        elif self.action == ActionEnum.REJECT:
            return ""
        elif self.action == ActionEnum.SHIFT:
            return f" s{self.state}"
        elif self.action == ActionEnum.REDUCE:
            return f" r{self.rule}"

class ParsingTable:
    def __init__(self, grammar):
        self._first = First(grammar)
        self._follow = Follow(grammar, self._first)
        self._items = ItemSets(grammar)
        self._grammar = grammar

        self._actions = [
                [Action(ActionEnum.REJECT) for _ in range(len(grammar.tokens()))]
                for _ in range(len(self._items))]

        self._goto = [
                ["" for _ in range(len(grammar.productions()) - 1)]
                for _ in range(len(self._items))]

        self._minprod = len(grammar.tokens())
        for s in range(len(self._items)):
            for p in grammar.productions():
                if p == grammar.goal:
                    continue
                x = p - self._minprod
                if self._items.goto(s, p) >= 0:
                    self._goto[s][x] = self._items.goto(s, p)

        self._populate(grammar)

    def _populate(self, grammar):
        for i, iset in enumerate(self._items):
            for item in iset:
                s = item.sym()
                if s is not None:
                    # terminal -- apply shift rule
                    if grammar.isterm(s):
                        j = self._items.goto(i, s)
                        self._actions[i][s] = Action(ActionEnum.SHIFT, j)
                else:
                    if item.lhs != grammar.goal:
                        # end of production, add follow set of the lhs
                        # (not including S')
                        last = item.rhs[-1]
                        for a in self._follow(item.lhs):
                            if self._actions[i][a].action != ActionEnum.REJECT:
                                raise ValueError("Grammar is not SLR(1)")
                            self._actions[i][a] = Action(ActionEnum.REDUCE, item.rule)
                    else:
                        if self._actions[i][grammar.EOF].action != ActionEnum.REJECT:
                            raise ValueError("Grammar is not SLR(1)")
                        self._actions[i][grammar.EOF] = Action(ActionEnum.ACCEPT)

    def action(self, state, tok):
        return self._actions[state][tok]

    def goto(self, state, prod):
        return self._goto[state][prod - self._minprod]

    def actionstab(self):
        import tabulate
        toks = [self._grammar.name(x) for x in self._grammar.tokens()]
        return tabulate.tabulate(self._actions, headers=toks)

    def gototab(self):
        import tabulate
        prods = [self._grammar.name(x) for x in self._grammar.productions()]
        return tabulate.tabulate(self._goto, headers=prods)

class Parser:
    def __init__(self, lexer, grammar):
        self._lexer = lexer
        self._grammar = grammar
        self._tab = ParsingTable(self._grammar)
        self._stack = [(0,None)]

    def parse(self):
        tok, val = next(self._lexer)

        while True:
            # get the action for current state + token
            state = self._peek()
            act = self._tab.action(state, tok)

            if act.action == ActionEnum.SHIFT:
                # add next state to the stack
                self._push((act.state, val))
                # update the current token
                tok, val = next(self._lexer)

            elif act.action == ActionEnum.REDUCE:
                # get the rule to reduce by and pop
                # |rhs| off the stack
                lhs, rhs = self._grammar.rule(act.rule)
                args = self._pop(len(rhs))
                # get new top of stack
                top = self._peek()
                # goto by lhs
                goto = self._tab.goto(top, lhs)
                assert(goto >= 0)
                # call the associated action
                rv = self._grammar.action(act.rule)(self._grammar, *args)
                self._push((goto, rv))

            elif act.action == ActionEnum.ACCEPT:
                return self._pop()[0]
            else:
                raise SyntaxError(f"Invalid syntax: unexpected {tok.name}")

    def _push(self, state):
        self._stack.append(state)

    def _pop(self, n=1):
        return [self._stack.pop()[1] for _ in range(n)]

    def _peek(self):
        return self._stack[-1][0]
