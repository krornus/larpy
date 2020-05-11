import re
import enum

def array(fmt, shape):
    # make an n-D list of known shape in (hopefully) one allocation
    nmemb = functools.reduce(lambda x,y: x*y, shape)
    size = struct.calcsize(fmt)
    ary = memoryview(bytearray(nmemb*size)).cast(fmt, shape=shape)
    return ary.tolist()

class arrayset(list):
    def __init__(self, len):
        super(arrayset, self).__init__(array("?", [len]))
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

# Adds default EOF and EPSILON values
# to the token enum
class TokenMeta(enum.EnumMeta):
    def __new__(metacls, cls, bases, classdict):
        # classdict is a subclassed dict (_EnumDict)
        if classdict._member_names:
            if 'EOF' not in classdict:
                # len(_EnumDict) returns a bad value
                classdict['EOF'] = len(classdict._member_names)
            if 'EPSILON' not in classdict:
                # len(_EnumDict) returns a bad value
                classdict['EPSILON'] = len(classdict._member_names)
        return super().__new__(metacls, cls, bases, classdict)

class TokenEnum(enum.IntEnum, metaclass=TokenMeta):
    def __new__(cls, value):
        if not isinstance(value, int):
            raise ValueError("Token must be of type int")
        # require incremental values. this allows us to have
        # an array of symbols which is contiguous from 0..len
        # and can be suffixed with productions
        if value != len(cls):
            raise ValueError("Bad value: {}: Tokens must be declared with incremental values"
                             .format(value))
        return int.__new__(cls, value)

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

class LexerError(ValueError):
    """Generic lexer error"""

class UnexpectedCharacter(ValueError):
    """Unexpected character in input"""

class Lexer:
    def __init__(self, buf, tokens):
        # for this prototype, we read
        # the entire buffer into memory
        self._buf = memoryview(buf)
        self._idx = 0
        self._tokens = []
        self._tokcls = tokens

    def __len__(self):
        return len(self._tokens)

    @property
    def tokens(self):
        return list(self._tokcls)

    @property
    def buf(self):
        return self._buf[self._idx:]

    def token(self, pat, tok=None, val=None, lookahead=None):
        if tok is not None:
            tok = self._tokcls(tok)
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

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._buf):
            raise StopIteration
        t = self.nexttok()
        if t is not None:
            return t
        else:
            raise StopIteration
