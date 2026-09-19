"""Minimal reader for R's XDR serialization (RDX3 .rda), enough for CASdatasets
data.frames (numeric/integer/factor/character columns). Big-endian throughout."""
import gzip, struct
import pandas as pd

class R:
    def __init__(self, b):
        self.b = b; self.i = 0; self.refs = []
    def int(self):
        v = struct.unpack(">i", self.b[self.i:self.i+4])[0]; self.i += 4; return v
    def dbl(self):
        v = struct.unpack(">d", self.b[self.i:self.i+8])[0]; self.i += 8; return v
    def raw(self, n):
        v = self.b[self.i:self.i+n]; self.i += n; return v
    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr = (flags >> 9) & 1
        has_tag  = (flags >> 10) & 1
        enc = (flags >> 12) & 0xF
        if t == 254:        # NILVALUE
            return None
        if t == 255:        # REFSXP
            idx = flags >> 8
            if idx == 0: idx = self.int()
            return self.refs[idx-1]
        if t == 0:          # NILSXP
            return None
        if t == 1:          # SYMSXP
            name = self.item(); self.refs.append(name); return name
        if t in (2,3,4,5,6,17,18,21,22,25):   # pairlist-like (LISTSXP/LANGSXP/...)
            attr = self.item() if has_attr else None
            tag  = self.item() if has_tag  else None
            car  = self.item()
            cdr  = self.item()
            return ("PAIR", tag, car, cdr, attr)
        if t == 9:          # CHARSXP
            n = self.int()
            if n == -1: return None
            b = self.raw(n)
            if enc & 8:   return b.decode("utf-8", "replace")
            if enc & 4:   return b.decode("latin-1", "replace")
            return b.decode("utf-8", "replace")
        if t == 10:         # LGLSXP
            n = self.int(); vals = [self.int() for _ in range(n)]
            return self._wrap([None if v==-2147483648 else bool(v) for v in vals], has_attr)
        if t == 13:         # INTSXP
            n = self.int(); vals = [self.int() for _ in range(n)]
            vals = [None if v==-2147483648 else v for v in vals]
            return self._wrap(vals, has_attr)
        if t == 14:         # REALSXP
            n = self.int(); vals = [self.dbl() for _ in range(n)]
            return self._wrap(vals, has_attr)
        if t == 16:         # STRSXP
            n = self.int(); vals = [self.item() for _ in range(n)]
            return self._wrap(vals, has_attr)
        if t == 19:         # VECSXP
            n = self.int(); vals = [self.item() for _ in range(n)]
            return self._wrap(vals, has_attr)
        if t == 238:        # ALTREP
            info = self.item(); state = self.item(); attr = self.item()
            return state
        raise ValueError(f"unhandled SEXP type {t} at byte {self.i}")
    def _wrap(self, vals, has_attr):
        attrs = {}
        if has_attr:
            node = self.item()
            while isinstance(node, tuple) and node[0]=="PAIR":
                _, tag, car, cdr, _ = node
                if tag is not None: attrs[tag] = car
                node = cdr
        return ("VEC", vals, attrs)

def load_rda(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    assert b[:5] == b"RDX3\n", b[:8]
    r = R(b[5:])
    assert r.raw(2) == b"X\n"
    fmt = r.int(); wv = r.int(); mrv = r.int()
    slen = r.int(); r.raw(slen)      # native encoding string
    top = r.item()                   # pairlist: tag -> data.frame
    _, tag, car, cdr, _ = top
    name = tag
    vec, cols, attrs = car
    colnames = attrs["names"][1]
    frame = {}
    for cname, col in zip(colnames, cols):
        _, vals, cattr = col
        if "levels" in cattr:                      # factor
            levels = cattr["levels"][1]
            frame[cname] = [None if v is None else levels[v-1] for v in vals]
        else:
            frame[cname] = vals
    return name, pd.DataFrame(frame)

if __name__ == "__main__":
    import sys
    name, df = load_rda(sys.argv[1])
    print("object:", name)
    print("shape:", df.shape)
    print(df.dtypes)
    print(df.head())
    out = sys.argv[2] if len(sys.argv)>2 else None
    if out:
        df.to_csv(out, index=False); print("wrote", out)
