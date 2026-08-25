"""Algorithmic-information measures (phase P4, plan WP-L).

    gzip_ratio      len(compress(s)) / len(s) -- protocol floor 4, the
                    brutal free baseline for anything complexity-flavoured
    block_entropy   Shannon entropy (bits) of the non-overlapping length-k
                    blocks of a string -- protocol floor 5, the level BDM 1.0
                    converges to (R6)
    bdm1            BDM 1.0 via pybdm. CTM coverage in pybdm 0.1.0:
                    1D alphabets {2, 4, 5, 6, 9} at block size 12, and 2D
                    binary at 4x4 -- nothing else (measured; the 2D
                    alphabet-4 table the config note expected does not
                    ship). DNA strings (alphabet 4) therefore use 1D BDM
                    with 12-symbol blocks.
    ncd             normalized compression distance per D7,
                    NCD(x,y) = (C(xy) - min(Cx,Cy)) / max(Cx,Cy),
                    compressors {zlib-9, bz2, lzma} on bytes

All functions are pure and deterministic; compressor levels are printed by
the producers that use them (G3).
"""

from __future__ import annotations

import bz2
import lzma
import math
import zlib
from collections import Counter

import numpy as np

__all__ = ["COMPRESSORS", "compressed_size", "gzip_ratio", "block_entropy",
           "bdm1", "ncd", "symbols_to_array"]

COMPRESSORS = {"zlib9": lambda b: zlib.compress(b, 9),
               "bz2": lambda b: bz2.compress(b),
               "lzma": lambda b: lzma.compress(b)}


def symbols_to_array(s: str) -> np.ndarray:
    """Alphabet-4 symbol string -> uint8 array with values 0..3 (the fixed
    symbol order is the string's own sorted distinct symbols, which for
    ``dna.ACTION_SYMBOLS``/``TEMPORAL_SYMBOLS`` is already 0..3)."""
    a = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    vals = sorted(set(a.tolist()))
    lut = np.zeros(256, dtype=np.uint8)
    for new, old in enumerate(vals):
        lut[old] = new
    return lut[a]


def compressed_size(data: bytes, comp: str = "zlib9") -> int:
    return len(COMPRESSORS[comp](data))


def gzip_ratio(s: str, comp: str = "zlib9") -> float:
    """Floor 4 (protocol §3): compressed size over raw size."""
    b = s.encode("ascii")
    if not b:
        return float("nan")
    return compressed_size(b, comp) / len(b)


def block_entropy(s: str, block_size: int = 12) -> float:
    """Shannon entropy (bits) of the non-overlapping length-k blocks
    (trailing partial block ignored). Floor 5 (protocol §3)."""
    if len(s) < block_size:
        return float("nan")
    blocks = [s[i:i + block_size]
              for i in range(0, len(s) - block_size + 1, block_size)]
    n = len(blocks)
    c = Counter(blocks)
    return float(-sum((v / n) * math.log2(v / n) for v in c.values()))


def bdm1(s: str, nsymbols: int = 4):
    """BDM 1.0 of a symbol string via pybdm, 1D, 12-symbol blocks
    (the tabulated CTM coverage; see module docstring)."""
    from pybdm import BDM
    arr = symbols_to_array(s)
    b = BDM(ndim=1, nsymbols=nsymbols, shape=(12,),
            partition=__import__("pybdm.partitions", fromlist=["PartitionIgnore"]).PartitionIgnore)
    return float(b.bdm(arr))


def ncd(x: str | bytes, y: str | bytes, comp: str = "zlib9") -> float:
    """D7 normalized compression distance on bytes."""
    bx = x.encode("ascii") if isinstance(x, str) else x
    by = y.encode("ascii") if isinstance(y, str) else y
    cx, cy = compressed_size(bx, comp), compressed_size(by, comp)
    return (compressed_size(bx + by, comp) - min(cx, cy)) / max(cx, cy)
