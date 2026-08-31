"""Read what a GGUF says about itself, without loading it.

A GGUF carries its metadata and its full tensor list in a header, before any
weights. That is enough to answer questions the catalogue cannot answer from a
file size -- currently one question: does this checkpoint carry a multi-token
prediction head?

Everything here reads the header and stops. An 18 GB checkpoint is inspected in
well under a second, because the tokenizer vocabulary is skipped rather than
materialised.

Why this is derived rather than declared: the catalogue can *say* a repo ships
MTP, but only the file on disk decides whether the engine can use it. Two
builds of the same model differ, quantisers strip the head, and a partially
downloaded file has no tensors at all. A flag passed to llama.cpp on the
strength of a YAML field would produce an engine that fails at load; read from
the file it cannot be wrong.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

#: GGUF metadata value types, by their on-disk tag.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

#: The fixed-width types, and the struct format that reads each.
_FIXED = {
    U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i",
    F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d",
}

#: Substrings that identify a multi-token prediction head in a tensor name.
#:
#: Qwen calls it ``nextn`` and puts it in one extra block past the last real
#: layer -- ``blk.64.nextn.*`` on the 27B, ``blk.40.nextn.*`` on the 35B-A3B.
#: DeepSeek uses the same word. ``mtp`` is accepted for anything that spells it
#: the other way.
MTP_MARKERS = ("nextn", "mtp")


class Malformed(Exception):
    """This file is not a GGUF, or its header does not parse."""


class _Reader:
    """A file, plus the one fact needed to refuse a nonsense length.

    Every length in a GGUF is read *from* the GGUF, so a corrupt or hostile
    header can ask for one near ``2**64``. Handing that to ``read()`` raises
    ``OverflowError`` or exhausts memory rather than failing as a parse error,
    and the caller is a start-up path that must do neither. The file's size is
    taken once here so the check costs nothing per read -- a modern vocabulary
    is 150k strings, and two extra seeks apiece would be felt.
    """

    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.size = fh.seek(0, 2)
        fh.seek(0)
        # Tracked rather than asked for. ``tell()`` per read is not free, and
        # a modern vocabulary is 150k strings: calling it made reading a real
        # checkpoint four times slower than counting the bytes here.
        self.pos = 0

    def read(self, n: int) -> bytes:
        if not 0 <= n <= self.size - self.pos:
            raise Malformed(f"length {n} is not inside a {self.size}-byte file")
        buf = self.fh.read(n)
        if len(buf) != n:
            raise Malformed(f"wanted {n} bytes, got {len(buf)}")
        self.pos += n
        return buf

    def skip(self, n: int) -> None:
        """Step over ``n`` bytes without materialising them."""
        if not 0 <= n <= self.size - self.pos:
            raise Malformed(f"skip of {n} is not inside a {self.size}-byte file")
        self.fh.seek(n, 1)
        self.pos += n


def _read(fh: _Reader, n: int) -> bytes:
    return fh.read(n)


def _u32(fh: _Reader) -> int:
    return int(struct.unpack("<I", _read(fh, 4))[0])


def _u64(fh: _Reader) -> int:
    return int(struct.unpack("<Q", _read(fh, 8))[0])


def _string(fh: _Reader) -> str:
    return _read(fh, _u64(fh)).decode("utf-8", "replace")


def _value(fh: _Reader, kind: int) -> Any:
    """One metadata value, returning a placeholder for the bulky ones.

    Arrays are where the tokenizer vocabulary lives -- a hundred thousand
    strings on a modern model. They are stepped over rather than decoded,
    because nothing here needs them and building the list is most of the cost
    of reading the header at all.
    """
    if kind in _FIXED:
        fmt = _FIXED[kind]
        return struct.unpack("<" + fmt, _read(fh, struct.calcsize(fmt)))[0]
    if kind == STRING:
        return _string(fh)
    if kind == ARRAY:
        element, count = _u32(fh), _u64(fh)
        if element in _FIXED:
            fh.skip(struct.calcsize(_FIXED[element]) * count)
        else:
            for _ in range(count):
                _value(fh, element)
        return f"<array of {count}>"
    raise Malformed(f"unknown metadata type {kind}")


def header(path: Path | str) -> tuple[dict[str, Any], list[str]]:
    """A checkpoint's metadata and tensor names, read from its header alone."""
    with open(path, "rb") as raw:
        fh = _Reader(raw)
        if _read(fh, 4) != b"GGUF":
            raise Malformed("no GGUF magic; this is not a checkpoint")
        _u32(fh)  # format version, unused: the layout below is stable across 2-3
        tensors, pairs = _u64(fh), _u64(fh)
        meta: dict[str, Any] = {}
        for _ in range(pairs):
            key = _string(fh)
            meta[key] = _value(fh, _u32(fh))
        names = []
        for _ in range(tensors):
            names.append(_string(fh))
            dims = _u32(fh)
            fh.skip(8 * dims)    # shape
            _u32(fh)             # ggml type
            _u64(fh)             # offset into the tensor blob
    return meta, names


def full_attention_layers(path: Path | str) -> int | None:
    """How many of this checkpoint's layers keep a KV cache, or None.

    The catalogue's ``kv_kib_per_token`` is
    ``kv_heads x (key_length + value_length) x 2 bytes x`` this number, and on
    the hybrid models that carry an MTP head it reproduces the hand-entered
    figures exactly. So this is what lets a declared field be *checked* rather
    than trusted.

    Two things make it less obvious than counting blocks:

    * ``full_attention_interval`` -- these are hybrids, and only one layer in
      four keeps a cache. The rest are SSM, whose state is per-sequence rather
      than per-token and therefore costs nothing as context grows.
    * ``block_count`` **includes the MTP head**, which is why it reads 65 for a
      64-layer model and 41 for a 40-layer one. Subtracting it is the whole
      reason the head can be priced as "one more of these".

    ``None`` for an architecture that is not shaped this way -- Gemma-4 with
    its sliding-window pattern and per-layer head counts, gpt-oss with no
    interval at all. That is not a failure: none of them carries an MTP head,
    so nothing needs the number.
    """
    try:
        meta, _ = header(path)
    except (OSError, Malformed, struct.error, ValueError, MemoryError):
        return None
    blocks = next(
        (v for k, v in meta.items() if k.endswith(".block_count")), None
    )
    interval = next(
        (v for k, v in meta.items() if k.endswith(".full_attention_interval")),
        None,
    )
    if not isinstance(blocks, int) or not isinstance(interval, int):
        return None
    if interval <= 0 or blocks <= 1:
        return None
    return (blocks - 1) // interval


def has_mtp(path: Path | str) -> bool:
    """Whether this checkpoint carries a usable multi-token prediction head.

    Requires the *tensors*, not merely the metadata key. A conversion can
    announce ``nextn_predict_layers`` and still ship no head, and llama.cpp
    refuses to start with ``--spec-type draft-mtp`` against a checkpoint that
    has none -- so the tensors are the only honest test.

    Any failure to read the file is a ``False``: this decides whether to add a
    flag to a working command line, and the cost of being wrong in that
    direction is an engine that will not start.
    """
    try:
        _, names = header(path)
    except (OSError, Malformed, struct.error, ValueError, MemoryError):
        # Every failure means "no MTP". This decides whether to append a flag
        # to a working command line: guessing yes on a file that could not be
        # parsed produces an engine that exits at load, guessing no produces
        # one that runs a little slower. ValueError covers OverflowError.
        return False
    return any(
        marker in name.lower() for name in names for marker in MTP_MARKERS
    )
