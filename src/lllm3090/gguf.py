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


def _read(fh: BinaryIO, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise Malformed(f"wanted {n} bytes, got {len(buf)}")
    return buf


def _u32(fh: BinaryIO) -> int:
    return int(struct.unpack("<I", _read(fh, 4))[0])


def _u64(fh: BinaryIO) -> int:
    return int(struct.unpack("<Q", _read(fh, 8))[0])


def _string(fh: BinaryIO) -> str:
    return _read(fh, _u64(fh)).decode("utf-8", "replace")


def _value(fh: BinaryIO, kind: int) -> Any:
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
            _read(fh, struct.calcsize(_FIXED[element]) * count)
        else:
            for _ in range(count):
                _value(fh, element)
        return f"<array of {count}>"
    raise Malformed(f"unknown metadata type {kind}")


def header(path: Path | str) -> tuple[dict[str, Any], list[str]]:
    """A checkpoint's metadata and tensor names, read from its header alone."""
    with open(path, "rb") as fh:
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
            _read(fh, 8 * dims)  # shape
            _u32(fh)             # ggml type
            _u64(fh)             # offset into the tensor blob
    return meta, names


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
    except (OSError, Malformed, struct.error):
        return False
    return any(
        marker in name.lower() for name in names for marker in MTP_MARKERS
    )
