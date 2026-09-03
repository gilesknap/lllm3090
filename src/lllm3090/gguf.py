"""Read what a GGUF says about itself, without loading it.

A GGUF carries its metadata and its full tensor list in a header, before any
weights. That is enough to answer questions the catalogue cannot answer from a
file size -- currently one question: does this checkpoint carry a multi-token
prediction head?

Everything here reads the header and stops. An 18 GB checkpoint is inspected in
tens of milliseconds, because the tokenizer vocabulary is walked as offsets
rather than materialised as strings.

Why this is derived rather than declared: the catalogue can *say* a repo ships
MTP, but only the file on disk decides whether the engine can use it. Two
builds of the same model differ, quantisers strip the head, and a partially
downloaded file has no tensors at all. A flag passed to llama.cpp on the
strength of a YAML field would produce an engine that fails at load; read from
the file it cannot be wrong.

**The header is not free, and the panel asks every two seconds.** Reading all
nine catalogued checkpoints takes about 0.65 s, which is why :func:`facts`
memoises on the file's identity rather than its name -- see its docstring for
what that costs and what invalidates it.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple

#: GGUF metadata value types, by their on-disk tag.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

#: The fixed-width types, and the struct format that reads each.
_FIXED = {
    U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i",
    F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d",
}

#: How much of the file to pull in at a time.
#:
#: A real header is a few megabytes -- almost all of it vocabulary -- so this
#: reads most of them in one call and the rest in two, while never reading a
#: whole checkpoint to answer a question about its first pages.
_CHUNK = 4 << 20

#: Substrings that identify a multi-token prediction head in a tensor name.
#:
#: Qwen calls it ``nextn`` and puts it in one extra block past the last real
#: layer -- ``blk.64.nextn.*`` on the 27B, ``blk.40.nextn.*`` on the 35B-A3B.
#: DeepSeek uses the same word. ``mtp`` is accepted for anything that spells it
#: the other way.
MTP_MARKERS = ("nextn", "mtp")


class Malformed(Exception):
    """This file is not a GGUF, or its header does not parse."""


class _Head:
    """The front of the file in memory, grown forward as the parse walks it.

    The parse is a walk over lengths that are themselves read *from the file*,
    so a corrupt or hostile header can ask to step 2**64 bytes forward. Every
    such step goes through :meth:`upto`, which refuses anything the file cannot
    back -- the caller is ``engine.start``, which must not raise ``OverflowError``
    or exhaust memory on a bad checkpoint.

    In memory rather than seeked over, because the cost here is per *element*
    and a modern vocabulary has 150k of them: three method calls and a buffered
    read apiece was 4.5x slower than walking offsets in a ``bytes`` already
    held. Not ``mmap``, which would be faster still and can take the whole
    process down with ``SIGBUS`` if the mapped file is truncated underneath it
    -- which is exactly what a re-download does.
    """

    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.size = fh.seek(0, 2)
        self.buf = b""

    def upto(self, end: int) -> bytes:
        """The file's first ``end`` bytes, reading more if they are not held."""
        if not 0 <= end <= self.size:
            raise Malformed(f"offset {end} is not inside a {self.size}-byte file")
        if end > len(self.buf):
            want = min(self.size, max(end, len(self.buf) + _CHUNK))
            self.fh.seek(len(self.buf))
            more = self.fh.read(want - len(self.buf))
            if len(self.buf) + len(more) < end:
                raise Malformed(f"wanted {end} bytes, the file gave fewer")
            self.buf += more
        return self.buf


def _u64_at(buf: bytes, pos: int) -> int:
    return int.from_bytes(buf[pos:pos + 8], "little")


def _string_at(head: _Head, pos: int) -> tuple[int, str]:
    """One length-prefixed UTF-8 string, and where it ends."""
    buf = head.upto(pos + 8)
    length = _u64_at(buf, pos)
    pos += 8
    buf = head.upto(pos + length)
    return pos + length, buf[pos:pos + length].decode("utf-8", "replace")


def _skip_strings(head: _Head, pos: int, count: int) -> int:
    """Step over ``count`` length-prefixed strings, decoding none of them.

    This is the tokenizer vocabulary, which is most of the header and none of
    the answer. It is written out flat rather than as a loop over
    :func:`_string_at` because it is the only part of the parse whose cost
    scales with the model rather than with the format: 150k iterations, where
    each avoided function call is worth more than it looks.
    """
    buf = head.upto(min(head.size, pos + _CHUNK))
    limit = len(buf) - 8
    # Bound as a local and the decode written out: `_u64_at` here rather than
    # inline costs a fifth of the whole read across the catalogue, because the
    # loop body runs 150k times per model and does nothing else.
    from_bytes = int.from_bytes
    for _ in range(count):
        if pos > limit:
            buf = head.upto(min(head.size, pos + _CHUNK))
            limit = len(buf) - 8
            if pos > limit:
                raise Malformed("a string array runs past the end of the file")
        pos += 8 + from_bytes(buf[pos:pos + 8], "little")
    # The final string's *bytes* still have to be inside the file; the loop
    # only proved that each length prefix was.
    head.upto(pos)
    return pos


def _value(head: _Head, pos: int, kind: int) -> tuple[int, Any]:
    """One metadata value, returning a placeholder for the bulky ones.

    Arrays are where the tokenizer vocabulary lives -- a hundred thousand
    strings on a modern model. They are stepped over rather than decoded,
    because nothing here needs them and building the list is most of the cost
    of reading the header at all.
    """
    if kind in _FIXED:
        fmt = _FIXED[kind]
        size = struct.calcsize(fmt)
        buf = head.upto(pos + size)
        return pos + size, struct.unpack_from("<" + fmt, buf, pos)[0]
    if kind == STRING:
        return _string_at(head, pos)
    if kind == ARRAY:
        buf = head.upto(pos + 12)
        element, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if element in _FIXED:
            # `count` is attacker-controlled and multiplied, so the product is
            # checked rather than the count: 2**62 fixed-width elements is a
            # step no file can back.
            end = pos + struct.calcsize(_FIXED[element]) * count
            head.upto(end)
            return end, f"<array of {count}>"
        if element == STRING:
            return _skip_strings(head, pos, count), f"<array of {count}>"
        for _ in range(count):
            pos, _unused = _value(head, pos, element)
        return pos, f"<array of {count}>"
    raise Malformed(f"unknown metadata type {kind}")


def header(path: Path | str) -> tuple[dict[str, Any], list[str]]:
    """A checkpoint's metadata and tensor names, read from its header alone."""
    with open(path, "rb") as raw:
        head = _Head(raw)
        if head.size < 24 or head.upto(4)[:4] != b"GGUF":
            raise Malformed("no GGUF magic; this is not a checkpoint")
        # Byte 4 is the format version, unused: the layout below is stable
        # across 2-3.
        buf = head.upto(24)
        tensors, pairs = struct.unpack_from("<QQ", buf, 8)
        pos = 24
        meta: dict[str, Any] = {}
        for _ in range(pairs):
            pos, key = _string_at(head, pos)
            buf = head.upto(pos + 4)
            kind = struct.unpack_from("<I", buf, pos)[0]
            pos, meta[key] = _value(head, pos + 4, kind)
        names = []
        for _ in range(tensors):
            pos, name = _string_at(head, pos)
            names.append(name)
            buf = head.upto(pos + 4)
            dims = struct.unpack_from("<I", buf, pos)[0]
            # The shape, then the ggml type and the offset into the tensor
            # blob. None of the three is read; stepping over them is checked
            # because `dims` came out of the file like everything else.
            pos += 4 + 8 * dims + 12
            head.upto(pos)
    return meta, names


class Facts(NamedTuple):
    """What the file itself says, as opposed to what ``models.yaml`` declares.

    Both fields are exactly what :func:`has_mtp` and
    :func:`full_attention_layers` return, and both are derived from one read.
    They used to be two, which meant a caller wanting both parsed an 18 GB
    checkpoint's header twice.
    """

    #: A usable multi-token prediction head is present.
    mtp: bool
    #: How many layers keep a KV cache, or None where the question does not
    #: apply to this architecture.
    full_attention_layers: int | None


#: Parsed headers, keyed on ``(path, size, mtime_ns)``.
#:
#: The stat is the point. Keyed on the path alone, a re-download or a truncated
#: part file would keep answering with the old file's head, and the flag that
#: follows from it decides whether llama.cpp starts at all. Keyed on identity,
#: a file that changes mints a new entry and the stale one falls out.
_FACTS: dict[tuple[str, int, int], Facts] = {}

#: How many to keep. Nine checkpoints is a full disk here, and a download in
#: progress mints a fresh entry on every mtime change -- so this is bounded to
#: stop a long-lived panel accumulating one entry per second of a download.
_FACTS_MAX = 64


def facts(path: Path | str) -> Facts:
    """Everything derived from this checkpoint's header, read at most once.

    A failed read is not cached: the overwhelmingly likely reason is a file
    that is still arriving, and remembering "not a GGUF" about a checkpoint
    that is about to become one would need a second invalidation rule to undo.
    Re-reading a header that fails to parse is cheap -- it fails in the first
    few bytes.
    """
    try:
        stat = Path(path).stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return Facts(False, None)
    hit = _FACTS.get(key)
    if hit is not None:
        return hit
    try:
        meta, names = header(path)
    except (OSError, Malformed, struct.error, ValueError, MemoryError):
        # Every failure means "no MTP". This decides whether to append a flag
        # to a working command line: guessing yes on a file that could not be
        # parsed produces an engine that exits at load, guessing no produces
        # one that runs a little slower. ValueError covers OverflowError.
        return Facts(False, None)
    found = Facts(
        mtp=any(m in n.lower() for n in names for m in MTP_MARKERS),
        full_attention_layers=_full_attention_layers(meta),
    )
    if len(_FACTS) >= _FACTS_MAX:
        del _FACTS[next(iter(_FACTS))]
    _FACTS[key] = found
    return found


def _full_attention_layers(meta: dict[str, Any]) -> int | None:
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
    return facts(path).full_attention_layers


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
    return facts(path).mtp
