"""The GGUF header reader, against files built byte by byte.

Real checkpoints are 15-20 GB and are not in the repository, so the fixtures
here are synthesised to the format's own layout. That is the point: if the
writer below and the reader in ``lllm3090.gguf`` agree, they agree about the
format rather than about one vendor's output.
"""

from __future__ import annotations

import struct

import pytest

from lllm3090 import gguf


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def build(tensors: list[str], meta: dict[str, int] | None = None) -> bytes:
    """A minimal but structurally valid GGUF header."""
    meta = meta or {}
    out = [b"GGUF", struct.pack("<I", 3),
           struct.pack("<Q", len(tensors)), struct.pack("<Q", len(meta))]
    for key, value in meta.items():
        out += [_s(key), struct.pack("<I", gguf.U32), struct.pack("<I", value)]
    for name in tensors:
        out += [_s(name), struct.pack("<I", 2),        # two dimensions
                struct.pack("<QQ", 4096, 4096),        # shape
                struct.pack("<I", 0),                  # ggml type
                struct.pack("<Q", 0)]                  # offset
    return b"".join(out)


def test_it_reads_metadata_and_every_tensor_name(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(build(["blk.0.attn_q.weight", "output.weight"],
                        {"qwen35.block_count": 64}))
    meta, names = gguf.header(f)
    assert meta["qwen35.block_count"] == 64
    assert names == ["blk.0.attn_q.weight", "output.weight"]


def test_an_mtp_head_is_found_by_its_tensors(tmp_path):
    """Qwen puts it one block past the last real layer, and calls it nextn."""
    f = tmp_path / "mtp.gguf"
    f.write_bytes(build([
        "blk.63.attn_q.weight",
        "blk.64.nextn.eh_proj.weight",
        "blk.64.nextn.enorm.weight",
    ]))
    assert gguf.has_mtp(f) is True


def test_metadata_alone_is_not_an_mtp_head(tmp_path):
    """The decisive case, and the reason this reads tensors rather than keys.

    A conversion can announce ``nextn_predict_layers`` and ship no head --
    the key describes the model it came from, the tensors describe the file.
    llama.cpp fails at load if the flag is passed without them, so believing
    the key would turn a working start into a broken one.
    """
    f = tmp_path / "claims.gguf"
    f.write_bytes(build(["blk.0.attn_q.weight"],
                        {"qwen35.nextn_predict_layers": 1}))
    meta, _ = gguf.header(f)
    assert meta["qwen35.nextn_predict_layers"] == 1
    assert gguf.has_mtp(f) is False


def test_a_large_array_is_stepped_over_rather_than_decoded(tmp_path):
    """The tokenizer vocabulary is most of the header and none of the answer."""
    vocab = b"".join(_s(f"token{i}") for i in range(5000))
    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
            + struct.pack("<Q", 1)
            + _s("tokenizer.ggml.tokens") + struct.pack("<I", gguf.ARRAY)
            + struct.pack("<I", gguf.STRING) + struct.pack("<Q", 5000) + vocab
            + _s("blk.40.nextn.eh_proj.weight") + struct.pack("<I", 1)
            + struct.pack("<Q", 8) + struct.pack("<I", 0) + struct.pack("<Q", 0))
    f = tmp_path / "big.gguf"
    f.write_bytes(body)
    meta, names = gguf.header(f)
    assert meta["tokenizer.ggml.tokens"] == "<array of 5000>"
    assert names == ["blk.40.nextn.eh_proj.weight"]
    assert gguf.has_mtp(f) is True


@pytest.mark.parametrize("body", [
    b"",                                  # empty
    b"NOPE" + b"\0" * 32,                 # wrong magic
    b"GGUF" + struct.pack("<I", 3),       # truncated before the counts
    b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 9999, 0),  # lies
])
def test_an_unreadable_file_never_adds_a_flag(tmp_path, body):
    """Failure has to mean "no MTP", because the alternative is a dead engine.

    ``has_mtp`` decides whether to append an argument to a working command
    line. Guessing yes on a file it could not parse produces an engine that
    exits at load; guessing no produces one that runs slightly slower.
    """
    f = tmp_path / "broken.gguf"
    f.write_bytes(body)
    assert gguf.has_mtp(f) is False


def test_the_header_read_does_not_depend_on_the_weights(tmp_path):
    """It must work on a file whose tensor data was never written.

    A checkpoint is inspected at start time, and reading 18 GB to answer a
    yes/no question would put a visible pause in front of every launch.
    """
    f = tmp_path / "headeronly.gguf"
    f.write_bytes(build(["blk.40.nextn.enorm.weight"]))
    assert f.stat().st_size < 4096
    assert gguf.has_mtp(f) is True


@pytest.mark.parametrize("length", [2**63, 2**64 - 1, 10**9])
def test_a_length_the_file_cannot_back_is_a_parse_error_not_a_crash(tmp_path, length):
    """Every length here is read *from* the file, so it is attacker-controlled.

    Handing one near 2**64 to ``read()`` raises ``OverflowError`` or exhausts
    memory rather than failing as a parse error -- and the caller is
    ``engine.start``, which must do neither. It has to come back as "no MTP".
    """
    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
            + struct.pack("<Q", 0) + struct.pack("<Q", length) + b"short")
    f = tmp_path / "huge.gguf"
    f.write_bytes(body)
    with pytest.raises(gguf.Malformed):
        gguf.header(f)
    assert gguf.has_mtp(f) is False


def test_an_oversized_array_count_is_refused(tmp_path):
    """The same, for the element count of a fixed-width array."""
    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", 1) + _s("k") + struct.pack("<I", gguf.ARRAY)
            + struct.pack("<I", gguf.U32) + struct.pack("<Q", 2**62))
    f = tmp_path / "arr.gguf"
    f.write_bytes(body)
    assert gguf.has_mtp(f) is False


def test_both_facts_come_from_one_read(tmp_path, monkeypatch):
    """`has_mtp` and `full_attention_layers` used to parse the file each.

    On an 18 GB checkpoint that is the same multi-megabyte header walked twice
    to answer two questions that were both settled by the first walk.
    """
    f = tmp_path / "both.gguf"
    f.write_bytes(build(["blk.40.nextn.enorm.weight"],
                        {"qwen35.block_count": 41,
                         "qwen35.full_attention_interval": 4}))
    reads = 0
    real = gguf.header

    def counted(path):
        nonlocal reads
        reads += 1
        return real(path)

    monkeypatch.setattr(gguf, "header", counted)
    assert gguf.has_mtp(f) is True
    assert gguf.full_attention_layers(f) == 10
    assert reads == 1


def test_a_second_look_at_the_same_file_does_not_read_it_again(tmp_path, monkeypatch):
    """The panel polls `/api/status` every two seconds, and the catalogue is
    nine checkpoints. Re-reading them all costs a third of the interval."""
    f = tmp_path / "memo.gguf"
    f.write_bytes(build(["blk.64.nextn.eh_proj.weight"]))
    assert gguf.facts(f).mtp is True

    def refuse(path):
        raise AssertionError("the header was read a second time")

    monkeypatch.setattr(gguf, "header", refuse)
    assert gguf.facts(f).mtp is True


def test_a_file_that_changes_underneath_is_read_again(tmp_path):
    """Keyed on the path alone this is the dangerous case: a re-download turns
    a checkpoint with a head into one without, and the flag that follows from
    the stale answer is one llama.cpp refuses to start with."""
    f = tmp_path / "changing.gguf"
    f.write_bytes(build(["blk.64.nextn.eh_proj.weight"]))
    assert gguf.has_mtp(f) is True
    f.write_bytes(build(["blk.0.attn_q.weight", "output.weight"]))
    assert gguf.has_mtp(f) is False


def test_a_file_that_could_not_be_read_is_not_remembered(tmp_path):
    """A download in progress is the overwhelmingly likely reason for a failed
    parse, and it stops being true without anything else changing."""
    f = tmp_path / "arriving.gguf"
    f.write_bytes(b"GGUF")
    assert gguf.has_mtp(f) is False
    f.write_bytes(build(["blk.64.nextn.eh_proj.weight"]))
    assert gguf.has_mtp(f) is True


def test_the_cache_cannot_grow_without_bound(tmp_path):
    """A download in progress mints a fresh key on every mtime change, and the
    panel is a process that runs for weeks."""
    f = tmp_path / "growing.gguf"
    for i in range(gguf._FACTS_MAX + 10):
        f.write_bytes(build([f"blk.{i}.attn_q.weight"] * (i + 1)))
        gguf.facts(f)
    assert len(gguf._FACTS) <= gguf._FACTS_MAX


def test_a_vocabulary_larger_than_one_read_is_still_walked(tmp_path):
    """The header is pulled in in chunks, so the interesting case is a string
    array that straddles the boundary -- which every real model's does."""
    tokens = b"".join(_s("t" * 200) for _ in range(30_000))
    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
            + struct.pack("<Q", 1)
            + _s("tokenizer.ggml.tokens") + struct.pack("<I", gguf.ARRAY)
            + struct.pack("<I", gguf.STRING) + struct.pack("<Q", 30_000)
            + tokens
            + _s("blk.64.nextn.eh_proj.weight") + struct.pack("<I", 1)
            + struct.pack("<Q", 8) + struct.pack("<I", 0) + struct.pack("<Q", 0))
    f = tmp_path / "chunky.gguf"
    f.write_bytes(body)
    assert len(body) > gguf._CHUNK, "the fixture has to cross a chunk boundary"
    assert gguf.has_mtp(f) is True


def test_a_string_array_that_runs_off_the_end_is_a_parse_error(tmp_path):
    """Its count and every length in it come from the file, so it can claim
    more strings than the file holds."""
    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", 1) + _s("tokenizer.ggml.tokens")
            + struct.pack("<I", gguf.ARRAY) + struct.pack("<I", gguf.STRING)
            + struct.pack("<Q", 5000) + _s("only-one"))
    f = tmp_path / "short-array.gguf"
    f.write_bytes(body)
    with pytest.raises(gguf.Malformed):
        gguf.header(f)
    assert gguf.has_mtp(f) is False
