"""The CLM GGUF's bare `pooling_type` key is renamed to `<arch>.pooling_type` in place, so Ollama
registers the model as an embedder, without moving the tensor data or changing the file size."""
import struct

import pytest

from hybrid_intent_router import gguf_pooling

pytestmark = pytest.mark.unit

DATA = b"\x07" * 64  # stands in for the tensor weights


def _s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def _gguf(path, kvs, pad_to=None):
    """A minimal GGUF v3: the given kvs, one 1-D tensor, 32-byte aligned data."""
    body = b"".join(kvs)
    tensor = _s(b"w") + struct.pack("<I", 1) + struct.pack("<Q", 64) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, len(kvs)) + body + tensor
    offset = pad_to or -(-len(head) // 32) * 32
    path.write_bytes(head + b"\0" * (offset - len(head)) + DATA)
    return offset


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", 8) + _s(v)


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _keys(path):
    kvs, _, _, _, _ = gguf_pooling._parse(path.read_bytes())
    return {k: raw for k, raw in kvs}


def test_renames_the_key_and_keeps_data_and_size(tmp_path):
    f = tmp_path / "clm.gguf"
    kvs = [_kv_str(b"general.architecture", b"qwen3"), _kv_u32(b"pooling_type", 3)]
    head_len = 24 + sum(map(len, kvs)) + len(_s(b"w")) + 4 + 8 + 4 + 8
    for n in range(32):  # a path length that leaves 4 bytes of padding, as in the published file
        path = b"/home/" + b"z" * n + b"/imatrix.gguf"
        if -(head_len + len(_kv_str(b"quantize.imatrix.file", path))) % 32 == 4:
            break
    offset = _gguf(f, kvs + [_kv_str(b"quantize.imatrix.file", path)])
    size = f.stat().st_size
    assert gguf_pooling.fix_pooling_key(str(f)) == "renamed pooling_type to qwen3.pooling_type"
    keys = _keys(f)
    assert b"qwen3.pooling_type" in keys and b"pooling_type" not in keys
    assert keys[b"qwen3.pooling_type"].endswith(struct.pack("<I", 3))
    assert keys[b"quantize.imatrix.file"].endswith(b"/imatrix.gguf")  # a shorter tail of the same path
    assert f.stat().st_size == size
    assert f.read_bytes()[offset:] == DATA
    assert gguf_pooling._parse(f.read_bytes())[3] == offset


def test_is_idempotent(tmp_path):
    f = tmp_path / "clm.gguf"
    _gguf(f, [_kv_str(b"general.architecture", b"qwen3"), _kv_u32(b"pooling_type", 3),
              _kv_str(b"quantize.imatrix.file", b"/a/b/imatrix.gguf")])
    gguf_pooling.fix_pooling_key(str(f))
    before = f.read_bytes()
    assert gguf_pooling.fix_pooling_key(str(f)) == "already declares qwen3.pooling_type"
    assert f.read_bytes() == before


def test_refuses_to_move_the_tensor_data(tmp_path):
    f = tmp_path / "tight.gguf"
    kvs = [_kv_str(b"general.architecture", b"qwen3"), _kv_u32(b"pooling_type", 3)]
    head_len = 24 + sum(map(len, kvs)) + len(_s(b"w")) + 4 + 8 + 4 + 8
    kvs.append(_kv_str(b"pad", b"x" * (-(head_len + 8 + 3 + 4 + 8) % 32)))  # header ends exactly on the data
    _gguf(f, kvs)
    before = f.read_bytes()
    with pytest.raises(gguf_pooling.GGUFError, match="move the tensor data"):
        gguf_pooling.fix_pooling_key(str(f))
    assert f.read_bytes() == before


def test_cli_reports_a_non_gguf_file(tmp_path, capsys):
    f = tmp_path / "x.gguf"
    f.write_bytes(b"not a gguf")
    assert gguf_pooling.main([str(f)]) == 1
    assert "not a GGUF file" in capsys.readouterr().err
