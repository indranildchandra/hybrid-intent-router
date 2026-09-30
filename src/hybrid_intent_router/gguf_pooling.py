"""Make the CLM GGUF declare its pooling where Ollama looks for it.

The published CLM GGUF stores last-token pooling under a bare `pooling_type` key. Ollama and
llama.cpp read `<architecture>.pooling_type` (here `qwen3.pooling_type`), so without the rename
Ollama registers the model for completion only and /api/embed answers HTTP 501 ("does not support
embeddings"). The rename is done in place: the header grows by the architecture prefix, and when the
alignment padding is too small for that, the quantizer's local imatrix path is shortened to pay for
it. The tensor data offset, the file size and every weight stay as downloaded. Idempotent.

    python -m hybrid_intent_router.gguf_pooling <file.gguf>
"""
import struct
import sys

_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # GGUF type -> bytes
_STRING, _ARRAY = 8, 9
_HEADER_READ = 64 << 20  # far larger than any header; the CLM one is about 6 MB


class GGUFError(ValueError):
    pass


def _parse(head: bytes):
    """Returns ([(key, raw kv bytes)], tensor info bytes, architecture, data offset)."""
    if head[:4] != b"GGUF":
        raise GGUFError("not a GGUF file")
    n_tensors, n_kv = struct.unpack_from("<QQ", head, 8)
    pos = 24

    def string():
        nonlocal pos
        n, = struct.unpack_from("<Q", head, pos)
        pos += 8 + n
        return head[pos - n:pos]

    def skip(t):
        nonlocal pos
        if t == _STRING:
            string()
        elif t == _ARRAY:
            et, n = struct.unpack_from("<IQ", head, pos)
            pos += 12
            if et in _SCALAR:
                pos += _SCALAR[et] * n
            else:
                for _ in range(n):
                    skip(et)
        else:
            pos += _SCALAR[t]

    kvs, arch, align = [], None, 32
    for _ in range(n_kv):
        start = pos
        key = string()
        t, = struct.unpack_from("<I", head, pos)
        pos += 4
        value = pos
        skip(t)
        if key == b"general.architecture":
            arch = head[value + 8:pos]
        elif key == b"general.alignment":
            align, = struct.unpack_from("<I", head, value)
        kvs.append((key, head[start:pos]))
    kv_end = pos
    for _ in range(n_tensors):
        string()
        n_dims, = struct.unpack_from("<I", head, pos)
        pos += 4 + 8 * n_dims + 4 + 8
    if pos > len(head):
        raise GGUFError("header is truncated or larger than expected")
    return kvs, head[kv_end:pos], arch, -(-pos // align) * align, align


def fix_pooling_key(path: str) -> str:
    """Renames `pooling_type` to `<arch>.pooling_type` in place. Returns what was done."""
    with open(path, "r+b") as f:
        head = f.read(_HEADER_READ)
        kvs, tensors, arch, data_offset, align = _parse(head)
        if arch is None:
            raise GGUFError("no general.architecture key")
        target = arch + b".pooling_type"
        keys = [k for k, _ in kvs]
        if target in keys:
            return f"already declares {target.decode()}"
        if b"pooling_type" not in keys:
            raise GGUFError("no pooling_type key: this GGUF does not declare pooling")

        def rebuild(imatrix_path=None):
            out = []
            for key, raw in kvs:
                if key == b"pooling_type":
                    raw = struct.pack("<Q", len(target)) + target + raw[8 + len(key):]
                elif key == b"quantize.imatrix.file" and imatrix_path is not None:
                    raw = raw[:8 + len(key) + 4] + struct.pack("<Q", len(imatrix_path)) + imatrix_path
                out.append(raw)
            return head[:24] + b"".join(out) + tensors

        new = rebuild()
        excess = len(new) - data_offset
        if excess > 0 and b"quantize.imatrix.file" in keys:
            # The padding cannot absorb the prefix, so the quantizer's local imatrix path pays for it:
            # the longest tail that starts after a "/" and fits, else the path minus exactly `excess` bytes.
            raw = dict(kvs)[b"quantize.imatrix.file"]
            path = raw[8 + len(b"quantize.imatrix.file") + 4 + 8:]
            tails = [path[i + 1:] for i, ch in enumerate(path) if ch == ord("/")] + [path[excess:]]
            new = next((h for h in map(rebuild, tails) if data_offset - align < len(h) <= data_offset), new)
        if not data_offset - align < len(new) <= data_offset:
            raise GGUFError(f"renamed header would move the tensor data ({len(new)} bytes, data at {data_offset})")
        f.seek(0)
        f.write(new + b"\0" * (data_offset - len(new)))
    return f"renamed pooling_type to {target.decode()}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    try:
        print(fix_pooling_key(argv[0]))
    except (OSError, GGUFError, struct.error) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
