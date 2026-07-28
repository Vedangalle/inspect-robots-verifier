"""Small NumPy-to-PNG encoder for inline judge evidence."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import numpy as np
import numpy.typing as npt

_COLOR_TYPE_BY_CHANNELS = {1: 0, 3: 2, 4: 6}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_png(image: npt.NDArray[Any]) -> bytes:
    """Encode a uint8-compatible HxW image as lossless PNG."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    height, width, channels = array.shape
    try:
        color_type = _COLOR_TYPE_BY_CHANNELS[channels]
    except KeyError:
        raise ValueError(f"unsupported PNG channel count: {channels}") from None
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def png_data_url(image: npt.NDArray[Any]) -> str:
    """Encode one image as an inline PNG data URL."""
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")
