import hashlib
import sys
from pathlib import Path


def md5_file(file_path: str, chunk_size: int = 8192) -> str:
    """Return the MD5 hash of a file."""
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()