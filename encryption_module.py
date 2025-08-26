"""
encryption_module.py
Student-prototype encryption layer for NoirLink (PPEC).

Provides:
- MockHEManager: Simulated Homomorphic Encryption for integer vectors (add/sub/mul/squared_diff).
- MockOPEManager: Order-preserving "encryption" (affine transform with positive scale).

Keys are saved under models/ to keep runs reproducible.

USAGE:
from encryption_module import MockHEManager, MockOPEManager

he = MockHEManager()
ct_a = he.encrypt_vec([5, 3, 1])
ct_b = he.encrypt_vec([2, 4, 7])
ct_diff2 = he.squared_diff(ct_a, ct_b)
print("Dec:", he.decrypt_vec(ct_diff2))  # [9, 1, 36]

ope = MockOPEManager()
ea = ope.encrypt(42); eb = ope.encrypt(100)
print("Order preserved:", ea < eb)       # True
print("Decrypt back:", ope.decrypt(ea))  # 42
"""

import os
import json
import secrets
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ------------------------------
# Utilities: simple persistent key store
# ------------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_json(path: str, obj: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def _load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------
# Mock OPE (Order Preserving "Encryption")
#   y = a * x + b, with a > 0  (so order is preserved strictly)
#   a, b are secret; integers.
#   This is NOT secure crypto; it's for classroom/demo use only.
# ------------------------------
class MockOPEManager:
    def __init__(self, key_dir: str = "models/ope_keys", key_file: str = "ope_key.json"):
        self.key_path = os.path.join(key_dir, key_file)
        self._load_or_create_key()

    def _load_or_create_key(self):
        data = _load_json(self.key_path)
        if data is None:
            # Choose positive scale a and offset b.
            # Keep a reasonably large to obscure small differences, but positive to preserve order.
            a = secrets.randbelow(10_000) + 1   # in [1..10000]
            b = secrets.randbelow(1_000_000)    # in [0..1e6)
            data = {"a": a, "b": b}
            _save_json(self.key_path, data)
        self.a = int(data["a"])
        self.b = int(data["b"])

    def encrypt(self, x: int) -> int:
        return int(self.a * int(x) + self.b)

    def decrypt(self, y: int) -> int:
        # Integer inverse (assumes exact transform); rounding just in case.
        return int(round((int(y) - self.b) / self.a))

    # Order-preserving comparisons can be done directly on ciphertexts:
    def less_than(self, enc_x: int, enc_y: int) -> bool:
        return enc_x < enc_y

    def encrypt_list(self, xs: List[int]) -> List[int]:
        return [self.encrypt(x) for x in xs]

    def decrypt_list(self, ys: List[int]) -> List[int]:
        return [self.decrypt(y) for y in ys]


# ------------------------------
# Mock HE (Homomorphic-like operations)
#   We wrap integer vectors in a Ciphertext object. Operations are done on the wrapped data.
#   This mimics the programming model of HE without requiring native libraries.
#   DO NOT use for security; it's meant to let the algorithm pipeline run end-to-end.
# ------------------------------
@dataclass
class _Ciphertext:
    # Stores an integer vector. In real HE this would be an opaque ciphertext blob.
    data: List[int]
    # "noise" or "nonce" simulated for realism (unused but kept to resemble HE objects)
    nonce: int = 0

class MockHEManager:
    def __init__(self, key_dir: str = "models/he_keys", key_file: str = "he_key.json"):
        self.key_path = os.path.join(key_dir, key_file)
        self._load_or_create_key()

    def _load_or_create_key(self):
        data = _load_json(self.key_path)
        if data is None:
            # In real HE you'd store public/secret keys. Here we keep a dummy secret.
            secret = secrets.token_hex(16)
            data = {"secret": secret}
            _save_json(self.key_path, data)
        self.secret = data["secret"]

    # -------- basic API --------
    def encrypt_vec(self, ints: List[int]) -> _Ciphertext:
        # Copy to avoid aliasing and simulate "ciphertext"
        return _Ciphertext(data=[int(x) for x in ints], nonce=secrets.randbelow(1 << 30))

    def decrypt_vec(self, ct: _Ciphertext) -> List[int]:
        # In real HE, this uses the secret key. Here we just unwrap.
        return list(ct.data)

    # Element-wise add/sub/mul of two ciphertext vectors (length must match)
    def add(self, a: _Ciphertext, b: _Ciphertext) -> _Ciphertext:
        self._assert_same_len(a, b)
        return _Ciphertext([x + y for x, y in zip(a.data, b.data)], nonce=secrets.randbelow(1 << 30))

    def sub(self, a: _Ciphertext, b: _Ciphertext) -> _Ciphertext:
        self._assert_same_len(a, b)
        return _Ciphertext([x - y for x, y in zip(a.data, b.data)], nonce=secrets.randbelow(1 << 30))

    def mul(self, a: _Ciphertext, b: _Ciphertext) -> _Ciphertext:
        self._assert_same_len(a, b)
        return _Ciphertext([x * y for x, y in zip(a.data, b.data)], nonce=secrets.randbelow(1 << 30))

    # Add/multiply ciphertext with a plaintext vector (same length) or scalar
    def add_plain(self, a: _Ciphertext, p) -> _Ciphertext:
        if isinstance(p, list):
            if len(p) != len(a.data): raise ValueError("Length mismatch in add_plain.")
            out = [x + int(y) for x, y in zip(a.data, p)]
        else:
            out = [x + int(p) for x in a.data]
        return _Ciphertext(out, nonce=secrets.randbelow(1 << 30))

    def mul_plain(self, a: _Ciphertext, p) -> _Ciphertext:
        if isinstance(p, list):
            if len(p) != len(a.data): raise ValueError("Length mismatch in mul_plain.")
            out = [x * int(y) for x, y in zip(a.data, p)]
        else:
            out = [x * int(p) for x in a.data]
        return _Ciphertext(out, nonce=secrets.randbelow(1 << 30))

    # -------- helpers for distance-like ops --------
    def squared_diff(self, x: _Ciphertext, y: _Ciphertext) -> _Ciphertext:
        """Element-wise (x - y)^2."""
        self._assert_same_len(x, y)
        out = [(a - b) * (a - b) for a, b in zip(x.data, y.data)]
        return _Ciphertext(out, nonce=secrets.randbelow(1 << 30))

    def sum_cipher(self, a: _Ciphertext) -> int:
        """Return plaintext sum of elements of a ciphertext vector.
        In real HE you'd return a ciphertext reduced sum; here we keep it simple for the prototype."""
        return int(sum(a.data))

    def squared_euclidean(self, x: _Ciphertext, y: _Ciphertext) -> int:
        """Return plaintext scalar ∑(xi - yi)^2. (Prototype simplification)
        In real HE you'd keep it encrypted. For the demo, we expose a scalar distance."""
        self._assert_same_len(x, y)
        return int(sum((a - b) * (a - b) for a, b in zip(x.data, y.data)))

    # -------- internal --------
    @staticmethod
    def _assert_same_len(a: _Ciphertext, b: _Ciphertext):
        if len(a.data) != len(b.data):
            raise ValueError("Ciphertext vector length mismatch.")


# ------------------------------
# Quick smoke test when run directly
# ------------------------------
def _demo_quick():
    print("=== Mock HE demo ===")
    he = MockHEManager()
    ct1 = he.encrypt_vec([5, 3, 1])
    ct2 = he.encrypt_vec([2, 4, 7])

    ct_add = he.add(ct1, ct2)         # [7, 7, 8]
    ct_mul = he.mul(ct1, ct2)         # [10, 12, 7]
    ct_sd  = he.squared_diff(ct1, ct2)  # [(5-2)^2, (3-4)^2, (1-7)^2] = [9, 1, 36]
    print("add   :", he.decrypt_vec(ct_add))
    print("mul   :", he.decrypt_vec(ct_mul))
    print("sdiff :", he.decrypt_vec(ct_sd))
    print("dist² :", he.squared_euclidean(ct1, ct2))  # 46

    print("\n=== Mock OPE demo ===")
    ope = MockOPEManager()
    ea = ope.encrypt(42)
    eb = ope.encrypt(100)
    print("enc(42) =", ea, " enc(100) =", eb)
    print("order preserved:", ea < eb)
    print("dec(enc(42)) =", ope.decrypt(ea))

if __name__ == "__main__":
    _demo_quick()
