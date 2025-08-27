"""
encryption_module.py

Dual-mode encryption module for NoirLink:
- Mock mode: fast simulated encryption (for demos / if building Pyfhel is not possible)
- Real mode: uses Pyfhel (BFV-like integer HE) for real homomorphic encryption

Switch mode by setting environment variable NOIRLINK_HE:
- NOIRLINK_HE=real    -> use Pyfhel
- NOIRLINK_HE=mock    -> use lightweight mock (default)

API classes:
- HEManager (abstracted): encrypt_vec, decrypt_vec, add, sub, mul, squared_diff_ct, squared_euclidean_ct, decrypt_scalar
- OPEManager (affine order-preserving substitution): encrypt, decrypt, encrypt_list, decrypt_list, less_than
"""

import os
import json
import secrets
from typing import List
import numpy as np

USE_REAL = os.environ.get("NOIRLINK_HE", "mock").lower() == "real"

# --------- OPE (affine transform) - same as before (simple, not cryptographically strong) -----
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _save_json(path: str, obj):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def _load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

class OPEManager:
    def __init__(self, key_dir: str = "models/ope_keys", key_file: str = "ope_key.json"):
        self.key_path = os.path.join(key_dir, key_file)
        self._load_or_create_key()

    def _load_or_create_key(self):
        data = _load_json(self.key_path)
        if data is None:
            a = secrets.randbelow(10_000) + 1
            b = secrets.randbelow(1_000_000)
            data = {"a": a, "b": b}
            _save_json(self.key_path, data)
        self.a = int(data["a"])
        self.b = int(data["b"])

    def encrypt(self, x: int) -> int:
        return int(self.a * int(x) + self.b)

    def decrypt(self, y: int) -> int:
        return int(round((int(y) - self.b) / self.a))

    def less_than(self, ca: int, cb: int) -> bool:
        return ca < cb

    def encrypt_list(self, xs: List[int]) -> List[int]:
        return [self.encrypt(x) for x in xs]

    def decrypt_list(self, ys: List[int]) -> List[int]:
        return [self.decrypt(y) for y in ys]


# ---------------- MOCK HE (if NOIRLINK_HE != "real") ----------------
if not USE_REAL:
    from dataclasses import dataclass

    @dataclass
    class _CT:
        data: List[int]
        nonce: int = 0

    class HEManager:
        def __init__(self, key_dir="models/he_keys", key_file="he_key.json"):
            self.key_path = os.path.join(key_dir, key_file)
            data = _load_json(self.key_path)
            if data is None:
                _save_json(self.key_path, {"secret": secrets.token_hex(16)})
            self.ope = OPEManager()

        def encrypt_vec(self, ints: List[int]) -> _CT:
            return _CT(data=[int(x) for x in ints], nonce=secrets.randbelow(1 << 30))

        def decrypt_vec(self, ct: _CT) -> List[int]:
            return list(ct.data)

        def add(self, a: _CT, b: _CT) -> _CT:
            return _CT([x + y for x, y in zip(a.data, b.data)])

        def sub(self, a: _CT, b: _CT) -> _CT:
            return _CT([x - y for x, y in zip(a.data, b.data)])

        def mul(self, a: _CT, b: _CT) -> _CT:
            return _CT([x * y for x, y in zip(a.data, b.data)])

        def squared_diff_ct(self, x: _CT, y: _CT) -> _CT:
            return _CT([(a - b) * (a - b) for a, b in zip(x.data, y.data)])

        def squared_euclidean_ct(self, x: _CT, y: _CT):
            val = int(sum((a - b) * (a - b) for a, b in zip(x.data, y.data)))
            return _CT([val])

        def decrypt_scalar(self, ct):
            if isinstance(ct, _CT):
                return int(ct.data[0])
            raise ValueError("Unsupported ct type for mock decrypt_scalar")
# ---------------- REAL HE using Pyfhel (if NOIRLINK_HE=real) ----------------
else:
    try:
        from Pyfhel import Pyfhel, PyCtxt
    except Exception as e:
        raise ImportError("Pyfhel is required for REAL HE mode. Install it after Visual C++ Build Tools + CMake. Error: " + str(e))

    class HEManager:
        """
        Real HE manager using Pyfhel (BFV-like integer operations).
        - encrypt_vec: encrypts each integer as a ciphertext (list of PyCtxt)
        - decrypt_vec: decrypts list of ciphertexts to ints
        - arithmetic ops: add, sub, mul perform ciphertext arithmetic
        - squared_euclidean_ct returns a ciphertext that encrypts the scalar sum; decrypt_scalar will decrypt it
        """
        def __init__(self, key_dir="models/he_keys", key_file="pyfhel_keys.json"):
            self.key_dir = key_dir
            os.makedirs(self.key_dir, exist_ok=True)
            self.key_file = os.path.join(self.key_dir, key_file)

            self.P = Pyfhel()
            p_params = {
                'scheme': 'BFV',
                'n': 4096,
                't': 65537,
                'scale': 2
            }
            self.P.contextGen(**p_params)
            self.P.keyGen()
            self.P.relinKeyGen()

            self.P.save_public_key(os.path.join(self.key_dir, "pyfhel_pub.key"))
            self.P.save_secret_key(os.path.join(self.key_dir, "pyfhel_sec.key"))

            self.ope = OPEManager()

        def encrypt_vec(self, ints: List[int]) -> List[PyCtxt]:
            out = []
            for x in ints:
                out.append(self.P.encrypt(np.array([x], dtype=np.int64)))
            return out

        def decrypt_vec(self, cts: List[PyCtxt]) -> List[int]:
            out = []
            for ct in cts:
                out.append(int(self.P.decryptInt(ct)[0]))
            return out

        def add(self, a: List[PyCtxt], b: List[PyCtxt]) -> List[PyCtxt]:
            return [self.P.add(cta, ctb) for cta, ctb in zip(a, b)]

        def sub(self, a: List[PyCtxt], b: List[PyCtxt]) -> List[PyCtxt]:
            return [self.P.sub(cta, ctb) for cta, ctb in zip(a, b)]

        def mul(self, a: List[PyCtxt], b: List[PyCtxt]) -> List[PyCtxt]:
            res = []
            for cta, ctb in zip(a, b):
                ct_mul = self.P.multiply(cta, ctb)
                self.P.relinearize(ct_mul)
                res.append(ct_mul)
            return res

        def squared_diff_ct(self, x: List[PyCtxt], y: List[PyCtxt]) -> List[PyCtxt]:
            out = []
            for cta, ctb in zip(x, y):
                diff = self.P.sub(cta, ctb)
                sq = self.P.multiply(diff, diff)
                self.P.relinearize(sq)
                out.append(sq)
            return out

        def squared_euclidean_ct(self, x: List[PyCtxt], y: List[PyCtxt]) -> PyCtxt:
            sqs = self.squared_diff_ct(x, y)
            acc = sqs[0]
            for ct in sqs[1:]:
                acc = self.P.add(acc, ct)
            return acc

        def decrypt_scalar(self, ct: PyCtxt) -> int:
            val = self.P.decryptInt(ct)
            return int(val[0])

# Expose a factory/helper to create managers
def get_managers(mode: str = None):
    """
    mode: "real" or "mock" or None (uses NOIRLINK_HE env)
    returns (he_manager_instance, ope_manager_instance)
    """
    if mode is None:
        mode = "real" if USE_REAL else "mock"
    
    he = HEManager()
    ope = OPEManager()
    return he, ope


# ------------------
# Quick smoke test when run directly
# ------------------
def _demo_quick():
    print(f"Current mode: {'REAL' if USE_REAL else 'MOCK'}")
    
    he, ope = get_managers()

    print("\n=== HE demo ===")
    v1 = [5, 3, 1]
    v2 = [2, 4, 7]
    ct1 = he.encrypt_vec(v1)
    ct2 = he.encrypt_vec(v2)
    print(f"Encrypting vector 1: {v1}")
    print(f"Encrypting vector 2: {v2}")

    ct_add = he.add(ct1, ct2)
    ct_mul = he.mul(ct1, ct2)
    ct_sd = he.squared_diff_ct(ct1, ct2)
    print("add   :", he.decrypt_vec(ct_add))
    print("mul   :", he.decrypt_vec(ct_mul))
    print("sdiff :", he.decrypt_vec(ct_sd))
    print("dist² :", he.decrypt_scalar(he.squared_euclidean_ct(ct1, ct2)))

    print("\n=== OPE demo ===")
    ea = ope.encrypt(42)
    eb = ope.encrypt(100)
    print("enc(42) =", ea, " enc(100) =", eb)
    print("order preserved:", ope.less_than(ea, eb))
    print("dec(enc(42)) =", ope.decrypt(ea))

if __name__ == "__main__":
    _demo_quick()