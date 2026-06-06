"""Compatibility shim — crypto engine moved to src/crypto/.

Import from src.crypto instead:
    from src.crypto import CryptoEngine
"""
from src.crypto.engine import CryptoEngine  # noqa: F401

__all__ = ["CryptoEngine"]
