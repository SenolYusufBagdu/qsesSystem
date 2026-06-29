"""
QSES — Algorithm Registry
To add a new algorithm: create algorithms/algorithm_d.py with a class
inheriting BaseAlgorithm, then add one line to REGISTRY below.
"""
from .algorithm_a import AlgorithmA
from .algorithm_b import AlgorithmB
from .algorithm_c import AlgorithmC

REGISTRY = {
    "AlgorithmA": AlgorithmA,
    "AlgorithmB": AlgorithmB,
    "AlgorithmC": AlgorithmC,
}

def get_algorithm(name: str) -> "BaseAlgorithm":
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(REGISTRY)}")
    return cls()

__all__ = ["REGISTRY", "get_algorithm", "AlgorithmA", "AlgorithmB", "AlgorithmC"]
