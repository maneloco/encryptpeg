from typing import List
import numpy as np

def create_dct_matrix() -> np.array():
    T = np.zeros((8,8), dtype=np.float64)
    for u in range(8):
        c = 1.0 / np.sqrt(8.0) if u == 0 else 0.5
        for x in range(8):
            T[u, x] = c * np.cos(((2.0 * x + 1.0) * u * np.pi) / 16.0)
    return T

_T = create_dct_matrix()
_T_T = _T.T

def to_frequencies(grid: List[List[float]]) -> List[List[float]]:
    """Calcula DCT 2D para el bloque 8x8 mediante el producto matricial?"""
    block = np.asarray(grid, dtype=np.float64) - 128.0
    freqs = T @ block @ T_T
    return freqs.tolist()

def from_frequencies(grid: List[List[float]]) -> List[List[float]]:
    """Calcula la inversa de la DCT 2D para el bloque 8x8 con producto matricial"""
    freq_block = np.asarray(freqs, dtype=np.float64)
    pixels = (T_T @ freq_block @ T) + 128.0
    pixels = np.clip(pixels, 0.0, 255.0)
    return pixels.tolist()
