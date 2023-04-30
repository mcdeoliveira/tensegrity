import numpy as np
import numpy.typing as npt


def rotation_2d(phi: float) -> npt.NDArray:
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
