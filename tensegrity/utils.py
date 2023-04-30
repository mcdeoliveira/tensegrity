import numpy as np
import numpy.typing as npt
import scipy


def rotation_2d(phi: float) -> npt.NDArray:
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def rotation_3d(v: npt.NDArray) -> npt.NDArray:
    return scipy.spatial.transform.Rotation.from_rotvec(v).as_matrix()
    # # compute rotation matrix
    # theta = np.linalg.norm(v)
    # if theta > epsilon:
    #     v = v / theta
    #     x, y, z = v
    #     ct = np.cos(theta)
    #     st = np.sin(theta)
    #     return np.array([
    #         [ct + (1 - ct) * x ** 2, (1 - ct) * x * y - st * z, (1 - ct) * x * z + st * y],
    #         [(1 - ct) * y * x + st * z, ct + (1 - ct) * y ** 2, (1 - ct) * y * z - st * x],
    #         [(1 - ct) * z * x - st * y, (1 - ct) * z * y + st * x, ct + (1 - ct) * z ** 2]
    #     ])
    # else:
    #     return np.eye(3)
