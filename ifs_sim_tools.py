import numpy as np

def _get_rect_extent(rect_dims: tuple, rotation: float = 0) -> np.ndarray:
    extents = np.array([
        [-1,-1,],
        [-1, 1],
        [1, -1],
        [1, 1] 
    ], dtype=np.float64)
    extents *= np.asarray(rect_dims, dtype=np.float64)/2
    theta = np.radians(rotation)
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
        ]
    )
    for k in range(extents.shape[0]):
        extents[k] = R @ extents[k]
    return [extents[:, 0].min(),
            extents[:, 0].max(),
            extents[:, 1].min(),
            extents[:, 1].max()]

if __name__ == '__main__':
    print(_get_rect_extent((13.5, 0.3), rotation=45))
