from functools import partialmethod
import hcipy
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class progbar(tqdm.tqdm):
    """A tqdm progress bar with overridable display defaults."""

    __init__ = partialmethod(
        tqdm.tqdm.__init__, leave=False, colour='blue', ascii=' ▊'
    )

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

def dispersion_func_grating_basic(wvl, cwvl, theta_i, g, m, fcam):
    """Paraxial model for grating dispersion, calculated assuming optical axis
    of camera system follows chief ray of central wavelength.

    Args:
        wvl (np.ndarray): wavelength(s) of light nm
        cwvl (float): central wavelength in nm
        theta_i (float): input angle to grating normal in radians
        g (float): grating constant in lines / mm
        m (int): spectral order
        fcam (float): camera focal length in m
    """
    g_m = g * 1e3
    theta_o = np.arcsin(m * wvl * g_m - np.sin(theta_i))
    theta_c = np.arcsin(m * cwvl * g_m - np.sin(theta_i))
    return - fcam * np.tan(theta_o - theta_c)

def pad_and_shift_field(field_: hcipy.Wavefront, pad_pix = 2, shift_pixel: tuple = (0,0)):
    
    efield = field_.electric_field.copy()
    efield.shape = field_.grid.shape
    # pad with pad_pix pixels around edges
    padded_efield = np.zeros(
        np.array(efield.shape)+int(2*pad_pix), dtype=efield.dtype)
    padded_efield[pad_pix:-pad_pix, pad_pix:-pad_pix] = efield
    grid = hcipy.make_uniform_grid(
        padded_efield.shape,
        extent=np.array(padded_efield.shape)*np.array(field_.grid.delta))
    padded_efield = hcipy.Field(padded_efield.flatten(), grid)

    # apply the Fourier shift
    fshift = hcipy.FourierShift(
        grid, shift_pixel * np.array(field_.grid.delta))
    shifted_efield = fshift.forward(padded_efield)

    shifted_efield.shape = grid.shape
    shifted_efield_view = shifted_efield[pad_pix:-pad_pix, pad_pix:-pad_pix]


    shifted_field = hcipy.Field(
        shifted_efield_view.flatten(),
        field_.grid)

    shifted_wf = hcipy.Wavefront(
        shifted_field, wavelength=field_.wavelength)
    
    return shifted_wf

if __name__ == '__main__':
    import hcipy
    import numpy as np

    grid = hcipy.make_pupil_grid(100, 1)
    ap = hcipy.make_circular_aperture(100)(grid)
    outgrid = hcipy.make_focal_grid(5, 10, f_number=100, reference_wavelength=500e-9)
    fp = hcipy.FraunhoferPropagator(ap.grid, outgrid, 100)
    ap_foc = fp.forward(hcipy.Wavefront(ap, wavelength=500e-9))

    new_foc = pad_and_shift_array(ap_foc, pad_pix=2, shift_pixel=(10,0))

    
    fig, axs = plt.subplots(2, 1)


    plt.figure()
    hcipy.imshow_field(ap_foc.intensity, ax=axs[0])
    hcipy.imshow_field(new_foc.intensity, ax=axs[1])
    plt.show()