import pytest
import configparser
from ifs_sim import ifs_sim
import hcipy
import numpy as np
import matplotlib.pyplot as plt
import aotools
import numpy as np

@pytest.fixture
def SpectrographSim():
    return ifs_sim.Spectrograph('./slicer.cfg')

@pytest.mark.parametrize('position', [(5.25e-3, 0), (-10.0e-3, 4.0e-3), (-12.0e-3, -8.0e-3)])
def test_create_dispersed_image(SpectrographSim: ifs_sim.Spectrograph, mocker, position):
    """Fills the detector image with a test image.
    """
    pixel_width = float(SpectrographSim.params_dict['pixel_width_um']) * 1e-6
    diameter = 100 * pixel_width
    grid = hcipy.make_pupil_grid(100, diameter=diameter)
    # Match detector spacing exactly despite roundoff in diameter / 100.
    grid.delta[:] = pixel_width
    aperture = hcipy.make_circular_aperture(diameter)(grid)
    wavefront = hcipy.Wavefront(aperture, wavelength=800e-9)

    mocker.patch.object(
        SpectrographSim,
        '_get_dispersion_function_wrapper',
        return_value=mocker.Mock(return_value=position[0]),
    )
    detector_img = SpectrographSim.forward([wavefront], y_rays=np.array([position[1]]))

    centroid = aotools.centre_of_gravity(detector_img) * pixel_width
    # distance from the top left corner edge
    centroid += pixel_width/2
    # to centre its half the array width
    arr_centre = pixel_width*detector_img.shape[0]/2
    dist = centroid - arr_centre 

    np.testing.assert_allclose(dist, np.asarray(position), rtol=1e-3)


@pytest.mark.parametrize('position', [
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1e20, 0.0), (-1e20, 0.0), (0.0, 1e20), (0.0, -1e20),
])
@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_create_dispersed_image_no_overlap(SpectrographSim, mocker, position):
    pixel_width = float(SpectrographSim.params_dict['pixel_width_um']) * 1e-6
    grid = hcipy.make_pupil_grid(100, diameter=100 * pixel_width)
    grid.delta[:] = pixel_width
    aperture = hcipy.make_circular_aperture(100 * pixel_width)(grid)
    wavefront = hcipy.Wavefront(aperture, wavelength=800e-9)
    mocker.patch.object(
        SpectrographSim,
        '_get_dispersion_function_wrapper',
        return_value=mocker.Mock(return_value=position[0]),
    )
    shift = mocker.patch.object(ifs_sim, 'pad_and_shift_field')

    detector_img = SpectrographSim.forward(
        [wavefront], y_rays=np.array([position[1]]),
    )

    assert not np.any(detector_img)
    shift.assert_not_called()


def test_spectrograph_without_slicer_configuration(tmp_path):
    config = tmp_path / 'spectrograph.cfg'
    config.write_text(
        '[CONFIG]\n'
        'detector_pixels = 64\n'
        'pixel_width_um = 13.5\n'
        '[GRATING]\n'
        'g_lpmm = 850\n'
        'm = 1\n'
        'cwvl_nm = 801\n'
        'thetai_deg = 35\n'
        'fcam_mm = 117.6\n'
    )
    spectrograph = ifs_sim.Spectrograph(str(config))
    pixel_width = 13.5 * 1e-6
    grid = hcipy.make_pupil_grid(8, diameter=8 * pixel_width)
    aperture = hcipy.make_circular_aperture(8 * pixel_width)(grid)
    wavefront = hcipy.Wavefront(aperture, wavelength=801e-9)

    detector = spectrograph.forward([wavefront], np.array([0.0]))

    assert detector.shape == (64, 64)
    np.testing.assert_allclose(detector[28:36, 28:36], wavefront.intensity.shaped,
                               atol=1e-12)
    np.testing.assert_allclose(detector.sum(), wavefront.intensity.sum())

    dispersion = spectrograph._get_dispersion_function_wrapper()
    lower, upper = spectrograph.get_useable_wavelengths(
        dispersion, 64 * pixel_width, 8 * pixel_width, 801,
    )
    np.testing.assert_allclose(dispersion(lower), -36 * pixel_width)
    np.testing.assert_allclose(dispersion(upper), 36 * pixel_width)


def test_slicer_without_spectrograph_configuration(tmp_path):
    parser = configparser.ConfigParser()
    parser.read('slicer.cfg')
    parser.remove_section('GRATING')
    parser.remove_option('CONFIG', 'detector_pixels')
    parser.remove_option('CONFIG', 'pixel_width_um')
    config = tmp_path / 'slicer.cfg'
    with config.open('w') as stream:
        parser.write(stream)

    slicer = ifs_sim.ImageSlicer(str(config))

    assert slicer._get_slice_centres().shape == (slicer.no_slices, 2)


def test_spectrograph_requires_one_y_position_per_wavefront(SpectrographSim):
    with pytest.raises(ValueError, match='one y position'):
        SpectrographSim.forward([], np.array([0.0]))


@pytest.mark.parametrize('factor', [0.5, 2.0])
def test_spectrograph_magnification(SpectrographSim, factor):
    grid = hcipy.make_pupil_grid((12, 8), diameter=(120e-6, 80e-6))
    wavefront = hcipy.Wavefront(grid.ones() * (1 + 2j), wavelength=801e-9)
    original = wavefront.copy()

    magnified, = SpectrographSim.magnification([wavefront], factor)

    np.testing.assert_allclose(magnified.grid.points, original.grid.points * factor)
    np.testing.assert_allclose(magnified.electric_field, original.electric_field / factor)
    np.testing.assert_allclose(magnified.total_power, original.total_power)
    np.testing.assert_array_equal(wavefront.grid.points, original.grid.points)
    np.testing.assert_array_equal(wavefront.electric_field, original.electric_field)
    assert magnified.wavelength == original.wavelength


@pytest.mark.parametrize('factor', [0, -1, np.inf, np.nan])
def test_spectrograph_invalid_magnification(SpectrographSim, factor):
    with pytest.raises(ValueError, match='finite and positive'):
        SpectrographSim.magnification([], factor)


@pytest.mark.parametrize('sampling_ratio', [0.5, 2.0, 1.3])
@pytest.mark.parametrize('polarized', [False, True])
def test_interpolate_exit_slit_complex_field(SpectrographSim, sampling_ratio, polarized):
    pixel_width = float(SpectrographSim.params_dict['pixel_width_um']) * 1e-6
    dims = np.array([12, 8])
    centre = np.array([3 * pixel_width, -2 * pixel_width])
    grid = hcipy.make_uniform_grid(
        dims, dims * sampling_ratio * pixel_width, center=centre,
    )

    def field_at(grid):
        return 1 + grid.x / (20 * pixel_width) + 1j * (0.5 + grid.y / (16 * pixel_width))

    stokes = [1, 0, 0, 0] if polarized else None
    wavefront = hcipy.Wavefront(field_at(grid), wavelength=801e-9,
                               input_stokes_vector=stokes)
    original = wavefront.copy()

    resampled, = SpectrographSim.interpolate_exit_slit([wavefront])

    np.testing.assert_array_equal(resampled.grid.delta, [pixel_width, pixel_width])
    np.testing.assert_array_equal(resampled.grid.shape, np.ceil(dims[::-1] * sampling_ratio))
    np.testing.assert_allclose(
        [resampled.grid.x.mean(), resampled.grid.y.mean()], centre,
    )
    expected = np.asarray(field_at(resampled.grid))
    outside = ((resampled.grid.x < grid.x.min()) | (resampled.grid.x > grid.x.max())
               | (resampled.grid.y < grid.y.min()) | (resampled.grid.y > grid.y.max()))
    expected[outside] = 0
    if polarized:
        expected = np.eye(2)[..., np.newaxis] * expected
    np.testing.assert_allclose(resampled.electric_field, expected, atol=1e-14)
    np.testing.assert_array_equal(resampled.input_stokes_vector, wavefront.input_stokes_vector)
    assert resampled.wavelength == wavefront.wavelength
    np.testing.assert_array_equal(wavefront.electric_field, original.electric_field)
    np.testing.assert_array_equal(wavefront.grid.points, original.grid.points)


def test_interpolate_exit_slit_matching_sampling(SpectrographSim, mocker):
    pixel_width = float(SpectrographSim.params_dict['pixel_width_um']) * 1e-6
    grid = hcipy.make_pupil_grid(100, diameter=100 * pixel_width)
    wavefront = hcipy.Wavefront(grid.ones() * (1 + 1j), wavelength=801e-9)
    interpolate = mocker.patch.object(hcipy, 'make_linear_interpolator')

    resampled, = SpectrographSim.interpolate_exit_slit([wavefront])

    np.testing.assert_array_equal(resampled.electric_field, wavefront.electric_field)
    np.testing.assert_array_equal(resampled.grid.delta, [pixel_width, pixel_width])
    assert resampled is not wavefront
    interpolate.assert_not_called()


def test_spectrograph_forward_resamples_magnified_exit_slit(SpectrographSim):
    SpectrographSim.params_dict['detector_pixels'] = '64'
    pixel_width = float(SpectrographSim.params_dict['pixel_width_um']) * 1e-6
    grid = hcipy.make_pupil_grid(8, diameter=8 * pixel_width)
    wavefront = hcipy.Wavefront(grid.ones(), wavelength=801e-9)
    magnified = SpectrographSim.magnification([wavefront], 2)

    detector = SpectrographSim.forward(magnified, np.array([0.0]))

    expected = np.zeros((64, 64))
    # Linear interpolation is zero beyond the outermost input sample centres.
    expected[25:39, 25:39] = 0.25
    np.testing.assert_allclose(detector, expected, atol=1e-12)

    
