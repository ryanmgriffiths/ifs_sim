import hcipy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import configparser
import pandas as pd
import tqdm
from hcipy import *
import logging
from ifs_sim.ifs_sim_tools import _get_rect_extent, progbar, dispersion_func_grating_basic, pad_and_shift_field
import torch
from scipy.optimize import fsolve

def plot_phase_amp(field: hcipy.Field):
    fig, axs = plt.subplots(1, 2)
    hcipy.imshow_field(
        field.amplitude, grid_units=1e3, ax=axs[0])
    hcipy.imshow_field(
        field.phase, grid_units=1e3, ax=axs[1])
    for a in axs:
        a.set_xlabel('x [mm]')
        a.set_ylabel('y [mm]')
    plt.show()

class GeneralFraunhoferPropagator(FraunhoferPropagator):
    """Generalised Fraunhofer propagator for field before a lens.
    """
    def __init__(self, *args, d=None, **kwargs):
        super().__init__(*args, **kwargs)
        if d is None:
            self._distance = self._focal_length
        elif not isinstance(d, (float, int)) or d<0:
            raise ValueError('d must be float/int and >0.')
        else:
            self._distance = d

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        focal_length = self.evaluate_parameter(self.focal_length, input_grid, output_grid, wavelength)
        
        instance_data.uv_grid = output_grid.scaled(2 * np.pi / (focal_length * wavelength))
        instance_data.fourier_transform = make_fourier_transform(input_grid, instance_data.uv_grid)

        #instance_data.norm_factor = 1 / (1j * focal_length * wavelength)
        instance_data.norm_factor = 1 / (1j * focal_length * wavelength) * np.exp(
            1j * (2 * np.pi / wavelength) / (2 * focal_length) * (output_grid.x**2 + output_grid.y**2) * (1-self._distance/focal_length)
        )

class WavefrontSC(hcipy.Wavefront):
    def subregion(
        self,
        origin: tuple[int, int],
        pix_w: int,
    ) -> hcipy.Wavefront:
        """Get a sub-region of the wavefront as a separate wavefront

        Args:
           origin (tuple): Integer positions of top-left corner in pixels.
           pix_w (int): Width of square region to cut from origin.

        Returns:
            hcipy.Wavefront: Returned sub-region as a wavefront
        """
        #print(np.array(self.electric_field.data).shape)
        efield = np.array(self.electric_field.data).reshape(self.grid.shape)
        #stokes = np.array(self.stokes_vector.data).reshape((4,)+self.grid.shape)
        origin = np.array(origin).astype(int)

        efield_sr = efield[origin[0]: origin[0] + pix_w,
                           origin[1]: origin[1] + pix_w]
        #stokes_sr = stokes[:, origin[0]+pix_w, origin[1]+pix_w]
        new_grid = hcipy.make_pupil_grid(
            efield_sr.shape, diameter=pix_w*self.grid.delta[0])

        efield_sr = efield_sr.flatten()
        #stokes_sr = stokes_sr.reshape((4, int(pix_w**2)))

        new_efield = hcipy.Field(efield_sr, grid=new_grid)
        #new_stokes = hcipy.Field(stokes_sr, grid=new_grid)

        return hcipy.Wavefront(
            electric_field=new_efield,
            wavelength=self.wavelength,
            #input_stokes_vector=new_stokes
        )

class Spectrograph(hcipy.OpticalElement):
    """Disperse image-slicer exit-slit wavefronts onto a detector.

    Detector settings come from CONFIG and grating settings from GRATING.
    Slicer geometry and mirror data are not required.
    ``magnification`` scales exit-slit images while conserving power, and
    ``interpolate_exit_slit`` resamples their complex fields to detector pixel
    spacing. ``forward`` performs this resampling before detector placement.
    """
    def __init__(self, config_file: str):
        super().__init__()
        parser = configparser.ConfigParser()
        parser.read(config_file)
        self.params_dict = parser['CONFIG']
        self.grating_dict = parser['GRATING']

    def magnification(
        self,
        exit_slit: list[hcipy.Wavefront],
        factor: float,
    ) -> list[hcipy.Wavefront]:
        """Scale exit-slit images about the origin without changing the inputs.

        ``factor`` must be positive: values above one enlarge the images and
        values below one shrink them. HCIPy's Magnifier scales the grid and
        electric-field amplitude to conserve power, retaining the sample count.
        Use ``interpolate_exit_slit`` afterwards to sample at detector spacing,
        or pass the magnified images directly to ``forward``.
        """
        if np.ndim(factor) != 0 or not np.isfinite(factor) or factor <= 0:
            raise ValueError('Magnification factor must be finite and positive.')
        magnifier = hcipy.Magnifier(float(factor))
        return [magnifier.forward(wavefront) for wavefront in exit_slit]

    def interpolate_exit_slit(
        self,
        exit_slit: list[hcipy.Wavefront],
    ) -> list[hcipy.Wavefront]:
        """Return copies of exit-slit wavefronts sampled at detector pixel size.

        Regular 2D Cartesian input grids are resampled with HCIPy's linear
        interpolator on the complex electric field, preserving wavelength and
        polarization. Output grids retain each input grid's centre and cover
        its pixel extent, rounded up to whole detector pixels. Samples outside
        the input grid are zero. Interpolation does not enforce total power
        conservation, particularly when reducing the sampling resolution.
        """
        pixel_width = float(self.params_dict['pixel_width_um']) * 1e-6
        if not np.isfinite(pixel_width) or pixel_width <= 0:
            raise ValueError('Detector pixel size must be finite and positive.')

        resampled = []
        for wavefront in exit_slit:
            grid = wavefront.grid
            if not (isinstance(grid, hcipy.CartesianGrid) and grid.is_regular and grid.ndim == 2):
                raise ValueError('Exit-slit images require regular 2D Cartesian grids.')

            dims = np.asarray(grid.shape[::-1])
            same_sampling = np.allclose(grid.delta, pixel_width, rtol=1e-12, atol=0)
            if not same_sampling:
                pixel_extent = dims * grid.delta / pixel_width
                nearest = np.rint(pixel_extent)
                pixel_extent = np.where(
                    np.isclose(pixel_extent, nearest, rtol=1e-12, atol=0),
                    nearest, pixel_extent,
                )
                dims = np.maximum(1, np.ceil(pixel_extent)).astype(int)
            centre = np.array([(axis[0] + axis[-1]) / 2 for axis in grid.separated_coords])
            output_grid = hcipy.CartesianGrid(hcipy.RegularCoords(
                np.full(2, pixel_width), dims, centre - (dims - 1) * pixel_width / 2,
            ))

            result = wavefront.copy()
            if same_sampling:
                # Avoid interpolation artifacts for spacing differences due to roundoff.
                result.electric_field.grid = output_grid
            else:
                electric_field = wavefront.electric_field
                components = np.asarray(electric_field).reshape(-1, grid.size)
                interpolated = [
                    hcipy.make_linear_interpolator(
                        hcipy.Field(component, grid), fill_value=0,
                    )(output_grid)
                    for component in components
                ]
                values = np.asarray(interpolated).reshape(
                    electric_field.shape[:-1] + (output_grid.size,),
                )
                result.electric_field = hcipy.Field(values, output_grid)
            resampled.append(result)
        return resampled

    def _create_dispersed_image(self, exit_slit: list[hcipy.Wavefront], y_rays: np.ndarray):

        n_pixels = int(self.params_dict['detector_pixels'])
        pixel_width = float(self.params_dict['pixel_width_um'])*1e-6

        detector_pupil = make_pupil_grid(n_pixels, n_pixels * pixel_width)
        detector_int = np.zeros(detector_pupil.shape, dtype=float)

        d_to_centre = detector_pupil.x.min()

        disp_func = self._get_dispersion_function_wrapper()

        for n, im in enumerate(self.interpolate_exit_slit(exit_slit)):
            wvl = im.wavelength
            r0 = np.array([disp_func(wvl/1e-9), y_rays[n]]) # position of slit image
            r0_corner = r0 + np.array([im.grid.x.min(), im.grid.y.min()]) # position of the corner of the slit image
            abs_pos = r0_corner - d_to_centre # position of corner of slit image from corner of detector image
            abs_pos_pix = abs_pos / pixel_width
            rounded_pos = np.round(abs_pos_pix)
            height, width = im.grid.shape
            # Reject off-detector fields before distant positions can overflow integers.
            if (np.any(rounded_pos >= n_pixels)
                    or np.any(rounded_pos + np.array([width, height]) <= 0)):
                continue
            abs_pos_pix_int = rounded_pos.astype(int)
            subpix_shift = abs_pos_pix - abs_pos_pix_int

            col_start, row_start = abs_pos_pix_int
            row_min, row_max = max(0, row_start), min(n_pixels, row_start + height)
            col_min, col_max = max(0, col_start), min(n_pixels, col_start + width)

            # pad efield and apply sub pixel shift
            shifted_wf = pad_and_shift_field(
                im, pad_pix=2, shift_pixel=subpix_shift)

            # now add the field to the detector
            subfield_int = shifted_wf.intensity.reshape(im.grid.shape)

            detector_int[row_min:row_max, col_min:col_max] += subfield_int[
                row_min - row_start:row_max - row_start,
                col_min - col_start:col_max - col_start,
            ]

        return detector_int

    def _get_dispersion_function_wrapper(self):
        def disp_func_wrapper(wvl_nm: float):
            return dispersion_func_grating_basic(
                wvl_nm*1e-9,
                cwvl=float(self.grating_dict['cwvl_nm'])*1e-9,
                theta_i=np.deg2rad(float(self.grating_dict['thetai_deg'])),
                m = int(self.grating_dict['m']),
                g = float(self.grating_dict['g_lpmm']),
                fcam= float(self.grating_dict['fcam_mm'])*1e-3,
            )
        return disp_func_wrapper

    def get_useable_wavelengths(
        self,
        dispersion_func,
        detector_width,
        slit_img_width,
        cwvl
    ) -> tuple:
        bounds = (-detector_width/2 - slit_img_width/2,
                  detector_width/2 + slit_img_width/2)

        disp_lower = lambda x: dispersion_func(x) - bounds[0]
        disp_upper = lambda x: dispersion_func(x) - bounds[1]

        fsolve_lower = fsolve(disp_lower, x0 = cwvl)
        fsolve_upper = fsolve(disp_upper, x0 = cwvl)

        return fsolve_lower, fsolve_upper

    def forward(
        self,
        exit_slit: list[hcipy.Wavefront],
        y_rays: np.ndarray,
    ) -> np.ndarray:
        """Return detector intensity for exit-slit wavefronts.

        Exit-slit wavefronts are interpolated to the detector pixel spacing.
        ``y_rays`` gives the vertical detector position of each wavefront in
        metres; horizontal positions are calculated from their wavelengths
        using the grating. Magnification of the input images can be applied
        separately with ``magnification``; ray positions are not scaled.
        """
        if len(exit_slit) != len(y_rays):
            raise ValueError('Provide one y position per exit-slit wavefront.')
        return self._create_dispersed_image(exit_slit, y_rays)

    def backward(self, wavefront):
        raise NotImplementedError('Spectrograph back-propagation not supported')

class ImageSlicer(hcipy.OpticalElement):
    def __init__(self, config_file: str):
        """Init function."""
        super().__init__()
        self._read_in_cfg(config_file)

    def _read_in_cfg(self, config_file):
        parser = configparser.ConfigParser()
        parser.read(config_file)

        cfg = parser['CONFIG']
        self.no_slices = int(cfg['spaxels'])
        self.slice_dims = tuple(float(x.strip()) for x in cfg['slice_dims'].split(','))
        self.pupil_dims = tuple(float(x.strip()) for x in cfg['pupil_dims'].split(','))
        self.transmission = float(cfg['transmission'])
        self.slicer_decenter = tuple(float(x.strip()) for x in cfg['slicer_decenter'].split(','))
        self.pupil_decenter = tuple(float(x.strip()) for x in cfg['pupil_decenter'].split(','))
        self.rotation = np.radians(float(cfg['stack_rotation']))
        self.pupil_rotation = np.radians(float(cfg['pupil_rotation']))
        self.mirror_file = cfg.get('mirror_file', '')
        self.mirror_file = pd.read_csv(self.mirror_file)
        self.slicer_mag = float(cfg.get('slicer_mag', 1.0))
        self.slicer_dims = (self.slice_dims[0], self.slice_dims[1] * self.no_slices)
        self.params_dict = cfg

    def _get_slice_centres(self):
        slice_centres = np.zeros((self.no_slices, 2))
        slice_centres[:,1] = np.arange(self.no_slices) * self.slice_dims[1]
        slice_centres[:,1] -= (self.no_slices // 2) * self.slice_dims[1]
        if self.no_slices%2 == 0:
            slice_centres[:,1] += 0.5 * self.slice_dims[1]
        return slice_centres + np.array(self.slicer_decenter)

    def _get_pupil_centres(self):
        pupil_centres = np.zeros((self.no_slices, 2), dtype=np.float32)
        pupil_centres[:,1] = np.arange(self.no_slices) * self.pupil_dims[1]
        pupil_centres[:,1] -= (self.no_slices // 2) * self.pupil_dims[1]
        if self.no_slices %2 == 0:
            pupil_centres[:,1] += 0.5 * self.pupil_dims[1]
        return pupil_centres + np.array(self.pupil_decenter)

    def _split_field(self, focal_plane: hcipy.Wavefront, preview: bool = True) -> list[hcipy.Wavefront]:
        """Split the input field using apodisation with rectangular apertures.

        If preview is enabled, display the input intensity with slice outlines.
        """
        slice_centres = self._get_slice_centres()


        slice_centres_pix = np.nan_to_num(
            np.round(slice_centres / focal_plane.grid.delta[0]).astype(int))

        R = np.array([
            [np.cos(self.rotation), -np.sin(self.rotation)],
            [np.sin(self.rotation), np.cos(self.rotation)]])

        slice_centres_pix = np.einsum('ij,pj->pi', R, slice_centres_pix)
        slice_centres_pix += np.array(focal_plane.grid.shape) // 2
        slice_centres_pix += np.array(self.slicer_decenter)
        subregion_size = int(max(self.slice_dims) / focal_plane.grid.delta[0])

        if preview:
            fig, ax = plt.subplots()
            field_image = hcipy.imshow_field(focal_plane.intensity, ax=ax)
            fig.colorbar(field_image, ax=ax, label='Intensity')
            # Rotate both the rectangles and their centres about the origin,
            # matching make_rotated_aperture below.
            mask_transform = Affine2D().rotate(self.rotation) + ax.transData
            for centre in slice_centres:
                ax.add_patch(Rectangle(
                    centre - np.asarray(self.slice_dims) / 2,
                    width=self.slice_dims[0],
                    height=self.slice_dims[1],
                    fill=False,
                    edgecolor='red',
                    transform=mask_transform,
                ))
            ax.set(xlabel='x (m)', ylabel='y (m)', title='Input field and slice masks')
            plt.show()

        # sanity check

        #extents = _get_rect_extent(self.slicer_dims, rotation=self.rotation)
        #subregion_size = np.ceil(
        #    max([
        #        abs(extents[1]-extents[0]),
        #        abs(extents[3]-extents[2])
        #    ])
        #)

        sub_images = []
        for p in tqdm.tqdm(
            range(slice_centres.shape[0]),
            colour='red',
            desc='Slicing Field',
            leave=False, ascii=' ▊'):

            slice_mask = hcipy.make_rectangular_aperture(
                size = self.slice_dims,
                center = slice_centres[p])
            slice_mask = hcipy.make_rotated_aperture(
                slice_mask, angle=self.rotation)
            apodizer = hcipy.Apodizer(slice_mask)
            wf = apodizer.forward(focal_plane)

            wf_temp = WavefrontSC(wf.electric_field, wf.wavelength)
            wf_subregion = wf_temp.subregion(
                origin=(slice_centres_pix[p,1] - subregion_size//2, 
                        slice_centres_pix[p,0] - subregion_size//2),
                pix_w= subregion_size
            )
            sub_images.append(wf_subregion)
        return sub_images

    def _propto_pupil_mirror(self, sub_images: list):

        #print(sub_images[0].grid.size, sub_images[0].grid.delta)

        fmin = np.min(np.abs(self.mirror_file.loc[:,'OAP EFL'] * 1e-3))
        input_grid_D = sub_images[0].grid.delta[0] * sub_images[0].grid.shape[0]
        resolution = fmin/input_grid_D * 800e-9

        output_grid = hcipy.make_focal_grid(
            4,
            num_airy=abs(6e-3)/(2*resolution),
            spatial_resolution=resolution
        )
    
        ims_out = []

        for ix, im in tqdm.tqdm(
            enumerate(sub_images),
            desc='Pupil Mirror',
            colour='blue',
            leave=False,
            total=self.no_slices,
            ascii=' ▊'):
            prop = GeneralFraunhoferPropagator(
                im.grid,
                output_grid,
                focal_length=abs(self.mirror_file.loc[ix,'OAP EFL'] * 1e-3),
                d = 0
            )
            imout = prop.forward(im)
            ims_out.append(imout)

        return ims_out

    def _apply_pupil_mirror(self, pupil_images: list, plot: bool = True):
        pupil_mirror = make_rectangular_aperture(
                        self.pupil_dims,
                        self.pupil_decenter
                    )
        pupil_mirror = make_rotated_aperture(
            pupil_mirror,
            angle=self.pupil_rotation
        )
        apod = Apodizer(pupil_mirror)

        for n, im in tqdm.tqdm(
            enumerate(pupil_images),
            total=self.no_slices,
            leave=False,
            colour='green',
            desc='Pupil apod',
            ascii=' ▊'):

            apod_im = apod.forward(im)
            pupil_images[n] = apod_im

        if plot:
            plot_phase_amp(pupil_images[-1])

        return pupil_images

    def _propto_exit_slit(self, micro_pupils: list[Field], plot=True) -> list[Field]:
        """Propagate to the exit slit of the spectrograph.

        Args:
            micro_pupils (list[hcipy.Field]): Micro-pupils.

        Returns:
            list[hcipy.Field]: Exit slit fields for all micro-pupils.\
        """
        # sample exit slit images with 2 pixels
        fmin = np.min(np.abs(self.mirror_file.loc[:,'PUP EFL'] * 1e-3))


        input_grid_D = micro_pupils[0].grid.shape[0] * micro_pupils[0].grid.delta[0]
        resolution = fmin/input_grid_D * 800e-9

        output_image_dims = np.asarray(self.slice_dims) * self.slicer_mag
        resolution = output_image_dims.min()
        
        """output_grid = make_focal_grid(
            q=2,
            num_airy=int(output_image_dims.max()/(2*resolution)),
            spatial_resolution=resolution,
            reference_wavelength=800e-9
        )
        """
        oversize = int(self.params_dict.get('exit_slit_oversize', 1))

        output_grid = make_focal_grid(
            q = 2,
            num_airy=oversize*int(output_image_dims.max()/(2*resolution)),
            spatial_resolution=output_image_dims.min()
        )

        exit_slit_images = []

        for m, pupil in tqdm.tqdm(
            enumerate(micro_pupils),
            total=len(micro_pupils),
            colour='magenta',
            leave=False,
            desc='Making exit slit',
            ascii=' ▊'):

            prop = GeneralFraunhoferPropagator(
                pupil.grid,
                output_grid,
                focal_length=abs(self.mirror_file.loc[m,'PUP EFL'] * 1e-3),
                d=0
            )

            exit_slit_images.append(prop.forward(pupil))

        if plot:
            plot_phase_amp(exit_slit_images[-1])

        return exit_slit_images

    def _create_exit_slit(self, slit_ims: list[Field]) -> np.ndarray:
        """Create the exit slit by vertically stacking slit images and leaving
        1 pixel spacing.
        """
        oversize = int(self.params_dict.get('exit_slit_oversize', 1))
        slit_im_dim_oversized = slit_ims[0].grid.shape[0]
        slit_im_dim = slit_im_dim_oversized//oversize

        exit_slit_master_array = np.zeros((
            slit_im_dim * self.no_slices + (self.no_slices - 1) + (slit_im_dim_oversized - slit_im_dim),
            slit_im_dim_oversized
        ), dtype=np.float64)

        pos_counter = 0
        for n, im in enumerate(slit_ims):
            exit_slit_master_array[
                pos_counter:pos_counter+slit_im_dim_oversized,
                :slit_im_dim_oversized] += im.intensity.reshape(
                    im.grid.shape
                    ).astype(np.float64)
            pos_counter += 1+slit_im_dim

        return exit_slit_master_array

    def _apply_micropupil_phase(
        self,
        micro_pupils: list[hcipy.Wavefront],
        phases: np.ndarray,
        sampling: np.ndarray
    ) -> list[hcipy.Wavefront]:
        """Resample OPD maps and add their phase to the wavefronts in place.

        ``phases[i]`` is a (y, x) optical path difference map in nm, with
        its geometric centre at (0, 0). ``sampling[i]`` is its pixel spacing
        in metres, either a scalar or an (dx, dy) pair. Outside the map's
        sampled extent, no phase is added. Return the updated input list.
        """
        if len(phases) != len(micro_pupils) or len(sampling) != len(micro_pupils):
            raise ValueError('Provide one phase map and sampling per micropupil.')

        new_pupils = []

        for i, pupil in progbar(
            enumerate(micro_pupils),
            total=len(micro_pupils),
            colour='red',
            desc='Adding phase maps'
        ):
            phase_map = np.nan_to_num(np.asarray(phases[i], dtype=float))
            spacing = np.broadcast_to(np.asarray(sampling[i], dtype=float), (2,)) * 1e-3
            if phase_map.ndim != 2:
                raise ValueError(f'Phase map {i} must be a 2D array.')
            if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
                raise ValueError(f'Sampling {i} must be finite and positive.')

            ny, nx = phase_map.shape
            phase_grid = hcipy.make_pupil_grid(
                (nx, ny), diameter=spacing * (nx, ny)
            )
            phase_field = hcipy.Field(phase_map.ravel(), phase_grid)
            interpolate = hcipy.make_linear_interpolator(
                phase_field, fill_value=0)
            opd_wv = interpolate(pupil.grid)
            
            apod = PhaseApodizer(opd_wv)            
            new_pupils.append(apod.forward(pupil))
        
        return new_pupils

    def forward(self, wavefront: hcipy.Wavefront) -> list[hcipy.Wavefront]:
        """Propagate the wavefront through a Fraunhofer diffraction model of the
        image slicer stack.

        Args:
            wavefront (hcipy.Wavefront): Wavefront in the slicer stack plane.

        Returns:
            list[hcipy.Wavefront]: Exit slit images for each slicer mirror.
        """
        slicer_stack_wf = self._split_field(wavefront, preview=False)
        pupil_mirror_wf = self._propto_pupil_mirror(slicer_stack_wf)
        pupil_mirror_wf = self._apply_pupil_mirror(pupil_mirror_wf, plot=False)
        exit_slit_wf = self._propto_exit_slit(pupil_mirror_wf, plot=False)
        return exit_slit_wf

    def backward(self, wavefront):
        return NotImplementedError("Image slicer back-propagation not supported")
