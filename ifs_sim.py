import hcipy
import matplotlib.pyplot as plt
import numpy as np
import configparser
import pandas as pd
import tqdm
from hcipy import *
import logging
from ifs_sim_tools import _get_rect_extent


class MyFraunhoferPropagator(AgnosticOpticalElement):
    '''A monochromatic perfect lens propagator.

        This implements the propagation of a wavefront through a perfect lens. The wavefront
        is assumed to be exactly in the front focal plane of the lens and is propagated to the
        back focal plane. The implementation follows [Goodman2005]_.

        .. [Goodman2005] Goodman, J.W., 2005 Introduction to Fourier optics. Roberts and Company Publishers.

        Parameters
        ----------
        input_grid : Grid
            The grid on which the incoming wavefront is defined.
        output_grid : Grid
            The grid on which the outgoing wavefront is to be evaluated.
        focal_length : scalar
            The focal length of the lens system.
    '''
    def __init__(self, input_grid, output_grid, focal_length=1, d=1):
        self._input_grid = input_grid
        self._output_grid = output_grid
        self._focal_length = focal_length
        self._distance = d

        AgnosticOpticalElement.__init__(self, grid_dependent=True, wavelength_dependent=True)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        focal_length = self.evaluate_parameter(self.focal_length, input_grid, output_grid, wavelength)

        instance_data.uv_grid = output_grid.scaled(2 * np.pi / (focal_length * wavelength))
        instance_data.fourier_transform = make_fourier_transform(input_grid, instance_data.uv_grid)

        #instance_data.norm_factor = 1 / (1j * focal_length * wavelength)
        instance_data.norm_factor = 1 / (1j * focal_length * wavelength) * np.exp(
            1j * (2 * np.pi / wavelength) / (2 * focal_length) * (output_grid.x**2 + output_grid.y**2) * (1-self._distance/focal_length)
        )

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, focal_length):
        self._focal_length = focal_length

        self.clear_cache()

    def get_input_grid(self, output_grid, wavelength):
        return self._input_grid

    def get_output_grid(self, input_grid, wavelength):
        return self._output_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        '''Propagate a wavefront forward through the lens.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        U_new = instance_data.fourier_transform.forward(wavefront.electric_field) * instance_data.norm_factor
        return Wavefront(Field(U_new, instance_data.output_grid), wavefront.wavelength, wavefront.input_stokes_vector)

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        '''Propagate a wavefront backward through the lens.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        U_new = instance_data.fourier_transform.backward(wavefront.electric_field) / instance_data.norm_factor
        return Wavefront(Field(U_new, instance_data.input_grid), wavefront.wavelength, wavefront.input_stokes_vector)


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

ELT_DIA = 39.14634

pupil_grid = hcipy.make_pupil_grid(dims=1024, diameter=ELT_DIA)
#pupil_grid = hcipy.make_pupil_grid(1024,100e-3)
#eltgrid = hcipy.make_circular_aperture(13.5e-3)(pupil_grid)

ELT_AP  = hcipy.make_elt_aperture()
eltgrid = ELT_AP(pupil_grid)

fpgrid  = hcipy.make_focal_grid(20, 30, f_number=850, reference_wavelength=800e-9)
print(fpgrid.x)


ELTFocusProp = hcipy.FraunhoferPropagator(pupil_grid, fpgrid, focal_length=850 * ELT_DIA)

wf = hcipy.Wavefront(eltgrid, wavelength=800e-9)
wf2 = ELTFocusProp.forward(wf)
#wf2 = hcipy.Wavefront(hcipy.make_rectangular_aperture(0.01)(fpgrid), wavelength=800e-9)


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
        self.mirror_file = cfg.get('mirror_file', '')
        self.mirror_file = pd.read_csv(self.mirror_file)
        self.slicer_mag = float(cfg.get('slicer_mag', 1.0))
        self.slicer_dims = (self.slice_dims[0], self.slice_dims[1] * self.no_slices)

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

    def _split_field(self, focal_plane: hcipy.Wavefront) -> list[hcipy.Wavefront]:
        """Split the input field using apodisation with rectangular apertures.
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
        #subregion_size = int(max(self.slice_dims) / focal_plane.grid.delta[0])
        extents = _get_rect_extent(self.slicer_dims, rotation=self.rotation)
        subregion_size = np.ceil(
            max([
                abs(extents[1]-extents[0]),
                abs(extents[3]-extents[2])
            ])
        )

        sub_images = []
        for p in tqdm.tqdm(
            range(slice_centres.shape[0]),
            colour='red',
            desc='Slicing Field',
            leave=False):

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

    def _propto_pupil_mirror(self, sub_images):

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
            total=self.no_slices):
            prop = MyFraunhoferPropagator(
                im.grid,
                output_grid,
                focal_length=abs(self.mirror_file.loc[ix,'OAP EFL'] * 1e-3),
                d = 0
            )
            imout = prop.forward(im)
            ims_out.append(imout)

        return ims_out

    def _apply_pupil_mirror(self, pupil_images):
        pupil_mirror = make_rectangular_aperture(
                        self.pupil_dims,
                        self.pupil_decenter
                    )
        pupil_mirror = make_rotated_aperture(
            pupil_mirror,
            angle=self.rotation
        )
        apod = Apodizer(pupil_mirror)

        for n, im in tqdm.tqdm(
            enumerate(pupil_images),
            total=self.no_slices,
            leave=False,
            colour='green',
            desc='Pupil apod'):

            apod_im = apod.forward(im)
            pupil_images[n] = apod_im 

        return pupil_images

    def _propto_exit_slit(self, micro_pupils: list[Field]) -> list[Field]:
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

        output_image_dims = np.asarray(self.slicer_dims) * self.slicer_mag
        
        output_grid = make_focal_grid(
            q=2,
            num_airy=int(output_image_dims.max()/(2*resolution)),
            spatial_resolution=resolution,
            reference_wavelength=800e-9
        )

        exit_slit_images = []

        for m, pupil in tqdm.tqdm(
            enumerate(micro_pupils),
            total=len(micro_pupils),
            colour='magenta',
            leave=False,
            desc='Making exit slit'):

            prop = MyFraunhoferPropagator(
                pupil.grid,
                output_grid,
                focal_length=abs(self.mirror_file.loc[m,'PUP EFL'] * 1e-3),
                d=0
            )

            exit_slit_images.append(prop.forward(pupil))

        return exit_slit_images

    def _create_slicer_mirror(self):
        return

    def forward(self, wavefront):
        return 
    def backward(self, wavefront):
        return

import matplotlib.pyplot as plt

plt.figure()
hcipy.imshow_field(wf2.amplitude, norm='log', grid_units=1e-3)
plt.colorbar()
plt.show()

#test_im = hcipy.make_pupil_grid(128, 0.25)


im = ImageSlicer('./slicer.cfg')
ims = im._split_field(wf2)
pups = im._propto_pupil_mirror(ims)

#print(ims[0].grid.shape, ims[0].grid.delta)

pups_apod = im._apply_pupil_mirror(pups)

exit_slits = im._propto_exit_slit(pups_apod)



print(ims[0].amplitude)

plt.figure()
hcipy.imshow_field(ims[0].power)
plt.colorbar()
plt.figure()
hcipy.imshow_field(pups_apod[0].power)
plt.show()
plt.figure()
hcipy.imshow_field(exit_slits[0].power)
plt.show()
