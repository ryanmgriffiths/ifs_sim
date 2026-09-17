import hcipy
import ifs_sim, ifs_sim_tools
import matplotlib.pyplot as plt
from hcipy import *
import numpy as np

NUM_MODES = 50
PUPIL_DIAMETER = 3.35e-3  # m
OCCULTMASK_SIZE = 450e-6  # um
INPUT_F_NUMBER = 850
CWL = 801e-9

# DEFINE ELEMENTS

# create a pupil
pupil_grid = hcipy.make_pupil_grid(512, diameter=PUPIL_DIAMETER)
ap = hcipy.make_circular_aperture(PUPIL_DIAMETER)(pupil_grid)

wf = hcipy.Wavefront(ap, wavelength=CWL)

# modes for the DM
modebasis = hcipy.make_zernike_basis(
    num_modes=NUM_MODES,
    D=PUPIL_DIAMETER,
    grid=pupil_grid
)

# create the DM
dm = hcipy.DeformableMirror(
    modebasis
)

lyot_occulting_mask = hcipy.make_circular_aperture(OCCULTMASK_SIZE)
lyot_occulting_mask = hcipy.make_inverted_aperture(lyot_occulting_mask)
lyot_occulting_grid = hcipy.make_focal_grid(
    20, 2, f_number=INPUT_F_NUMBER, reference_wavelength=CWL
)
focal_plane_grid = lyot_occulting_mask(lyot_occulting_grid)
focal_plane_grid = hcipy.Apodizer(focal_plane_grid)


lyot_stop_mask = hcipy.make_circular_aperture(PUPIL_DIAMETER*0.95)
lyot_stop_grid = pupil_grid.copy()
lyot_stop_field = lyot_stop_mask(lyot_stop_grid)
lyot_stop_field = hcipy.Apodizer(lyot_stop_field)

LyotCoronaGraph = hcipy.LyotCoronagraph(
    input_grid=ap,
    focal_plane_mask=focal_plane_grid,
    lyot_stop=lyot_stop_field,
    focal_length=756e-3)

dm.random(1e-9)
wf = dm.forward(wf)

wf2 = LyotCoronaGraph.forward(wf)

wf2 = hcipy.Magnifier(0.265)(wf2)

fpgrid  = hcipy.make_focal_grid(20, 30, f_number=850, reference_wavelength=800e-9)

ELTFocusProp = hcipy.FraunhoferPropagator(
    pupil_grid, fpgrid, focal_length=850 * wf2.grid.delta[0] * wf2.grid.shape[0])

#wf = hcipy.Wavefront(eltgrid, wavelength=800e-9)
wf2 = ELTFocusProp.forward(wf2)
#wf2 = hcipy.Wavefront(hcipy.make_rectangular_aperture(0.01)(fpgrid), wavelength=800e-9)

plt.figure()
hcipy.imshow_field(wf2.power, grid_units=1e-3)
plt.show()

import matplotlib.pyplot as plt


#test_im = hcipy.make_pupil_grid(128, 0.25)


im = ifs_sim.ImageSlicer('./slicer.cfg')
ims = im._split_field(wf2)
pups = im._propto_pupil_mirror(ims)

#print(ims[0].grid.shape, ims[0].grid.delta)

pups_apod = im._apply_pupil_mirror(pups)

exit_slits = im._propto_exit_slit(pups_apod)

exit_slit = im._create_exit_slit(exit_slits)

print(ims[0].amplitude)

plt.figure()
hcipy.imshow_field(ims[0].power)
plt.colorbar()
plt.figure()
hcipy.imshow_field(pups_apod[0].power)
plt.show()
plt.figure()
hcipy.imshow_field(exit_slits[0].power, norm='log')
plt.show()
plt.figure()
plt.imshow(exit_slit)
plt.show()