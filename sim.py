import hcipy
import ifs_sim, ifs_sim_tools


# create a pupil
"""ELT_DIA = 39.14634
pupil_grid = hcipy.make_pupil_grid(dims=1024, diameter=ELT_DIA)
ELT_AP  = hcipy.make_elt_aperture()
eltgrid = ELT_AP(pupil_grid)"""

pupil_grid = hcipy.make_pupil_grid(512, diameter=3.35e-3)
ap = hcipy.make_circular_aperture(3.35e-3)(pupil_grid)


modebasis = hcipy.make_zernike_basis(
    num_modes=50,
    D=3.35e-3,
    grid=pupil_grid
)

dm = hcipy.DeformableMirror(
    modebasis
)



lyot_spot = 2 * 

LyotCoronaGraph = hcipy.LyotCoronagraph(
    input_grid=ap,

)




fpgrid  = hcipy.make_focal_grid(20, 30, f_number=850, reference_wavelength=800e-9)
print(fpgrid.x)


ELTFocusProp = hcipy.FraunhoferPropagator(pupil_grid, fpgrid, focal_length=850 * ELT_DIA)

wf = hcipy.Wavefront(eltgrid, wavelength=800e-9)
wf2 = ELTFocusProp.forward(wf)
#wf2 = hcipy.Wavefront(hcipy.make_rectangular_aperture(0.01)(fpgrid), wavelength=800e-9)
import matplotlib.pyplot as plt

"""plt.figure()
hcipy.imshow_field(wf2.amplitude, norm='log', grid_units=1e-3)
plt.colorbar()
plt.show()"""

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