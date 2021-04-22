from astropy.io import fits as pf
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interpolate
import numpy as np
from numpy.polynomial.polynomial import polyval
from ngps import NGPS
import pkg_resources
import os
import matplotlib.pyplot as pl


class AtlasSpectrum(NGPS):
    """
    Atlas spectrum class for reading an atlas spectrum for NGPS

    The Thorium-Argon lamp is from the NOAO website here:
        http://iraf.noao.edu/specatlas/thar/thar.html,
    where it specifies that in the wavelength range 3,262 - 10,598 Ang, the
    resolution is (lambda/dlambda) = 75,000.  The region from 3,005 to 3262 Ang
    has a resolution of 12,000.

    """
    flux = None
    waves = None
    header = None
    native_dispersion = None
    det_flux = None
    det_waves = None
    det_res_pixels = None
    det_seg_waves = None
    lamp = None

    def __init__(self, lamp, verbose=False):

        self.lamp = lamp

        # path to atlas for input lamp
        path = "data/%s.fits" % lamp.lower()
        pkg = __name__.split('.')[0]
        atpath = pkg_resources.resource_filename(pkg, path)

        # verify path
        if os.path.exists(atpath):
            print("Reading atlas spectrum in: %s" % atpath)
        else:
            print("ERROR: Atlas spectrum not found for %s" % atpath)

        # read in atlas data and header
        ff = pf.open(atpath)
        self.flux = ff[0].data
        self.header = ff[0].header

        # construct native wavlength scale
        self.native_dispersion = ff[0].header['CDELT1']
        self.waves = np.arange(0, len(ff[0].data)) * ff[0].header['CDELT1'] + \
            ff[0].header['CRVAL1']

        print("Read %d points covering %.2f to %.2f Angstroms" %
              (len(self.waves),
               float(np.nanmin(self.waves)), float(np.nanmax(self.waves))))
        ff.close()

        # get a resampled spectrum for each detector
        det_flux = []
        det_waves = []
        det_res_pixels = []
        det_seg_waves = []

        # loop over detectors (u, g, r, i)
        for i, lims in enumerate(self.detector_wave_limits):

            # get model wavelength scale
            det_pix = np.arange(0, self.detector_npix[i])
            waves = polyval(det_pix, self.detector_disp_coeffs[i])
            flux = waves.copy()
            det_waves.append(waves)

            # get resolution in pixels
            res_pix = []
            seg_waves = []

            # loop over resolution segments (11)
            for j, slit_pix in enumerate(self.slit_pix[i]):

                # Skip the last one
                if j < len(self.slit_pix[i]) - 1:

                    # segment slit projection in pixels
                    sp = (slit_pix + self.slit_pix[i][j+1]) / 2.

                    # add spot size in quadrature
                    eff_res = np.sqrt(sp**2 +
                                      (self.spot_size_microns[i][j] /
                                       self.pix_size_microns)**2)

                    # get segment effective wavelength
                    seg_waves.append(
                        (self.seg_waves[i][j] + self.seg_waves[i][j+1]) / 2.
                    )

                    # store resolution
                    res_pix.append(eff_res)

                    # filter atlas spectrum to resolution
                    spec = gaussian_filter1d(self.flux, eff_res)

                    # make interpolation function
                    det_int = interpolate.interp1d(self.waves, spec, kind='cubic',
                                                   bounds_error=False,
                                                   fill_value='extrapolate')
                    # get segment wavelength limits
                    w0 = self.seg_waves[i][j]
                    w1 = self.seg_waves[i][j+1]

                    # segment pixels
                    seg_pix = [k for k, w in enumerate(waves) if w0 <= w <= w1]

                    # resolution corrected flux for this segment
                    flux[seg_pix] = det_int(waves[seg_pix])

                    # report if verbose
                    if verbose:
                        print("%s-band: seg %d; %.2f - %.2f A at %.4f px sig"
                              % (self.det_bands[i], j, w0, w1, sp))

            # store results
            det_flux.append(flux)
            det_res_pixels.append(res_pix)
            det_seg_waves.append(seg_waves)

        # record results
        self.det_flux = det_flux
        self.det_waves = det_waves
        self.det_res_pixels = det_res_pixels
        self.det_seg_waves = det_seg_waves

    def plot_spec(self):
        pl.clf()
        for i in range(self.n_det):
            pl.plot(self.det_waves[i], self.det_flux[i],
                    color=self.det_colors[i], label=self.det_bands[i])
        pl.title("NGPS Simulated %s" % self.lamp)
        pl.xlabel("Wavelength(A)")
        pl.ylabel("Arb. Flux")
        pl.legend()
        pl.show()

    def plot_res(self):
        pl.clf()
        for i in range(self.n_det):
            pl.plot(self.det_seg_waves[i], self.det_res_pixels[i],
                    color=self.det_colors[i], label=self.det_bands[i])
        pl.title("NGPS Resolution 0.5\" slit")
        pl.xlabel("Wavelength(A)")
        pl.ylabel("RMS px")
        pl.legend()
        pl.show()
