from astropy.io import fits as pf
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel,\
    Trapezoid1DKernel
from scipy.interpolate import interpolate
from scipy.signal.windows import boxcar
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
import numpy as np
import scipy as sp
from numpy.polynomial.polynomial import polyval
from ngps import NGPS
import pkg_resources
import os
import matplotlib.pyplot as pl


def gaus(x, a, mu, sigma):
    """Gaussian fitting function"""
    return a * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def get_line_window(y, c, thresh=0., verbose=False, strict=False):
    """Find a window that includes the fwhm of the line"""
    nx = len(y)
    # check edges
    if c < 2 or c > nx - 2:
        if verbose:
            print("input center too close to edge")
        return None, None, 0
    # get initial values
    x0 = c - 2
    x1 = c + 2
    mx = np.nanmax(y[x0:x1+1])
    count = 5
    # check low side
    if x0 - 1 < 0:
        if verbose:
            print("max check: low edge hit")
        return None, None, 0
    while y[x0-1] > mx:
        x0 -= 1
        count += 1
        if x0 - 1 < 0:
            if verbose:
                print("Max check: low edge hit")
            return None, None, 0

    # check high side
    if x1 + 1 >= nx:
        if verbose:
            print("max check: high edge hit")
        return None, None, 0
    while y[x1+1] > mx:
        x1 += 1
        count += 1
        if x1 + 1 >= nx:
            if verbose:
                print("Max check: high edge hit")
            return None, None, 0
    # adjust starting window to center on max
    cmx = x0 + y[x0:x1+1].argmax()
    x0 = cmx - 2
    x1 = cmx + 2
    mx = np.nanmax(y[x0:x1 + 1])
    # make sure max is high enough
    if mx < thresh:
        return None, None, 0
    #
    # expand until we get to half max
    hmx = mx * 0.5
    #
    # Low index side
    prev = mx
    while y[x0] > hmx:
        if y[x0] > mx or x0 <= 0 or y[x0] > prev:
            if verbose:
                if y[x0] > mx:
                    print("hafmax check: low index err - missed max")
                if x0 <= 0:
                    print("hafmax check: low index err - at edge")
                if y[x0] > prev:
                    print("hafmax check: low index err - wiggly")
            return None, None, 0
        prev = y[x0]
        x0 -= 1
        count += 1
    # High index side
    prev = mx
    while y[x1] > hmx:
        if y[x1] > mx or x1 >= nx or y[x1] > prev:
            if verbose:
                if y[x1] > mx:
                    print("hafmax check: high index err - missed max")
                if x1 >= nx:
                    print("hafmax check: high index err - at edge")
                if y[x1] > prev:
                    print("hafmax check: high index err - wiggly")
            return None, None, 0
        prev = y[x1]
        if x1 < (nx-1):
            x1 += 1
            count += 1
        else:
            if verbose:
                print("Edge encountered")
            return None, None, 0
    if strict:
        # where did we end up?
        if c < x0 or x1 < c:
            if verbose:
                print("initial position outside final window")
            return None, None, 0

    return x0, x1, count
    # END: get_line_window()


def findpeaks(x, y, wid, sth, ath, pkg=None, verbose=True):
    """Find peaks in spectrum"""
    # derivative
    grad = np.gradient(y)
    # smooth derivative
    win = boxcar(wid)
    d = sp.signal.convolve(grad, win, mode='same') / sum(win)
    # size
    nx = len(x)
    # set up windowing
    if not pkg:
        pkg = wid
    hgrp = int(pkg/2)
    print("hgrp: %d" % hgrp)
    hgt = []
    pks = []
    sgs = []
    # loop over spectrum
    # limits to avoid edges given pkg
    for i in np.arange(pkg, (nx - pkg), dtype=int):
        # find zero crossings
        if np.sign(d[i]) > np.sign(d[i+1]):
            # pass slope threshhold?
            if (d[i] - d[i+1]) > sth * y[i]:
                # pass amplitude threshhold?
                if y[i] > ath or y[i+1] > ath:
                    # get subvectors around peak in window
                    xx = x[(i-hgrp):(i+hgrp+1)]
                    yy = y[(i-hgrp):(i+hgrp+1)]
                    if len(yy) > 3:
                        try:
                            # gaussian fit
                            res, _ = curve_fit(gaus, xx, yy,
                                               p0=[y[i], x[i], 1.])
                            # check offset of fit from initial peak
                            r = abs(x - res[1])
                            t = r.argmin()
                            if abs(i - t) > pkg:
                                if verbose:
                                    print(i, t, x[i], res[1], x[t])
                            else:
                                hgt.append(res[0])
                                pks.append(res[1])
                                sgs.append(abs(res[2]))
                        except RuntimeError:
                            continue
    # clean by sigmas
    cvals = []
    cpks = []
    sgmn = None
    if len(pks) > 0:
        cln_sgs, low, upp = sigmaclip(sgs, low=3., high=3.)
        for i in range(len(pks)):
            if low < sgs[i] < upp:
                cpks.append(pks[i])
                cvals.append(hgt[i])
        sgmn = cln_sgs.mean()
        # sgmd = float(np.nanmedian(cln_sgs))
    else:
        print("No peaks found!")
    return cpks, sgmn, cvals
    # END: findpeaks()


def ngps_noise_model(spec, gain, rdnoise):
    """
    Calculate noise model.

    Includes Poisson and readnoise

    :param spec: flux per pixel in DN
    :param gain: e-/DN
    :param rdnoise: in e-
    :return: flux per pixel with noise
    """

    # Only positive values
    pos_spec = np.absolute(spec)
    # Add Poisson noise first
    noisy = np.random.poisson(pos_spec)

    # Add Read Noise
    mean = 0.
    sigma = rdnoise / gain  # convert to DN
    gauss = np.random.normal(mean, sigma, len(spec))

    # Calc noise
    noise = np.sqrt(pos_spec) + sigma

    return noisy + gauss, noise


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
    peaks = None
    native_dispersion = None
    det_flux = None
    det_noise = None
    det_flux_with_noise = None
    det_waves = None
    det_thrpt = None
    det_res_pixels = None
    det_seg_waves = None
    det_seg_slit = None
    det_seg_spot = None
    lamp = None
    ymax = None

    def __init__(self, lamp, calib=True, kernel='box', verbose=False):

        self.lamp = lamp
        self.ymax = 0.

        # path to atlas for input lamp
        path = "data/%s.fits" % lamp.lower()
        pkg = __name__.split('.')[0]
        atpath = pkg_resources.resource_filename(pkg, path)

        # verify path
        if os.path.exists(atpath):
            print("Reading atlas spectrum in: %s" % atpath)
        else:
            print("ERROR: Atlas spectrum not found for %s" % atpath)
            return

        # check kernel
        if 'gaussian' not in kernel.lower():
            print('Warning: kernel will be identical at all wavelengths')

        # read in throughput and set wavelength scale
        self.read_thrpt()
        thrpt_waves = self.thrpt[:, 0] * 10.  # convert from nm to Ang

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

        # get line list
        smooth_width = 4
        peak_width = 5.0
        slope_thresh = 0.07 * smooth_width / 2. / 100.
        ampl_thresh = 0.
        at_cent, at_avwsg, at_hgt = findpeaks(self.waves, self.flux,
                                              smooth_width, slope_thresh,
                                              ampl_thresh, peak_width)
        self.peaks = at_cent

        # get proper gain values
        if calib:
            gains = self.det_gain_cal
        else:
            gains = self.det_gain

        # get a resampled spectrum for each detector
        det_flux = []           # spectrum
        det_noise = []          # noise
        det_flux_with_noise = []  # pure spectrum
        det_waves = []          # wavelengths
        det_thrpt = []          # throughput
        det_seg_res = []        # seg resolution (pixels)
        det_seg_waves = []      # seg waves (A)
        det_seg_slit = []       # seg slit size (pixels)
        det_seg_spot = []       # seg spot size (pixels)
        max_flux = 0.

        # loop over detectors (u, g, r, i)
        for idet in range(self.n_det):

            # get model wavelength scale
            det_pix = np.arange(0, self.detector_npix[idet])
            waves = polyval(det_pix, self.detector_disp_coeffs[idet])
            det_waves.append(waves)
            flux = waves.copy()

            # make interpolation function
            thrpt_int = interpolate.interp1d(thrpt_waves, self.thrpt[:, idet+1],
                                             kind='cubic',
                                             bounds_error=False,
                                             fill_value='extrapolate')
            # get throughput on det wavelength scale
            thrpt = thrpt_int(waves)
            det_thrpt.append(thrpt)

            # get resolution in pixels, which varies over the segments
            seg_res_pix = []
            seg_waves = []
            seg_spot_pix = []
            seg_slit_pix = []

            # loop over optical model segments (11)
            for iseg in range(len(self.seg_waves[idet])):

                # Skip the last one
                if iseg < len(self.slit_pix[idet]) - 1:

                    # segment slit projection in pixels for segment
                    slit_pix = (self.slit_pix[idet][iseg] +
                                self.slit_pix[idet][iseg+1]) / 2.
                    seg_slit_pix.append(slit_pix)

                    # spot size for segment
                    spot_pix = (self.spot_size_microns[idet][iseg] +
                                self.spot_size_microns[idet][iseg+1]) / (15. *
                                                                         2.)
                    seg_spot_pix.append(spot_pix)

                    # add slit size and spot size to get resolution in pixels
                    res_pix = np.sqrt(slit_pix**2 + spot_pix**2)
                    seg_res_pix.append(res_pix)

                    # get segment wavelength limits
                    w0 = self.seg_waves[idet][iseg]
                    w1 = self.seg_waves[idet][iseg + 1]

                    # get segment effective wavelength
                    seg_waves.append((w0 + w1) / 2.)

                    # filter atlas spectrum to resolution
                    # get appropriate kernel for convolution
                    if 'gaussian' in kernel.lower():
                        con_kernel = Gaussian1DKernel(res_pix)
                    elif 'trapazoid' in kernel.lower():
                        con_kernel = Trapezoid1DKernel(round(res_pix), 0.75,
                                                       mode='oversample')
                    else:
                        con_kernel = Box1DKernel(round(res_pix))

                    spec = convolve(self.flux, con_kernel)
                    # spec = gaussian_filter1d(self.flux, res_pix)

                    # make interpolation function
                    det_int = interpolate.interp1d(self.waves, spec,
                                                   kind='cubic',
                                                   bounds_error=False,
                                                   fill_value='extrapolate')

                    # segment pixels
                    seg_pix = [k for k, w in enumerate(waves) if w0 <= w <= w1]

                    # resolution corrected flux for this segment
                    flux[seg_pix] = det_int(waves[seg_pix])

                    # report if verbose
                    if verbose:
                        print("%s-band: seg %d; %.2f - %.2f A at %.4f px sig"
                              % (self.det_bands[idet], iseg, w0, w1, res_pix))

            # apply throughput function
            flux *= thrpt

            # store results
            det_flux.append(flux)
            det_seg_res.append(seg_res_pix)
            det_seg_waves.append(seg_waves)
            det_seg_slit.append(seg_slit_pix)
            det_seg_spot.append(seg_spot_pix)
            # get max flux
            if np.nanmax(flux) > max_flux:
                max_flux = np.nanmax(flux)

            # update gain in header
            self.header["GAIN%d" % (idet+1)] = (gains[idet], 'e-/ADU')

        # normalize flux
        flux_scale = 1.0
        for idet in range(self.n_det):
            if (60000 * gains[idet]) > self.ymax:
                self.ymax = (60000 * gains[idet])
            flux_scale = (60000 * gains[idet]) / np.nanmax(det_flux[idet])
            scaled_flux = det_flux[idet] * flux_scale
            det_flux[idet] = scaled_flux.astype(int)    # as raw data
            # apply noise
            noisy, noise = ngps_noise_model(det_flux[idet], gains[idet],
                                            self.det_readnoise[idet])
            det_flux_with_noise.append(noisy)
            det_noise.append(noise)

        self.flux *= (flux_scale / 5.0)     # scale for reference

        # record results
        self.det_flux = det_flux
        self.det_noise = det_noise
        self.det_flux_with_noise = det_flux_with_noise
        self.det_waves = det_waves
        self.det_thrpt = det_thrpt
        self.det_res_pixels = det_seg_res
        self.det_seg_waves = det_seg_waves
        self.det_seg_slit = det_seg_slit
        self.det_seg_spot = det_seg_spot

    def plot_spec(self):
        pl.plot(self.waves, self.flux, color='gray', alpha=0.5,
                label="Atlas")
        pl.plot(self.peaks, np.zeros(len(self.peaks)), 'g+')
        for i in range(self.n_det):
            pl.errorbar(self.det_waves[i], self.det_flux_with_noise[i],
                        yerr=self.det_noise[i], color=self.det_colors[i],
                        ecolor='black', label=self.det_bands[i], barsabove=True)
            if i == 0:
                pl.plot(self.det_waves[i], self.det_flux[i], color='black',
                        ls='--', label="No Noise")
            pl.plot(self.det_waves[i], self.det_flux[i],
                    color=self.det_colors[i], ls='--')
        pl.title("NGPS Simulated %s" % self.lamp)
        pl.xlabel("Wavelength(A)")
        pl.ylabel("Simulated DN")
        pl.legend()
        pl.ylim((-200, self.ymax))
        pl.xlim((3000., 10500.))
        pl.show()

    def plot_s2n(self):
        for i in range(self.n_det):
            s2n = self.det_flux[i] / self.det_noise[i]
            pl.plot(self.det_waves[i], s2n, color=self.det_colors[i],
                    label=self.det_bands[i])
        pl.title("NGPS Simulated S/N for %s" % self.lamp)
        pl.xlabel("Wavelength(A)")
        pl.ylabel("Simulated S/N per px")
        pl.legend()
        pl.show()

    def plot_thrpt(self):
        for i in range(self.n_det):
            pl.plot(self.det_waves[i], self.det_thrpt[i],
                    color=self.det_colors[i], label=self.det_bands[i])
        pl.title("NGPS Calculated Throughput")
        pl.xlabel("Wavelength(A)")
        pl.ylabel("Throughput")
        pl.legend()
        pl.show()

    def plot_res(self):
        for idet in range(self.n_det):
            pl.plot(self.det_seg_waves[idet], self.det_res_pixels[idet],
                    color=self.det_colors[idet], label=self.det_bands[idet])
            pl.scatter(self.det_seg_waves[idet], self.det_seg_slit[idet],
                       color=self.det_colors[idet], marker='+')
            pl.scatter(self.det_seg_waves[idet], self.det_seg_spot[idet],
                       color=self.det_colors[idet], marker='x')
        pl.text(6000., 0.75, "Spot Size")
        pl.text(6000., 2.45, "Slit Size")
        pl.title("NGPS Resolution 0.5\" slit")
        pl.xlabel("Wavelength(A)")
        pl.ylabel("RMS px")
        pl.legend(loc="center right")
        pl.grid()
        pl.show()
