import astropy.io.fits as pf
import matplotlib.pyplot as pl
import argparse
import numpy as np


def et_calc(reflectivity=0.6, etalon_separation=40.,
            fits_file=None, delta_wave=0.2, do_plot=False):
    """Calculate etalon throughput using a simplified formula

        T = 1 / (1 + 4 * R / ( (1 - R)^2) * ( sin(2*pi*t / wl) ^2))
        R = reflectivity = 0.6
        t - etalon separation = 40um
        wl = wavelength
        Wikipedia has a good article on it:
        https://en.wikipedia.org/wiki/Fabry%E2%80%93P%C3%A9rot_interferometer
    """
    # set up constants
    refl = reflectivity
    etsp = etalon_separation * 10000.
    dw = delta_wave
    w0 = 3000.
    w1 = 10500.

    wls = np.arange(start=w0, stop=w1, step=dw, dtype=np.float)
    print("%d points" % len(wls))
    thpt = 1. / (1. + 4. * refl /
                 ((1.-refl)**2) * (np.sin(2.*np.pi*etsp / wls) ** 2))
    # plot
    if do_plot:
        pl.plot(wls, thpt, label='Et')
        pl.legend()
        pl.xlabel('Wavelength(A)')
        pl.ylabel('Thrpt')
        pl.show()
    if fits_file:
        # create new file
        hdu = pf.PrimaryHDU(thpt)
        hdu.header['CDELT1'] = dw
        hdu.header['CRVAL1'] = w0
        hdu.header['ETALON'] = True
        hdu.writeto(fits_file)
        print("created %s" % fits_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Convert CSV file from Lightmachines into a linear dispersion FITS file
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='input CSV file')
    parser.add_argument('-o', '--outfile', type=str, default='et.fits',
                        help='output FITS file')
    parser.add_argument('-p', '--plot', action="store_true", default=False,
                        help='set to plot spectrum')

    args = parser.parse_args()

    if args.filename is not None:
        etfits(csv_file=args.filename, fits_file=args.outfile,
               do_plot=args.plot)
