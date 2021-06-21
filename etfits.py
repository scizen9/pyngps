import pandas
from scipy.interpolate import interpolate
import astropy.io.fits as pf
import matplotlib.pyplot as pl
import argparse
import numpy as np


def etfits(csv_file=None, fits_file='et.fits', do_plot=False):
    # read into pandas data frame
    df = pandas.read_csv(csv_file, comment='#')
    # set up a linear wavelength scale
    dw = 0.2    # Angstroms
    w0 = float(np.nanmin(df['wl1'])) * 10.
    w1 = float(np.nanmax(df['wl1'])) * 10.
    wl_ref = np.arange(w0, w1, step=dw)
    # set up interpolation
    wl_interp = interpolate.interp1d(df['wl1']*10., df['tft'], kind='linear',
                                     bounds_error=False,
                                     fill_value='extrapolate')
    # interpolate onto linear wavelength scale
    tft = wl_interp(wl_ref)
    neg = np.where(tft < 0.)
    tft[neg] = 0.
    pos = np.where(tft > 100.)
    tft[pos] = 100.
    # plot
    if do_plot:
        pl.plot(wl_ref, tft, label='Int')
        pl.plot(df['wl1']*10., df['tft'], label='Dat')
        pl.legend()
        pl.xlabel('Wavelength(A)')
        pl.ylabel('Thrpt')
        pl.show()
    # create new file
    hdu = pf.PrimaryHDU(tft)
    hdu.header['CDELT1'] = dw
    hdu.header['CRVAL1'] = w0
    hdu.header['OFNAME'] = csv_file
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
