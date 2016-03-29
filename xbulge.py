from __future__ import print_function

import os
import tempfile

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='computer modern roman')
matplotlib.rc('font', **{'sans-serif': 'computer modern sans serif'})

import pylab as plt
import numpy as np
import fitsio
from astrometry.util.util import (anwcs_create_hammer_aitoff, Tan,
                                  median_smooth)
from astrometry.util.starutil_numpy import radectolb
from astrometry.util.fits import fits_table

def main():
    '''
    This function generates the plots in the paper.

    Some files and directories are assumed to exist in the current directory:

    * WISE atlas tiles, from http://unwise.me/data/allsky-atlas.fits
    * unwise-neo1-coadds, from http://unwise.me/data/neo1/
    * unwise-neo1-coadds-half, unwise-neo1-coadds-quarter: directories

    '''
    # First, create the WCS into which we want to render
    # degrees width to render in galactic coords
    # |l| < 60
    # |b| < 30
    width = 120
    # ~2 arcmin per pixel
    W = int(width * 60.) / 2
    H = W/2
    zoom = 360. / width
    wcs = anwcs_create_hammer_aitoff(0., 0., zoom, W, H, 0)

    # Select WISE tiles that overlap.  This atlas table is available
    # from http://unwise.me/data/allsky-atlas.fits

    # Select WISE tiles that overlap.
    T = fits_table('allsky-atlas.fits')
    print(len(T), 'tiles total')
    T.ll,T.bb = radectolb(T.ra, T.dec)
    I = np.flatnonzero(np.logical_or(T.ll < width+1,
                                     T.ll > (360-width-1)) *
                                     (T.bb > -width/2-1) * (T.bb < width/2+1))
    T.cut(I)
    print(len(I), 'tiles in L,B range')

    # Create a coadd for each WISE band
    lbpat = 'unwise-neo1-w%i-lb.fits'
    imgs = []
    for band in [1,2]:
        outfn = lbpat % (band)
        if os.path.exists(outfn):
            print('Exists:', outfn)
            img = fitsio.read(outfn)
            imgs.append(img)
            continue

        coimg  = np.zeros((H,W), np.float32)
        conimg = np.zeros((H,W), np.float32)

        for i,brick in enumerate(T.coadd_id):
            # We downsample by 2, twice, just to make repeat runs a
            # little faster.
            # unWISE
            fn = os.path.join('unwise-neo1-coadds', brick[:3], brick,
                              'unwise-%s-w%i-img-u.fits' % (brick, band))
            qfn = os.path.join('unwise-neo1-coadds-quarter',
                               'unwise-%s-w%i.fits' % (brick, band))
            hfn = os.path.join('unwise-neo1-coadds-half',
                               'unwise-%s-w%i.fits' % (brick, band))

            if not os.path.exists(qfn):
                if not os.path.exists(hfn):
                    print('Reading', fn)
                    halfsize(fn, hfn)
                halfsize(hfn, qfn)
            fn = qfn

            print('Reading', fn)
            img = fitsio.read(fn)
            bwcs = Tan(fn, 0)
            bh,bw = img.shape

            # Coadd each unWISE pixel into the nearest target pixel.
            xx,yy = np.meshgrid(np.arange(bw), np.arange(bh))
            rr,dd = bwcs.pixelxy2radec(xx, yy)
            ll,bb = radectolb(rr.ravel(), dd.ravel())
            ll = ll.reshape(rr.shape)
            bb = bb.reshape(rr.shape)
            ok,ox,oy = wcs.radec2pixelxy(ll, bb)
            ox = np.round(ox - 1).astype(int)
            oy = np.round(oy - 1).astype(int)
            K = (ox >= 0) * (ox < W) * (oy >= 0) * (oy < H) * ok

            #print('ok:', np.unique(ok), 'x', ox.min(), ox.max(), 'y', oy.min(), oy.max())
            assert(np.all(np.isfinite(img)))
            if np.sum(K) == 0:
                # no overlap
                print('No overlap')
                continue
    
            np.add.at( coimg, (oy[K], ox[K]), img[K])
            np.add.at(conimg, (oy[K], ox[K]), 1)

        img = coimg / np.maximum(conimg, 1)

        # Hack -- write and then read FITS WCS header.
        fn = 'wiselb.wcs'
        wcs.writeto(fn)
        hdr = fitsio.read_header(fn)
        hdr['CTYPE1'] = 'GLON-AIT'
        hdr['CTYPE2'] = 'GLAT-AIT'

        fitsio.write(outfn, img, header=hdr, clobber=True)
        fitsio.write(outfn.replace('.fits', '-n.fits'), conimg,
                     header=hdr, clobber=True)
        imgs.append(img)

    w1,w2 = imgs

    # Get/confirm L,B bounds...
    H,W = w1.shape
    print('Image size', W, 'x', H)
    ok,l1,b1 = wcs.pixelxy2radec(1, (H+1)/2.)
    ok,l2,b2 = wcs.pixelxy2radec(W, (H+1)/2.)
    ok,l3,b3 = wcs.pixelxy2radec((W+1)/2., 1)
    ok,l4,b4 = wcs.pixelxy2radec((W+1)/2., H)
    print('L,B', (l1,b1), (l2,b2), (l3,b3), (l4,b4))
    llo,lhi = l2,l1+360
    blo,bhi = b3,b4
    
    # Set plot sizes
    plt.figure(1, figsize=(10,5))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    plt.figure(2, figsize=(5,5))
    plt.subplots_adjust(left=0.11, right=0.96, bottom=0.1, top=0.95)

    suffix = '.pdf'
    
    rgb = wise_rgb(w1, w2)
    xlo,ylo = 0,0
    
    plt.figure(1)
    plt.clf()
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo, ylo, lticks=[60,30,0,330,300], bticks=[-30,-15,0,15,30])
    plt.savefig('xbulge-00' + suffix)

    # Compute the median of each row as a crude way of suppressing the
    # Galactic plane
    medy1 = np.median(w1, axis=1)
    medy2 = np.median(w2, axis=1)

    rgb = wise_rgb(w1 - medy1[:,np.newaxis],
                   w2 - medy2[:,np.newaxis])

    # Zoom in a bit for Galactic plane subtracted version
    lhi,llo,blo,bhi = 40, 320, -20, 20
    okxy = np.array([wcs.radec2pixelxy(l,b) for l,b in [
            (llo, blo), (llo, bhi), (lhi, blo), (lhi, bhi)]])
    xlo = int(np.floor(min(okxy[:,-2])))
    xhi = int(np.ceil (max(okxy[:,-2])))
    ylo = int(np.floor(min(okxy[:,-1])))
    yhi = int(np.ceil (max(okxy[:,-1])))
    
    plt.clf()
    plt.imshow(rgb[ylo:yhi, xlo:xhi, :],origin='lower', interpolation='nearest')
    #lbticks(wcs, xlo, ylo, lticks=[40,20,0,340,320], bticks=[-20,-10,0,10,20])
    lbticks(wcs, xlo, ylo, lticks=[30,15,0,345,330], bticks=[-20,-10,0,10,20])
    plt.savefig('xbulge-01' + suffix)

    # Zoom in on the core
    lhi,llo,blo,bhi = 15, 345, -15, 15
    ok,x1,y1 = wcs.radec2pixelxy(llo, blo)
    ok,x2,y2 = wcs.radec2pixelxy(llo, bhi)
    ok,x3,y3 = wcs.radec2pixelxy(lhi, blo)
    ok,x4,y4 = wcs.radec2pixelxy(lhi, bhi)

    xlo = int(np.floor(min(x1,x2,x3,x4)))
    xhi = int(np.ceil (max(x1,x2,x3,x4)))
    ylo = int(np.floor(min(y1,y2,y3,y4)))
    yhi = int(np.ceil (max(y1,y2,y3,y4)))
    print('xlo,ylo', xlo, ylo)

    w1 = w1[ylo:yhi, xlo:xhi]
    w2 = w2[ylo:yhi, xlo:xhi]

    plt.figure(2)

    # Apply color cut
    w1mag = -2.5*(np.log10(w1) - 9.)
    w2mag = -2.5*(np.log10(w2) - 9.)
    cc = w1mag - w2mag
    goodcolor = np.isfinite(cc)
    mlo,mhi = np.percentile(cc[goodcolor], [5,95])
    print('W1 - W2 color masks:', mlo,mhi)
    mask = goodcolor * (cc > mlo) * (cc < mhi)

    plt.clf()
    rgb = wise_rgb(w1, w2)
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Data')
    plt.savefig('xbulge-fit-data' + suffix)

    plt.clf()
    rgb = wise_rgb(w1 * mask, w2 * mask)
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Data (masked)')
    plt.savefig('xbulge-fit-masked' + suffix)
    
    ie = mask.astype(np.float32)

    from tractor import (Image, NCircularGaussianPSF, LinearPhotoCal, Tractor,
                         PixPos, Fluxes)
    from tractor.galaxy import ExpGalaxy, GalaxyShape

    # Create Tractor images
    tim1 = Image(data=w1 * mask, inverr=ie,
                 psf=NCircularGaussianPSF([1.],[1.]),
                 photocal=LinearPhotoCal(1., 'w1'))
    tim2 = Image(data=w2 * mask, inverr=ie,
                 psf=NCircularGaussianPSF([1.],[1.]),
                 photocal=LinearPhotoCal(1., 'w2'))
    H,W = w1.shape
    gal = ExpGalaxy(PixPos(W/2, H/2), Fluxes(w1=w1.sum(), w2=w2.sum()),
                    GalaxyShape(200, 0.4, 90.))
    tractor = Tractor([tim1, tim2],[gal])

    # fitsio.write('data-w1.fits', w1 * mask, clobber=True)
    # fitsio.write('data-w2.fits', w2 * mask, clobber=True)
    # fitsio.write('mask.fits', mask.astype(np.uint8), clobber=True)

    # Optimize galaxy model
    tractor.freezeParam('images')
    for step in range(50):
        dlnp,x,alpha = tractor.optimize()
        print('dlnp', dlnp)
        print('x', x)
        print('alpha', alpha)
        print('Galaxy', gal)
        if dlnp == 0:
            break

    # Get galaxy model images, compute residuals
    mod1 = tractor.getModelImage(0)
    resid1 = w1 - mod1
    mod2 = tractor.getModelImage(1)
    resid2 = w2 - mod2

    rgb = wise_rgb(mod1, mod2)
    plt.clf()
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Model')
    plt.savefig('xbulge-fit-model' + suffix)

    rgb = resid_rgb(resid1, resid2)
    plt.clf()
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Residuals')
    plt.savefig('xbulge-fit-resid' + suffix)

    rgb = resid_rgb(resid1*mask, resid2*mask)
    plt.clf()
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Residuals (masked)')
    plt.savefig('xbulge-fit-residmasked' + suffix)

    # fitsio.write('resid1.fits', resid1, clobber=True)
    # fitsio.write('resid2.fits', resid2, clobber=True)

    # Compute median-smoothed residuals
    fr1 = np.zeros_like(resid1)
    fr2 = np.zeros_like(resid2)
    median_smooth(resid1, np.logical_not(mask), 25, fr1)
    median_smooth(resid2, np.logical_not(mask), 25, fr2)

    rgb = resid_rgb(fr1, fr2)
    plt.clf()
    plt.imshow(rgb, origin='lower', interpolation='nearest')
    lbticks(wcs, xlo,ylo)
    plt.title('Residuals (smoothed)')
    plt.savefig('xbulge-fit-smooth2' + suffix)
    

def wise_rgb(w1, w2):
    ''' Converts WISE W1 and W2 images into an RGB image,
    with arcsinh stretch.
    '''
    import numpy as np
    H,W = w1.shape

    S,Q = 30000,25
    alpha = 1.5

    b = w1 / S
    r = w2 / S
    g = (r + b) / 2.

    # FIXME -- needed?
    m = -2e-2
    
    r = np.maximum(0, r - m)
    g = np.maximum(0, g - m)
    b = np.maximum(0, b - m)
    I = (r+g+b)/3.
    fI = np.arcsinh(alpha * Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    R = fI * r / I
    G = fI * g / I
    B = fI * b / I
    RGB = (np.clip(np.dstack([R,G,B]), 0., 1.) * 255.).astype(np.uint8)
    return RGB

def resid_rgb(resid1, resid2):
    ''' Converts WISE W1 and W2 residual images into an RGB image,
    with arcsinh stretch.
    '''
    S,Q = 30000,25
    alpha = 5.
    
    w1,w2 = resid1,resid2
    b = w1 / S
    r = w2 / S
    g = (r + b) / 2.

    I = (r+g+b)/3.
    fI = np.arcsinh(alpha * Q * I) / np.sqrt(Q)

    R = fI * r / I
    G = fI * g / I
    B = fI * b / I

    RGB = np.dstack([R,G,B])
    RGB = (np.clip((RGB + 1.) / 2., 0., 1.) * 255.99).astype(np.uint8)
    return RGB
    
def lbticks(wcs, xlo, ylo, lticks=None, bticks=None):
    ''' Add Galactic lat,long tick marks to a plot.
    '''
    ax = plt.axis()
    if lticks is None:
        lticks = [345, 350, 355, 0, 5, 10, 15]
    if bticks is None:
        bticks = [-15, -10, -5, 0, 5, 10, 15]
    xx = [wcs.radec2pixelxy(v, 0)[-2] - xlo for v in lticks]
    plt.xticks(xx, lticks)
    yy = [wcs.radec2pixelxy(0, v)[-1] - ylo for v in bticks]
    plt.yticks(yy, bticks)
    plt.xlabel('Galactic longitude $\ell$ (deg)')
    plt.ylabel('Galactic latitude $b$ (deg)')
    plt.axis(ax)

def halfsize(sourcefn, halffn):
    ''' Reads and image and WCS from the given source file,
    Bins down by a factor of 2, and writes to the given output file.
    '''
    im,hdr = fitsio.read(sourcefn, header=True)
    H,W = im.shape
    # make even size; smooth down
    if H % 2 == 1:
        im = im[:-1,:]
    if W % 2 == 1:
        im = im[:,:-1]

    # bin (excluding NaNs)
    q1 = im[::2,::2]
    q2 = im[1::2,::2]
    q3 = im[1::2,1::2]
    q4 = im[::2,1::2]

    f1 = np.isfinite(q1)
    f2 = np.isfinite(q2)
    f3 = np.isfinite(q3)
    f4 = np.isfinite(q4)

    im = (np.where(f1, q1, 0) +
          np.where(f2, q2, 0) +
          np.where(f3, q3, 0) +
          np.where(f4, q4, 0)) / np.maximum(1, f1+f2+f3+f4)
    #im = (im[::2,::2] + im[1::2,::2] + im[1::2,1::2] + im[::2,1::2])/4.
    im = im.astype(np.float32)
    # shrink WCS too
    wcs = Tan(sourcefn, 0)
    # include the even size clip; this may be a no-op
    H,W = im.shape
    wcs = wcs.get_subimage(0, 0, W, H)
    subwcs = wcs.scale(0.5)
    hdr = fitsio.FITSHDR()
    subwcs.add_to_header(hdr)
    dirnm = os.path.dirname(halffn)
    f,tmpfn = tempfile.mkstemp(suffix='.fits.tmp', dir=dirnm)
    os.close(f)
    # To avoid overwriting the (empty) temp file (and fitsio
    # printing "Removing existing file")
    os.unlink(tmpfn)
    fitsio.write(tmpfn, im, header=hdr, clobber=True)
    os.rename(tmpfn, halffn)
    print('Wrote', halffn)

if __name__ == '__main__':
    main()
