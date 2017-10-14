import numpy as np
from enlib import enmap
import orphics.tools.io as io

def gauss_ellipse(shape,wcs,ypos=0.,xpos=0.,amajor=1.,bminor=1.,angle=0.):

    pos = enmap.posmap(shape,wcs)
    y = pos[0,:,:]
    x = pos[1,:,:]

    r_square = ((x-xpos)*np.cos(angle)+(y-ypos)*np.sin(angle))**2./amajor**2.+((x-xpos)*np.sin(angle)-(y-ypos)*np.cos(angle))**2./bminor**2.
    
    modrmap = enmap.modrmap(shape,wcs)
    retmap = modrmap.copy()*0.
    retmap[r_square<1] = 1.

    return retmap



arcsec = 20.
px_arcsec = 0.5
a = 8.
b = np.random.uniform(0.,1.)*a
angle = np.random.uniform(0.,360.)*np.pi/180.
print a,b
shape,wcs = enmap.rect_geometry(arcsec/60.,px_arcsec/60.,proj="car",pol=False)
sec2rad = lambda x: x*np.pi/180./3600.
gell = gauss_ellipse(shape,wcs,amajor=sec2rad(a),bminor=sec2rad(b),angle=angle)

io.quickPlot2d(gell,"gal.png")

