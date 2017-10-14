from __future__ import print_function 
import numpy as np
from enlib import enmap
import orphics.tools.io as io



class EllipseSim(object):

    def __init__(self,shape,wcs):
        pos = enmap.posmap(shape,wcs)
        self.y = pos[0,:,:]
        self.x = pos[1,:,:]
        self.modrmap = enmap.modrmap(shape,wcs)
        self.shape = shape
        self.wcs = wcs

    def get_sims(self,num_sims,arange,brange,angle_range,seed=None):

        np.random.seed(seed)
        amajors = np.random.uniform(arange[0],arange[1],num_sims)
        bminors = np.random.uniform(brange[0],brange[1],num_sims)
        angles = np.random.uniform(angle_range[0],angle_range[1],num_sims)

        ellipse_mat = self.ellipse(amajor=amajors,bminor=bminors,angle=angles)

        
        return ellipse_mat,amajors,bminors,angles
           
    def ellipse(self,ypos=0.,xpos=0.,amajor=1.,bminor=1.,angle=0.):
        amajor = np.asarray(amajor)
        bminor = np.asarray(bminor)
        angle = np.asarray(angle)

        assert amajor.size == bminor.size == angle.size
        N = amajor.size

        amajor = np.reshape(amajor,(N,1,1))
        bminor = np.reshape(bminor,(N,1,1))
        angle = np.reshape(angle,(N,1,1))

        
        
        xmap = np.resize(self.x,(N,self.shape[0],self.shape[1]))
        ymap = np.resize(self.y,(N,self.shape[0],self.shape[1]))

        
        r_square = ((xmap-xpos)*np.cos(angle)+(ymap-ypos)*np.sin(angle))**2./amajor**2.+((xmap-xpos)*np.sin(angle)-(ymap-ypos)*np.cos(angle))**2./bminor**2.

        retmap = np.zeros((N,self.shape[0],self.shape[1])).astype(np.int32)
        retmap[r_square<1] = 1

        return retmap


def test():

    arcsec = 20.
    px_arcsec = 0.5
    shape,wcs = enmap.rect_geometry(arcsec/60.,px_arcsec/60.,proj="car",pol=False)
    sec2rad = lambda x: x*np.pi/180./3600.

    N = 10000
    arange = [sec2rad(3),sec2rad(9)] 
    brange = [sec2rad(3),sec2rad(9)]
    angle_range = [0,2.*np.pi]

    esim = EllipseSim(shape,wcs)
    mat = esim.get_sims(N,arange,brange,angle_range)
    print ("Megabytes : ", mat.nbytes/1024./1024.)
    
    print(mat.shape)
    io.quickPlot2d(mat[0,:,:],"gal0.png")
    io.quickPlot2d(mat[1,:,:],"gal1.png")
    

def main():

    arcsec = 20.
    px_arcsec = 0.5
    a = 8.
    b = np.random.uniform(0.,1.)*a
    angle = np.random.uniform(0.,360.)*np.pi/180.
    print (a,b)
    shape,wcs = enmap.rect_geometry(arcsec/60.,px_arcsec/60.,proj="car",pol=False)
    sec2rad = lambda x: x*np.pi/180./3600.
    gell = gauss_ellipse(shape,wcs,amajor=sec2rad(a),bminor=sec2rad(b),angle=angle)

    io.quickPlot2d(gell,"gal.png")

