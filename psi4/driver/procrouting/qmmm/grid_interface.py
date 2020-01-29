from psi4 import core
from psi4.driver import constants
import numpy as np
from math import *
from scipy import interpolate
import copy
import sys

#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr


#******************************************************
# --------- this is used for testing only -------------
#   input a list of MMenv = [ [ q1 , x1 , y1 , z1 ] , [ [ q2 , x2 , y2 , z2 ] , ... ]
#   compute and set external potential on electronic quadrature grid or nuclear potential
#
#   input units:
#        assume coordinates in MMenv are input in Angstrom, and convert internally to Bohr
#
#******************************************************
def set_MMfield_internal( MMsys , wfn , flag_nuclear=False  ):

    # local copy to keep unit conversion internal
    MMenv = copy.deepcopy(MMsys)

    # conversion Angstrom to Bohr
    for i in range(len(MMenv)):
        for j in range(1,4):
            MMenv[i][j] = MMenv[i][j]  / constants.bohr2angstroms
    #*************************************************

    x=[]; y=[]; z=[]
    if flag_nuclear :
        #************ Adding MM external field to nuclear potential
        for i in range( wfn.molecule().natom() ):
            xyz = wfn.molecule().xyz(i)
            x.append( xyz[0] )
            y.append( xyz[1] )
            z.append( xyz[2] )

    else:
        #*********  Adding to one electron potential in Hamiltonian **********
        # grid points for quadrature in V_potential
        x, y, z, w = wfn.V_potential().get_np_xyzw()


    # compute Coulomb potential on required xyz points
    V_ext = [0.0] * len(x)
    # loop over grid points
    for i in range(len(V_ext)):
        # loop over MM sites
        for j in range(len(MMenv)):
            q = MMenv[j][0]
            r = ( (x[i] - MMenv[j][1])**2 + (y[i] - MMenv[j][2])**2 + (z[i] - MMenv[j][3])**2 )**(0.5)
            V_ext[i] += q/r

    V_ext = np.array(V_ext)
    if flag_nuclear :
        # add to external nuclear potential
        # convert to Psi4 vector object
        vext_vector = core.Vector.from_array(V_ext)
        flag = wfn.molecule().set_vext(vext_vector)    
    else :
        # add to quadrature
        V_ext = list(V_ext)
        flag = set_blocks_vext( wfn.V_potential() , V_ext )
   
    return 1



# ** here we access block data to set vext on grid points
#  see method _core_vbase_get_np_xyzw in ~/driver/p4util/python_helpers.py
#  to see access of Vpot.get_np_xyzw(), and we set vext points the same way
def set_blocks_vext( Vpot, Vext ):

    index_start=0
    # Loop over every block in the potenital
    for b in range(Vpot.nblocks()):

        # Obtain the block
        block = Vpot.get_block(b)
        # number of grid points on this block
        npoints_ = block.npoints()
        # get vext grid values for this block

        vext_block = np.array(Vext[index_start:index_start+npoints_])

        # convert to Psi4 vector object
        vext_vector = core.Vector.from_array(vext_block)

        #print("setting vext for block", b )
        # set vext for this block
        block.set_vext( vext_vector )
        # increment for next block
        index_start = index_start + npoints_
    print( 'done setting vext')
    # test
    #block = Vpot.get_block(0)
    #print( 'Vext' , block.vext() )

    return 1


#**********************************
# this subroutine is in charge of getting/passing gridpoints/vext for
# the one electron potential in the DFT machinery.  Input **kwargs determine choice of method:
#
# Method 1:  Interpolate in real space using PME_grid_positions (limitation, no PBC, only cubic box ...)
#
# Method 2:  Project quadrature grid to PME grid, and interpolate on PME grid
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_quadrature_grid_vext( wfn , pme_grid_size , vext_tot , interpolation_method, **kwargs ):   

    #*********  Adding to one electron potential in Hamiltonian **********
    # grid points for quadrature in V_potential
    x, y, z, w = wfn.V_potential().get_np_xyzw()
    quadrature_grid=[]
    for i in range(len(x)):
        quadrature_grid.append( [ x[i] , y[i] , z[i] ] )
    quadrature_grid = np.array( quadrature_grid )

    print(' interpolating quadature grid .... \n')

    # get **kwargs to determine interpolation method
    if 'pmegrid_xyz' in kwargs:
        # Method 1: real-space interpolation
        PME_grid_positions = kwargs['pmegrid_xyz']
        #*********** now interpolate ....
        V_ext = interpolate_PME_grid( pme_grid_size , vext_tot , quadrature_grid , interpolation_method,  pmegrid_xyz=PME_grid_positions )

    else:
        # Method 2: interpolation on PME grid
        box = kwargs['box']
        inverse_box = calculate_inverse_box_vectors( box )
        # project to PME grid
        quadrature_grid_project = project_to_PME_grid( quadrature_grid , inverse_box , pme_grid_size )
        #*********** now interpolate ....
        V_ext = interpolate_PME_grid( pme_grid_size , vext_tot , quadrature_grid_project , interpolation_method )
 

    V_ext = list(V_ext)

    flag = set_blocks_vext( wfn.V_potential() , V_ext )



#**********************************
# this subroutine is in charge of getting/passing gridpoints/vext for
# the nuclear repulsion energy   Input **kwargs determine choice of method:
#
# Method 1:  Interpolate in real space using PME_grid_positions (limitation, no PBC, only cubic box ...)
#
# Method 2:  Project quadrature grid to PME grid, and interpolate on PME grid
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_nuclei_vext( wfn , pme_grid_size , vext_tot , interpolation_method, **kwargs ):

    #********* Get atom coordinates and pass vext into molecule object
    #print("xyz nuclei")
    x=[]; y=[]; z=[]
    for i in range( wfn.molecule().natom() ):
        xyz = wfn.molecule().xyz(i)
        x.append( xyz[0] )
        y.append( xyz[1] )
        z.append( xyz[2] )

        print( i , xyz )

    sys.exit()

    nuclei_grid=[]
    for i in range(len(x)):
        nuclei_grid.append( [ x[i] , y[i] , z[i] ] )
    nuclei_grid = np.array( nuclei_grid )

    print(' interpolating nuclei grid .... \n')

    # get **kwargs to determine interpolation method
    if 'pmegrid_xyz' in kwargs:
        # Method 1: real-space interpolation
        PME_grid_positions = kwargs['pmegrid_xyz']
        #*********** now interpolate ....
        V_ext = interpolate_PME_grid( pme_grid_size , vext_tot , nuclei_grid , interpolation_method, pmegrid_xyz=PME_grid_positions )

    else:
        # Method 2: interpolation on PME grid
        box = kwargs['box']
        inverse_box = calculate_inverse_box_vectors( box )
        # project to PME grid
        nuclei_grid_project = project_to_PME_grid( nuclei_grid , inverse_box , pme_grid_size )
        #*********** now interpolate ....
        V_ext = interpolate_PME_grid( pme_grid_size , vext_tot , nuclei_grid_project , interpolation_method )


    print(' done interpolating nuclei grid .... \n')

    # convert to Psi4 vector object
    vext_vector = core.Vector.from_array(V_ext)

    flag = wfn.molecule().set_vext(vext_vector)


#*******************************************
# this is a wrapper that calls scipy routines to
# interpolate from vext values evaluated at PME grid points.
# input "interpolation_method" should be set to either :
#       "interpn"  :: calls  scipy.interpolate.interpn()
#       "griddata" :: calls  scipy.interpolate.griddata()
#
#  Method 1: interpolate in real space 
#  Method 2: intepolate on PME grid (with projected quadrature points)
#    which method to use is based on what's in **kwargs
#*******************************************
def interpolate_PME_grid( pme_grid_size , vext_tot , interpolate_coords , interpolation_method, pad_grid=True, **kwargs ):

    # See if we're interpolating in real-space
    if 'pmegrid_xyz' in kwargs:
       PME_grid_positions = kwargs['pmegrid_xyz']

       # decide what interpolation methods we're using:
       if ( interpolation_method == "interpn" ):
          # using scipy.interpolate.interpn()
          # need to form new data structures consistent with scipy.interpolate.interpn() input ...
       
          # ********* this needs to be modified if not cubic grid! *****************
          xdim = PME_grid_positions[:,2]
          xdim = xdim[0:pme_grid_size]
          grid = (xdim , xdim , xdim )     #!!!!!!!!!!!!! cubic grid only !!!!!!!!!!!!!!#

          vext_tot_3d = np.reshape( vext_tot , ( pme_grid_size , pme_grid_size , pme_grid_size ) )
          output_interpolation = interpolate.interpn( grid, vext_tot_3d , interpolate_coords , method='linear' )

       elif ( interpolation_method == "griddata" ):
          output_interpolation = interpolate.griddata( PME_grid_positions , vext_tot , interpolate_coords , method='linear' )
       else:
          raise ValidationError(" interpolation_method must be set to either 'interpn' or 'griddata' ... ")

    else:
       # interpolating on PME grid, quadrature coordinates should already be projected
       if pad_grid:
           # pad grid
           vext_tot_pad = pad_vext_PME_grid( vext_tot , pme_grid_size )
           xdim = np.array( [ i for i in range(-1,pme_grid_size+1) ] )
           grid = (xdim , xdim , xdim )
           vext_tot_3d = np.reshape( vext_tot_pad , ( pme_grid_size+2 , pme_grid_size+2 , pme_grid_size+2 ) )
       else:
           # not padding grid ...
           xdim = np.array( [ i for i in range(pme_grid_size) ] )
           grid = (xdim , xdim , xdim )
           vext_tot_3d = np.reshape( vext_tot , ( pme_grid_size , pme_grid_size , pme_grid_size ) )
   
       output_interpolation = interpolate.interpn( grid, vext_tot_3d , interpolate_coords , method='linear' )

    return output_interpolation




#***********************************
# this method pads the boundaries of vext_tot for interpolation.
# vext_tot satisfies the PBC of the system, but this is done so that we can avoid PBC in the interpolation.
# We might have an interpolation point at the edge of the grid, if we're
# using linear interpolation then we just need one more point to pad the edge
#    so if vext_tot     is dimension pme_grid_size x pme_grid_size x pme_grid_size,
#    then  vext_tot_pad is dimension (pme_grid_size + 2) x (pme_grid_size + 2) x (pme_grid_size + 2)
#    where the padding on either side of the grid is done by using the PBC
def pad_vext_PME_grid( vext_tot , pme_grid_size ):
    vext_tot_pad = []
    for i in range(-1,pme_grid_size+1):
        i_pbc = ( i + pme_grid_size ) % pme_grid_size  # pbc to find grid point ...
        for j in range(-1,pme_grid_size+1):
            j_pbc = ( j + pme_grid_size ) % pme_grid_size  # pbc to find grid point ...
            for k in range(-1,pme_grid_size+1):
                k_pbc = ( k + pme_grid_size ) % pme_grid_size  # pbc to find grid point ...    
                index = i_pbc*pme_grid_size*pme_grid_size + j_pbc*pme_grid_size + k_pbc
                vext_tot_pad.append( vext_tot[index] )

    vext_tot_pad = np.array( vext_tot_pad )
    
    return vext_tot_pad


#*********************************
# project real space points to PME grid
# this algorithm is identical to that used in method
# 'pme_update_grid_index_and_fraction' in OpenMM source code,
# ReferencePME.cpp.  See comments in ReferencePME.cpp
# about how algorithm works ...
#*********************************
def project_to_PME_grid( real_grid_points , inverse_box , pme_grid_size ):
    
    #*************** naive code with loops , let's keep this for readability....
    #scaled_grid_points=[]
    #for i in range(real_grid_points.shape[0]):
    #    xyz_scaled=[]
    #    for d in range(3):
    #        t = real_grid_points[i][0] * inverse_box[0][d] + real_grid_points[i][1] * inverse_box[1][d] + real_grid_points[i][2] * inverse_box[2][d]
    #        t = ( t - floor(t) ) * pme_grid_size
    #        ti = int(t)
    #        fraction = t - ti
    #        xyz_scaled.append( ti % pme_grid_size + fraction )
    #    scaled_grid_points.append( xyz_scaled )
    #scaled_grid_points = np.array( scaled_grid_points )

    #************ efficient numpy code
    scaled_grid_points = np.matmul( real_grid_points , inverse_box )
    scaled_grid_points = ( scaled_grid_points - np.floor( scaled_grid_points ) ) * pme_grid_size
    scaled_grid_points = np.mod( scaled_grid_points.astype(int) , pme_grid_size ) + ( scaled_grid_points - scaled_grid_points.astype(int) )

    return scaled_grid_points


#**********************************
# this is identical to OpenMM method in ReferencePME.cpp
# void invert_box_vectors(const Vec3 boxVectors[3], Vec3 recipBoxVectors[3])
# assumes triclinic box has specific form.
#**********************************
def calculate_inverse_box_vectors( boxVectors ):
    determinant = boxVectors[0][0] * boxVectors[1][1] * boxVectors[2][2]
    scale = 1.0/determinant
    recipBoxVectors=[]
    recipBoxVectors.append( [ boxVectors[1][1]*boxVectors[2][2], 0, 0 ]  )
    recipBoxVectors.append( [ -boxVectors[1][0]*boxVectors[2][2], boxVectors[0][0]*boxVectors[2][2], 0 ] )
    recipBoxVectors.append( [ boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0], -boxVectors[0][0]*boxVectors[2][1], boxVectors[0][0]*boxVectors[1][1] ] )
    recipBoxVectors = np.array(recipBoxVectors)*scale

    return recipBoxVectors




