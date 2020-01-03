from psi4 import core
from psi4.driver import constants
import numpy as np
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
# the one electron potential in the DFT machinery
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_quadrature_grid_vext( wfn , pme_grid_size , vext_tot , PME_grid_positions , interpolation_method ):   

    #*********  Adding to one electron potential in Hamiltonian **********
    # grid points for quadrature in V_potential
    x, y, z, w = wfn.V_potential().get_np_xyzw()

    #****** test **********
    # print grid points
    #for i in range(len(x)):
    #    print( i, x[i] , y[i] , z[i], w[i] )
    # external potential on quadrature grid points
    #V_ext = return_external_V( x , y , z )    
    # print external V
    #for i in range(len(V_ext)):
    #    print( i, V_ext[i] )
    #**********************

    #************ parse PME_grid , vext before interpolation ...
    #xmin = np.amin(x); xmax = np.amax(x); ymin = np.amin(x); ymax = np.amax(x); zmin = np.amin(z); zmax = np.amax(z)
    #PME_parse , vext_parse = trim_shift_PME_grid( PME_grid_positions , vext_tot , xmin , xmax , ymin , ymax , zmin , zmax )
    #PME_parse = np.array( PME_parse )
    #vext_parse = np.array( vext_parse )

    print(' interpolating quadature grid .... \n')
    #*********** now interpolate ....
    quadrature_grid=[]
    for i in range(len(x)):
        quadrature_grid.append( [ x[i] , y[i] , z[i] ] )
    quadrature_grid = np.array( quadrature_grid )

#    V_ext = interpolate.griddata( PME_parse , vext_parse , quadrature_grid , method='linear' )

    V_ext = interpolate_PME_grid( pme_grid_size , PME_grid_positions , vext_tot , quadrature_grid , interpolation_method )
    V_ext = list(V_ext)

#    print(' done interpolating quadature grid .... \n')
#    print( V_ext )
#    for i in range(len(V_ext)):
#       print( i , V_ext[i] )
#    sys.exit()

    flag = set_blocks_vext( wfn.V_potential() , V_ext )



#**********************************
# this subroutine is in charge of getting/passing gridpoints/vext for
# both the one electron potential, as well as the nuclear repulsion energy
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_nuclei_vext( wfn , pme_grid_size , vext_tot , PME_grid_positions, interpolation_method ):

    #********* Get atom coordinates and pass vext into molecule object
    #print("xyz nuclei")
    x=[]; y=[]; z=[]
    for i in range( wfn.molecule().natom() ):
        #print( i , scf_wfn.molecule().xyz(i) )
        xyz = wfn.molecule().xyz(i)
        x.append( xyz[0] )
        y.append( xyz[1] )
        z.append( xyz[2] )

#    nuclei_grid = np.array( nuclei_grid )

    #******* test
    # external potential on nuclei
    #V_ext = return_external_V( x , y , z )
    #V_ext = np.array(V_ext)


    #************ parse PME_grid , vext before interpolation ...
    #xmin = np.amin(x); xmax = np.amax(x); ymin = np.amin(x); ymax = np.amax(x); zmin = np.amin(z); zmax = np.amax(z)
    # if only one nuclei, above range won't work...
    #thresh = 5.0
    #if ( xmax - xmin ) < thresh:
    #   xmax = xmax + thresh/2
    #   xmin = xmin - thresh/2
    #if ( ymax - ymin ) < thresh:
    #   ymax = ymax + thresh/2
    #   ymin = ymin - thresh/2
    #if ( zmax - zmin ) < thresh:
    #   zmax = zmax + thresh/2
    #   zmin = zmin - thresh/2        
    #PME_parse , vext_parse = trim_shift_PME_grid( PME_grid_positions , vext_tot , xmin , xmax , ymin , ymax , zmin , zmax )
    #PME_parse = np.array( PME_parse )
    #vext_parse = np.array( vext_parse )

    print(' interpolating nuclei grid .... \n')
    #*********** now interpolate ....
    nuclei_grid=[]
    for i in range(len(x)):
        nuclei_grid.append( [ x[i] , y[i] , z[i] ] )
    nuclei_grid = np.array( nuclei_grid )

    #V_ext = interpolate.griddata( PME_parse , vext_parse , nuclei_grid , method='linear' )

    V_ext = interpolate_PME_grid( pme_grid_size , PME_grid_positions , vext_tot , nuclei_grid , interpolation_method )

    print(' done interpolating nuclei grid .... \n')

    # convert to Psi4 vector object
    vext_vector = core.Vector.from_array(V_ext)

    #print( 'Vext in python')
    #print( V_ext )

    flag = wfn.molecule().set_vext(vext_vector)


#*******************************************
# this is a wrapper that calls scipy routines to
# interpolate from vext values evaluated at PME grid points.
# input "interpolation_method" should be set to either :
#       "interpn"  :: calls  scipy.interpolate.interpn()
#       "griddata" :: calls  scipy.interpolate.griddata()
#*******************************************
def interpolate_PME_grid( pme_grid_size , PME_grid_positions , vext_tot , interpolate_coords , interpolation_method ):

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

    return output_interpolation



#********************************************
# call this as an intermediate method for interpolating
# the external potential on the PME grid to the DFT quadrature grid
#
# this method trims the list of PME grid points and external potential
# to only those points which bound the QM quadrature region as specified by
# xmin < x < max, ymin < y < ymax , zmin < z < zmax
# input box_a , box_b , box_c are the periodic box vectors
#********************************************
def trim_shift_PME_grid( PME_grid_positions , vext_tot , xmin , xmax , ymin , ymax , zmin, zmax ):
    vext_parse = []
    PME_grid_parse = []

    # buffer pct
    buffer_pct = 0.3
    # add 10% buffer to all boundaries
    delta_x = xmax - xmin
    delta_y = ymax - ymin
    delta_z = zmax - zmin

    # padded boundaries
    xmin_pad = xmin - delta_x * buffer_pct
    xmax_pad = xmax + delta_x * buffer_pct
    ymin_pad = ymin - delta_y * buffer_pct
    ymax_pad = ymax + delta_y * buffer_pct
    zmin_pad = zmin - delta_z * buffer_pct
    zmax_pad = zmax + delta_z * buffer_pct

    # loop over PME grid
    for i in range(len(PME_grid_positions)):
        x_grid = PME_grid_positions[i][0]
        y_grid = PME_grid_positions[i][1]
        z_grid = PME_grid_positions[i][2]

        # see if grid position is within bounds
        if ( x_grid > xmin_pad ) and ( x_grid < xmax_pad ) and ( y_grid > ymin_pad ) and ( y_grid < ymax_pad ) and ( z_grid > zmin_pad ) and ( z_grid < zmax_pad ) :
            # add grid point
            vext_parse.append( vext_tot[i] )
            PME_grid_parse.append( [ PME_grid_positions[i][0] , PME_grid_positions[i][1] , PME_grid_positions[i][2] ] )


    return PME_grid_parse , vext_parse

