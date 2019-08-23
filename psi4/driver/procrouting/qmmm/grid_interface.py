from psi4 import core
import numpy as np


#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr


#*******************************************
# this function returns an external potential computed at input grid points, as defined by vectors grid_x, grid_y, grid_z
# returns a list V_ext with same dimensions as grid_x, etc.
def return_external_V( grid_x , grid_y , grid_z ):

    #************* this is for testing... *************
    print( "applying Q=2 at (3.0, 0.0, 0.0) Angstrom " )
    MM_sites=[ [ 2 , 3.0  , 0.0  , 0.0 ] ]

    # conversion Angstrom to Bohr
    conv = 1.88973
    for i in range(len(MM_sites)):
        for j in range(1,3):
            MM_sites[i][j] = MM_sites[i][j] * conv
    #*************************************************


    V_ext = [0] * len(grid_x)
    # loop over grid points
    for i in range(len(V_ext)):
        # loop over MM sites
        for j in range(len(MM_sites)):
            q = MM_sites[j][0]
            r = ( (grid_x[i] - MM_sites[j][1])**2 + (grid_y[i] - MM_sites[j][2])**2 + (grid_z[i] - MM_sites[j][3])**2 )**(0.5)
            V_ext[i] += q/r

    return V_ext



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

        print("setting vext for block", b )
        # set vext for this block
        block.set_vext( vext_vector )
        # increment for next block
        index_start = index_start + npoints_
    print( 'done setting vext')
    # test
    block = Vpot.get_block(0)
    print( 'Vext' , block.vext() )
    return 1


#**********************************
# this subroutine is in charge of getting/passing gridpoints/vext for
# the one electron potential in the DFT machinery
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_quadrature_grid_vext( wfn ):   

    #*********  Adding to one electron potential in Hamiltonian **********
    # grid points for quadrature in V_potential
    x, y, z, w = wfn.V_potential().get_np_xyzw()

    # print grid points
    for i in range(len(x)):
        print( i, x[i] , y[i] , z[i], w[i] )

    # external potential on quadrature grid points
    V_ext = return_external_V( x , y , z )

    # print external V
    #for i in range(len(V_ext)):
    #    print( i, V_ext[i] )

    flag = set_blocks_vext( wfn.V_potential() , V_ext )



#**********************************
# this subroutine is in charge of getting/passing gridpoints/vext for
# both the one electron potential, as well as the nuclear repulsion energy
#
#  Units:  Work in Bohr within this module to avoid unit errors.
#  scf_wfn.molecule().xyz(i) returns atomic coordinates in bohr
#  scf_wfn.V_potential().get_np_xyzw()  returns grid coordinates in bohr
#**********************************
def pass_nuclei_vext( wfn ):

    #********* Get atom coordinates and pass vext into molecule object
    #print("xyz nuclei")
    x=[]
    y=[]
    z=[]
    for i in range( wfn.molecule().natom() ):
        #print( i , scf_wfn.molecule().xyz(i) )
        xyz = wfn.molecule().xyz(i)
        x.append( xyz[0] )
        y.append( xyz[1] )
        z.append( xyz[2] )
  
    # external potential on nuclei
    V_ext = return_external_V( x , y , z )

    V_ext = np.array(V_ext)
    # convert to Psi4 vector object
    vext_vector = core.Vector.from_array(V_ext)

    print( 'Vext in python')
    flag = wfn.molecule().set_vext(vext_vector)
