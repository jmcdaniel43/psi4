from psi4 import core
import numpy as np

# this function returns an external potential computed at input grid points, as defined by vectors grid_x, grid_y, grid_z
# input MM_sites contain lists of [ q , x , y , z ] for all external atoms that contribute to external potential
# returns a list V_ext with same dimensions as grid_x, etc.
def return_external_V( grid_x , grid_y , grid_z , MM_sites ):

    V_ext = [0] * len(grid_x)
    # conversion Angstrom^-1 to Bohr^-1
    conv = 0.529177
    # loop over grid points
    for i in range(len(V_ext)):
        # loop over MM sites
        for j in range(len(MM_sites)):
            q = MM_sites[j][0]
            r = ( (grid_x[i] - MM_sites[j][1])**2 + (grid_y[i] - MM_sites[j][2])**2 + (grid_z[i] - MM_sites[j][3])**2 )**(0.5)
            V_ext[i] += q/r * conv

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


#def pass_quadrature_grid( sup , basis ):
def pass_quadrature_grid( Vpot ):   
#    Vpot = core.VBase.build(basis, sup, "RV")
#    Vpot.initialize()
    x, y, z, w = Vpot.get_np_xyzw()

    # print grid points
#    for i in range(len(x)):
#        print( i, x[i] , y[i] , z[i], w[i] )

    print( "applying Q=2 at (3.0, 0.0, 0.0) " )
    MM_charge=[ [ 2 , 3.0  , 0.0  , 0.0 ] ]

    V_ext = return_external_V( x , y , z , MM_charge )

    # print external V
    #for i in range(len(V_ext)):
    #    print( i, V_ext[i] )

    flag = set_blocks_vext( Vpot , V_ext )

