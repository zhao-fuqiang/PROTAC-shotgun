#!/usr/bin/env python3

import os,sys
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sortedcontainers
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate, convolve
from scipy.spatial.transform import Rotation as R
import mrcfile
from sklearn.linear_model import LinearRegression

import random

from config import config

def normalize(arr):
    #max_val = 2**16 -1
    #f = lambda x: x * max_val
    #arr_normed = f(arr)
    #return arr_normed
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 -1
    exploded = np.zeros(np.concatenate([size,shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x,y,z = indices
    x[1::2,:,:] += 1
    y[:,1::2,:] += 1
    z[:,:,1::2] += 1
    return x,y,z

def plot_cube(cube, lim, outname, angle=320):
    cube = normalize(cube)
    cutoff = np.percentile(cube,80)

    facecolors = plt.cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    #facecolors = explode(facecolors)
    #print(np.min(cube, axis=None), np.max(cube, axis=None))
    mid = int(lim/2)
    #img = plt.imshow(cube[mid,:,:], cmap=plt.get_cmap('viridis_r'),vmin=0,vmax=1)
    #plt.colorbar(img)
    #plt.show()

    filled = facecolors[:,:,:,-1] >= cutoff

    x,y,z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(30,angle)
    ax.set_xlim(right=lim)
    ax.set_ylim(top=lim)
    ax.set_zlim(top=lim)
    ax.voxels(x,y,z,filled, facecolors=facecolors, shade=True, alpha=0.2)
    plt.show()
    plt.savefig(fig_folder + '/' + '%s_tmp.png' % outname)

    return 0
##=====================END PLOTTING FUNCTIONS==============

def create_protein_grid(proteinpdb , dist , angs_per_voxel, resID_pocket_receptor, config, opencl = True):

    ## Creating grid based on protein dimension--ligand grid must contain the same dimensions as protein grid!
    print('protein')
    X, Y, Z, atom_radius, p_centroid, atom_name, resname, chainID, resID, pocket_com, pocket_plane_params, pocket_atom_coors = parse_pdb(proteinpdb, config.resID_pocket_receptor)
    xyz_dim = np.ceil((max(np.ptp(X), np.ptp(Y), np.ptp(Z)))) ## in Angstroms
    buffer = 30*1/angs_per_voxel ## 30A buffer
    grid_dim = int((xyz_dim / angs_per_voxel)+ 2 * buffer)
    powerof2 = np.ceil(np.log2(grid_dim))
    new_grid_dim = int(2**powerof2) ## FFT wants a grid size to be a power of 2...

    offset = float("{:.3f}".format((new_grid_dim*angs_per_voxel/2)))

    print("============\nProtein in %2d A cube using %0.2f A per voxel. This requires %3d cubic array. The offset is %0.2f.\n============"
          % (xyz_dim, angs_per_voxel, new_grid_dim, offset))

    # ss: Here we create an function alias so we can switch which one we want to use (see below when called)
    gaussian_maps_fnc = gaussian_maps_cl if opencl else gaussian_maps
    protein_grid = gaussian_maps_fnc(new_grid_dim, angs_per_voxel, dist, X, Y, Z, atom_radius, atom_name, resname, chainID, resID, 'protein', resID_pocket_receptor, config)

    """"## Plot cross section or debugging
    mid = int((new_grid_dim/2))
    img = plt.imshow(ligand_grid[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()

    ## Plot cross section or debugging
    img = plt.imshow(ligand_grid_rot[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()
    """

    return protein_grid , p_centroid, offset, new_grid_dim, pocket_com, pocket_plane_params, pocket_atom_coors

def create_ligand_grid(ligandpdb, new_grid_dim, angs_per_voxel, dist, resID_pocket_ligand, config, opencl=True):
    ##=====LIGAND======##
    # this should be an input PDB #TODO: SDFs for ligands...
    print('ligand')
    X, Y, Z, atom_radius, l_centroid, atom_name, resname, chainID, resID, pocket_com, pocket_plane_params, pocket_atom_coors = parse_pdb(ligandpdb, config.resID_pocket_ligand)

    gaussian_maps_fnc = gaussian_maps_cl if opencl else gaussian_maps
    ligand_grid = gaussian_maps_fnc(new_grid_dim, angs_per_voxel, dist, X, Y, Z, atom_radius, atom_name, resname, chainID, resID, 'ligand', resID_pocket_ligand, config)

    return ligand_grid, l_centroid, pocket_com, pocket_plane_params, pocket_atom_coors

def parse_pdb(inputpdb, resID_pocket):
    X, Y, Z, atom_radius, atom_name, resname, chainID, resID = [], [], [], [], [], [], [], []
    pocket_com = np.array([0, 0, 0])
    pocket_plane_params = np.array([0.0, 0.0, -1.0, 0.0])
    pocket_atom_n = 0
    pocket_atom_coors = []
    with open(inputpdb, 'r') as f:
        for line in f.readlines():
            if line[0:6] == "ATOM  " or line[0:6] == "HETATM":
                X.append(float(line[30:38].strip()))
                Y.append(float(line[38:46].strip()))
                Z.append(float(line[46:54].strip()))
                atom_name.append(str(line[11:17].strip()))
                resname.append(str(line[17:20].strip()))
                chainID.append(str(line[20:22].strip()))
                resID.append(float(line[22:30].strip()))

                if atom_name[-1] == 'CA':
                    if resID[-1] in resID_pocket:
                        pocket_atom_n += 1
                        pocket_com[0] += X[-1]
                        pocket_com[1] += Y[-1]
                        pocket_com[2] += Z[-1]
                        pocket_atom_coors.append([X[-1], Y[-1], Z[-1]])

                atom_type = line.split()[-1]  ## atomic radii values from Covalent Radius Wikipedia page
                if atom_type == "C":
                    atom_radius.append(float(0.76*2))  # sp3
                elif atom_type == "N":
                    atom_radius.append(float(0.71*2))
                elif atom_type == "O":
                    atom_radius.append(float(0.66*2))
                elif atom_type == "H":
                    atom_radius.append(float(0.31*2))
                elif atom_type == "CL":
                    atom_radius.append(float(1.02*2))
                else:
                    atom_radius.append(float(1.00*2))  # For now...
    f.close()

    pocket_atom_coors = np.array(pocket_atom_coors)
    reg = LinearRegression().fit(pocket_atom_coors[0:2,0:2], pocket_atom_coors[0:2, 2])
    pocket_plane_params[0] = reg.coef_[0]
    pocket_plane_params[1] = reg.coef_[1]
    pocket_plane_params[3] = reg.intercept_
    pocket_com = pocket_com / pocket_atom_n
    print('pocket_atom_n:')
    print(pocket_atom_n)
    print('pocket_com:')
    print(pocket_com)
    print('pocket_plane_params:')
    print(pocket_plane_params)

    # set XYZ grid at (0,0,0)
    centroid = ((np.max(X)+np.min(X))/2,(np.max(Y)+np.min(Y))/2,(np.max(Z)+np.min(Z))/2)
    print('PDB centroid:')
    print(centroid)
    X[:] = map(lambda x: x-centroid[0], X)
    Y[:] = map(lambda y: y-centroid[1], Y)
    Z[:] = map(lambda z: z-centroid[2], Z)

    vec_center_to_pocket = pocket_com - centroid
    cos_theta = np.dot(pocket_plane_params[0:3], vec_center_to_pocket)
    #cos_theta /= np.linalg.norm(pocket_plane_params[0:3]) * np.linalg.norm(vec_center_to_pocket)
    if cos_theta < 0:
        pocket_plane_params = -pocket_plane_params

    return X,Y,Z,atom_radius, centroid, atom_name, resname, chainID, resID, pocket_com, pocket_plane_params, pocket_atom_coors

def dump_transformed_input_pdbs(inputpdb, outputpdb, offset, centroid):
    with open(inputpdb, 'r') as f:
        with open(outputpdb, 'w') as g:
            for line in f.readlines():
                if line[0:6] == "ATOM  " or line[0:6] == "HETATM":
                    newX = float(line[31:38].strip()) - float(centroid[0]) + offset
                    newY = float(line[39:46].strip()) - float(centroid[1]) + offset
                    newZ = float(line[47:54].strip()) - float(centroid[2]) + offset
                    newline = str(line[0:30] + "%8.3f%8.3f%8.3f"%(newX, newY, newZ) + line[54:] )
                    g.write(newline)
        g.close()
    f.close()

    return 0

def dump_transformed_input_pdbs2(inputpdb, outpdb, offset, p_centroid, l_offset, l_centroid, euler_angles):
    #print('p_centroid:')
    #print(p_centroid)

    r = R.from_euler('zyx', euler_angles, degrees=True)
    #print('roattion:')
    #print(r.as_euler)

    with open(inputpdb, 'r') as f:
        with open(outpdb, 'w') as g:
            for line in f.readlines():
                if line[0:6] == "ATOM  " or line[0:6] == "HETATM":
                    newX = float(line[31:38].strip()) - float(l_centroid[0])
                    newY = float(line[39:46].strip()) - float(l_centroid[1])
                    newZ = float(line[47:54].strip()) - float(l_centroid[2])

                    newX, newY, newZ = r.apply([newX, newY, newZ])

                    newX = newX + float(l_offset[0])
                    newY = newY + float(l_offset[1])
                    newZ = newZ + float(l_offset[2])
                    newline = str(line[0:30] + "%8.3f%8.3f%8.3f"%(newX, newY, newZ) + line[54:] )
                    g.write(newline)
        g.close()
    f.close()

    return 0

def dump_transformed_grid(input_grid):

    return transformed_grid

def gaussian_maps_cl(num_voxels, angs_per_voxel, dist, X, Y, Z, atom_radius, atom_name, resname, chainID, resID, codeword, resID_pocket, config):
    # ss: Here, import is in the function because if you don't want/have pyopencl you can still use CPU mode
    import pyopencl as cl
    print("Creating Gaussian distance-based distribution map for %s" % (codeword))

    ## Set protein "inside" to -15 and ligand to 2 (mimicking a paper that did this and just trying to replicate their stuff for now)
    surface_intensity = config.surface_intensity
    pocket_intensity = config.pocket_intensity
    if codeword == "protein":
        inner_intensity = config.protein_inner_intensity
    elif codeword == "ligand":
        inner_intensity = config.ligand_inner_intensity

    # ss: We need to transfer the data tensors to the GPU.
    # The numpy arrays must contain opencl types!
    xyzr = np.array([X, Y, Z, atom_radius, resID], dtype=cl.cltypes.float)
    grid = np.zeros([num_voxels,num_voxels,num_voxels], dtype=cl.cltypes.float)
    params = np.array([angs_per_voxel, dist, inner_intensity, surface_intensity, pocket_intensity, len(resID_pocket)], dtype=cl.cltypes.float)
    atom_residue = np.array([resID, resID], dtype=cl.cltypes.float)
    resID_pocket_array = np.array(resID_pocket, dtype=cl.cltypes.int)
    #print(xyzr)
    #print(grid)
    #print(params)
    #print(atom_residue)

    # ss: It is not really necessary to create multiple contexts here for every single call, but since it's only two times in the script
    platforms = cl.get_platforms()
    #print('platforms:')
    #print(platforms)
    devices = platforms[0].get_devices()
    #print('devices:')
    #print(devices)
    #ctx = cl.create_some_context()
    ctx = cl.Context([devices[0]])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    clsource = open("gaussian_maps.cl", "r").read()
    prg = cl.Program(ctx, clsource).build()


    img_xyzr = cl.Image(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        xyzr.shape[1::-1],
        hostbuf=xyzr)

    buf_params = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=params)

    img_atom_residue = cl.Image(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        atom_residue.shape[1::-1],
        hostbuf=atom_residue)

    buf_resID_pocket = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=resID_pocket_array)

    img_grid = cl.Image(
        ctx, mf.WRITE_ONLY,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid.shape)

    img_pocket_grid = cl.Image(
        ctx, mf.WRITE_ONLY,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid.shape)

    kernel = prg.gaussian_map
    # ss: This is equivalent to enqueueNDRangeKernel:
    # Each pixel is processed independently, in parallel (or at least part of it in batches depending on GPU hardware and image size)
    kernel(queue, grid.shape, None, img_xyzr, buf_params, buf_resID_pocket, img_atom_residue, img_grid)

    grid_gpu = np.empty_like(grid, dtype=cl.cltypes.float) # ss: Here, we retreive the results from the GPU
    cl.enqueue_copy(queue, grid_gpu, img_grid, origin=(0, 0, 0), region=grid_gpu.shape)

    grid_cp = np.zeros(grid_gpu.shape, dtype=cl.cltypes.float)
    img_grid_cp = cl.Image(
        ctx, mf.WRITE_ONLY,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid_cp.shape)

    img_grid_gpu = cl.Image(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid_gpu.shape,
        hostbuf=grid_gpu)

    if num_voxels < 0:    
        return grid_gpu

    kernel = prg.quantize
    kernel(queue, grid_gpu.shape, None, img_grid_gpu, buf_params, img_grid_cp)

    cl.enqueue_copy(queue, grid_cp, img_grid_cp, origin=(0, 0, 0), region=grid_cp.shape)

    """ ## Plot cross section or debugging
    mid = int((num_voxels/2))
    img = plt.imshow(grid_cp_rot[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()
    """

    return grid_cp

## OLD WAY WITHOUT GPU
def gaussian_maps(num_voxels, angs_per_voxel, dist, X, Y, Z, atom_radius, atom_name, resname, chainID, resID, codeword):
    print("WARNING!! Using non-GPU version--seriously don't use this unless you don't have OpenCL/wayyy too much time on your hands!")
    print("WARNING!! To avoid this, run using -opencl")
    print("Creating Gaussian distance-based distribution map for %s" % (codeword))

    centroid_xyz = (num_voxels/2 + 0.5 , num_voxels/2 + 0.5 , num_voxels/2 + 0.5)

    ## Set protein "inside" to -15 and ligand to 5 (mimicking a paper that did this and just trying to replicate their stuff for now)
    surface_intensity = 1
    if codeword == "protein":
        inner_intensity = -15
    elif codeword == "ligand":
        inner_intensity = 5


    ##Here's all the loopppyyyy-nesss
    ##Every voxel is getting assigned an intensity. This loops over each dimension of the 3d grid (i,j,k) and is given an
    ##intensity value based on the contributions from atoms within a given distance (default 5A) and hence why
    ##I need the X,Y,Z,atom_radius lists. These contributions are given by
    ##a Gaussian distribution based on the atomic radii of the atom.
    ##The code here works and is correct, but it takes forever.

    # Precalculate expensive constants
    sqrt_2pi = np.sqrt(2 * np.pi)
    dist_sq = dist ** 2 

    grid = np.zeros([num_voxels,num_voxels,num_voxels])
    for i in range(0,num_voxels-1):
        for j in range(0,num_voxels-1):
            for k in range(0,num_voxels-1):
                #print(i, j, k, num_voxels-1)
                ## set voxel center to be in middle of voxel AKA in voxel (0,0,0) --> (0.5,0.5,0.5)
                i_cent, j_cent, k_cent = i+0.5, j+0.5, k+0.5
                u,v,w = -1*angs_per_voxel*(centroid_xyz[0]-i_cent) , -1*angs_per_voxel*(centroid_xyz[1]-j_cent) , -1*angs_per_voxel*(centroid_xyz[2]-k_cent)
                intensity = 0
                for x, y, z, sigma, res_ID in zip(X, Y, Z, atom_radius, resID):
                    const = 1 / (sqrt_2pi * sigma) 
                    r_sq = ((u-x)**2) + ((v-y)**2) + ((w-z)**2)
                    if r_sq <= dist_sq:
                        I = const * np.exp(-0.5 * ((np.sqrt(r_sq) / sigma) ** 2))
                        intensity += 1
                    elif r_sq <= 400 and res_ID > 400:
                        intensity += 1
                        break
                #grid[i, j, k] = intensity
                grid[k, j, i] = intensity
    if num_voxels < 0:
        return grid

    grid_cp = np.zeros(grid.shape)
    for i in range(0, num_voxels-1,1):
        for j in range(0, num_voxels-1,1):
            for k in range(0, num_voxels-1,1):
                #print(i, j, k, num_voxels-1)
                neighbors = grid[i-1:i+2, j-1:j+2, k-1:k+2]
                neighbor_sum = np.sum(neighbors) 
                val = 0
                if neighbor_sum != 0:
                    if np.abs(neighbor_sum) >= np.abs(20*inner_intensity) : ## completely inside--this should technically be 26*
                        # but it's dependent on resolution. E.g. if you use 2A/voxel, you're not going to get many voxels completely inside.
                        val = inner_intensity
                    elif 0 < np.abs(neighbor_sum) < np.abs(20*inner_intensity):
                        val = surface_intensity
                grid_cp[i, j, k] = val
                #grid_cp[k, j, i] = val

    """ For debugging
    print(grid_cp[grid_cp != 0])
    mid = int(num_voxels/2)
    img = plt.imshow(grid_cp[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()
    """
    return grid_cp

def rotate_grid(input_grid , rot_angles) :
    print("Ligand FFT Euler angles: %0.2f %0.2f %0.2f" % (rot_angles[0], rot_angles[1], rot_angles[2]))
    grid_rot = rotate(input_grid, angle=rot_angles[0], axes=[0,1],  reshape=False, mode='constant',cval=0.001)
    grid_rot = rotate(grid_rot, angle=rot_angles[1], axes=[0,2],  reshape=False, mode='constant',cval=0.001)
    grid_rot = rotate(grid_rot, angle=rot_angles[2], axes=[1,2],  reshape=False, mode='constant',cval=0.001)

    """
    ## Plot cross section or debugging
    new_grid_dim = input_grid.shape[0]
    mid = int((new_grid_dim/2))
    img = plt.imshow(input_grid[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()

    ## Plot cross section or debugging
    img = plt.imshow(grid_rot[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.title("Ligand FFT rotated %d degrees on the %s axis" % (rot_angle, axes_list[axis]))
    plt.show()
    """
    return  grid_rot

def rotate_grid_cl(l_fft, rot_angle, axis):
    # ss: Here, import is in the function because if you don't want/have pyopencl you can still use CPU mode
    import pyopencl as cl
    num_angles = np.round(360/rot_angle, 0)
    total_angles = 3*num_angles
    print("Creating rotated ligand FFT grids for every %s degrees... "
          "Running %d angles x 3 axis = %d ligand FFTs." % (rot_angle, num_angles, total_angles))

    ## Set protein "inside" to -15 and ligand to 2 (mimicking a paper that did this and just trying to replicate their stuff for now)
    surface_intensity = 1
    if codeword == "protein":
        inner_intensity = 15
    elif codeword == "ligand":
        inner_intensity = -2

    # ss: We need to transfer the data tensors to the GPU.
    # The numpy arrays must contain opencl types!
    xyzr = np.array([X, Y, Z, atom_radius], dtype=cl.cltypes.float)
    grid = np.zeros([num_voxels,num_voxels,num_voxels], dtype=cl.cltypes.float)
    params = np.array([angs_per_voxel, dist, inner_intensity, surface_intensity], dtype=cl.cltypes.float);

    # ss: It is not really necessary to create multiple contexts here for every single call, but since it's only two times in the script
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    clsource = open("gaussian_maps.cl", "r").read()
    prg = cl.Program(ctx, clsource).build()


    img_xyzr = cl.Image(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        xyzr.shape[1::-1],
        hostbuf=xyzr)

    buf_params = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=params)

    img_grid = cl.Image(
        ctx, mf.WRITE_ONLY,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid.shape)

    kernel = prg.gaussian_map
    # ss: This is equivalent to enqueueNDRangeKernel:
    # Each pixel is processed independently, in parallel (or at least part of it in batches depending on GPU hardware and image size)
    kernel(queue, grid.shape, None, img_xyzr, buf_params, img_grid)

    grid_gpu = np.empty_like(grid, dtype=cl.cltypes.float) # ss: Here, we retreive the results from the GPU
    cl.enqueue_copy(queue, grid_gpu, img_grid, origin=(0, 0, 0), region=grid_gpu.shape)

    grid_cp = np.zeros(grid_gpu.shape, dtype=cl.cltypes.float)
    img_grid_cp = cl.Image(
        ctx, mf.WRITE_ONLY,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid_cp.shape)

    img_grid_gpu = cl.Image(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
        grid_gpu.shape,
        hostbuf=grid_gpu)

    kernel = prg.quantize
    kernel(queue, grid_gpu.shape, None, img_grid_gpu, buf_params, img_grid_cp)

    cl.enqueue_copy(queue, grid_cp, img_grid_cp, origin=(0, 0, 0), region=grid_cp.shape)


    """ ## Plot cross section or debugging
    mid = int((num_voxels/2))
    img = plt.imshow(grid_cp_rot[mid,:,:], cmap=plt.get_cmap('viridis_r'))
    plt.colorbar(img)
    plt.show()
    """

    return grid_cp


def calc_pstruct_fft(protein_grid):
    """
    ## This protocol was taken directly from Katchalski-Katzir, Vakser (1992)
    p_bar from coordinates of pstruct
    Pconj = [DFT(p_bar)]*
    for rot in rotation:
        l_bar from coordinates of lstruct
        L = DFT(l_bar)
        C = Pconj * L
        c_bar = IFT(C)
    """
    p_bar = protein_grid

    Pconj = np.fft.fftn(p_bar)
    print('Pconj.shape:')
    print(Pconj.shape)
    #print(Pconj)
    #Pconj = (Pconj / len(p_bar)).astype('float32')
    Pconj = np.conjugate(Pconj)
    print('Pconj.shape:')
    print(Pconj.shape)
    #print(Pconj)

    return Pconj

def calc_lstruct_fft(ligand_grid):

    l_bar = ligand_grid
    L = np.fft.fftn(l_bar)
    print('L.shape:')
    print(L.shape)
    #print(L)

    return L

def calc_ifft(Pconj_shifted, L_shifted):
    C = Pconj_shifted * L_shifted
    c_bar = np.fft.ifftn(C)
    print('c_bar.shape:')
    print(c_bar.shape)
    #print(c_bar)
    c_bar_real = np.fft.fftshift(c_bar.real)
    c_bar_real = np.flip(c_bar_real)
    return np.where(c_bar_real < 0, 0, c_bar_real)

def get_max_ifft_voxels(ifft_in, n ):
    # get indices corresponding to maximum N values from IFFT
    #flat = ifft_in.flatten()
    #indices = np.argpartition(flat, -n)[-n:]
    #indices = indices[np.argsort(-flat[indices])]
    #indices = np.unravel_index(indices, ifft_in.shape)
    #ifft_out = np.zeros(ifft_in.shape).astype('float32')
    #for i,j,k in zip(indices[0],indices[1], indices[2]):
    #    ifft_out[i][j][k] = ifft_in[i][j][k]
    ## TODO: Save the IFFT as list with indices and values to save space, especially useful when only getting the top N output hotspots

    return ifft_out

def main():

    args = commandlineparser()
    num_rots = int(args.rot_angle)
    nstruct = args.nstruct
    dist = float(args.dist)
    angs_per_vox = float(args.angs_per_vox)

    resID_pocket_receptor = config.resID_pocket_receptor
    resID_pocket_ligand = config.resID_pocket_ligand
    
    print('resID_pocket_receptor:')
    print(resID_pocket_receptor)
    print('resID_pocket_ligand:')
    print(resID_pocket_ligand)

    surface_intensity = config.surface_intensity
    pocket_intensity = config.pocket_intensity
    protein_inner_intensity = config.protein_inner_intensity
    ligand_inner_intensity = config.ligand_inner_intensity

    fft_thredshold = config.fft_thredshold

    # only compute initial grids ONCE
    pstruct , p_centroid, offset , new_grid_dim, pocket_com_recptor, pocket_plane_params_receptor, pocket_atom_coors_receptor = \
        create_protein_grid(args.input_pstruct , dist , angs_per_vox, resID_pocket_receptor, config, args.opencl)
    num_voxels = pstruct.shape[0]
    pstruct_fft= calc_pstruct_fft(pstruct) ## return Pconj_shifted

    #np.savetxt('pstruct60.txt', pstruct[60], delimiter='\t', fmt='%.2f')

    fig_folder = 'figs'
    mrc_folder = 'mrcs'
    pdb_folder = 'pdbs'    
    os.mkdir(fig_folder)
    os.mkdir(mrc_folder)
    os.mkdir(pdb_folder)

    fig_dim = int(np.ceil(np.sqrt(new_grid_dim)))
    fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
    for i in range(len(pstruct)):
        ii = i // fig_dim
        ij = i % fig_dim
        im = axs[ii][ij].imshow(pstruct[i], origin='lower', vmin=-20, vmax=5)
        #axs[ii][ij].set_title('A slice of the protein grid')
        #fig.colorbar(im, ax=ax[ii][ij], label='Grid colorbar')
    #fig.colorbar(im, ax=axs[:, :], location='right', shrink=0.6)
    fig.colorbar(im, ax=axs[:, :], location='right')
    #plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(fig_folder + '/' + 'pstruct.png')

    fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
    for i in range(len(pstruct_fft)):
        ii = i // fig_dim
        ij = i % fig_dim
        im = axs[ii][ij].imshow(pstruct_fft.real[i], origin='lower', vmin=-20, vmax=20)
    fig.colorbar(im, ax=axs[:, :], location='right')
    plt.savefig(fig_folder + '/' + 'pstruct_fft_real.png')
    fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
    for i in range(len(pstruct_fft)):
        ii = i // fig_dim
        ij = i % fig_dim
        im = axs[ii][ij].imshow(pstruct_fft.imag[i], origin='lower', vmin=-20, vmax=20)
    fig.colorbar(im, ax=axs[:, :], location='right')
    plt.savefig(fig_folder + '/' + 'pstruct_fft_imag.png')

    print('pstruct:')
    print(pstruct.shape)
    #print(pstruct)
    dump_transformed_input_pdbs(args.input_pstruct, 'pstruct_transformed.pdb', offset, p_centroid)
    with mrcfile.new(mrc_folder + '/' + 'pstruct.mrc', overwrite=True) as mrc:
        mrc.set_data(pstruct.astype(np.float32))
        mrc.header.cella.x = angs_per_vox * num_voxels
        mrc.header.cella.y = angs_per_vox * num_voxels
        mrc.header.cella.z = angs_per_vox * num_voxels
        #mrc.print_header()
    print('pstruct_fft:')
    print(pstruct_fft.shape)
    #print(pstruct_fft)

    if len(args.input_lstructs) != 0:
        # only compute protein grid and FFT ONCE above--then ligand FFT
        for i in range(0, len(args.input_lstructs)):
            sd = sortedcontainers.SortedDict()
            dict_posi_rota_score = {}
            ligand_name = os.path.basename(args.input_lstructs[i]).split('.pdb')[0]
            lstruct, l_centroid, pocket_com_ligand, pocket_plane_params_ligand, pocket_atom_coors_ligand = create_ligand_grid(args.input_lstructs[i], new_grid_dim, args.angs_per_vox, args.dist, resID_pocket_ligand, config, args.opencl)
            lstruct_fft = calc_lstruct_fft(lstruct)

            print('Params of two planes:')
            print(pocket_plane_params_ligand)
            print(pocket_plane_params_receptor)
            print('Angle between two planes:')
            cos_theta_pocket_planes = np.dot(pocket_plane_params_ligand[0:3], pocket_plane_params_receptor[0:3])
            cos_theta_pocket_planes /= np.linalg.norm(pocket_plane_params_ligand[0:3]) * np.linalg.norm(pocket_plane_params_receptor[0:3])
            print(cos_theta_pocket_planes)
            theta_pocket_planes = np.emath.arccos(cos_theta_pocket_planes) / np.pi *180
            print(theta_pocket_planes)
            dis_pockets = np.linalg.norm(pocket_com_ligand - pocket_com_recptor)
            print('distance between two pockets:')
            print(dis_pockets)

            '''
            lstruct_conv = convolve(pstruct, lstruct, mode='constant', cval=0.0)
            print('lstruct_conv:')
            print(lstruct_conv.shape)
            print(lstruct_conv)
            np.savetxt('lstruct_conv60.txt', lstruct_conv[60], delimiter='\t', fmt='%.d')

            fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
            for iter in range(len(lstruct_conv)):
                ii = iter // fig_dim
                ij = iter % fig_dim
                im = axs[ii][ij].imshow(lstruct_conv[iter], origin='lower', vmin=0, vmax=10000)
            fig.colorbar(im, ax=axs[:, :], location='right')
            plt.savefig(fig_folder + '/' + 'lstruct_conv.png')
            '''

            fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
            for iter in range(len(pstruct)):
                ii = iter // fig_dim
                ij = iter % fig_dim
                im = axs[ii][ij].imshow(lstruct[iter], origin='lower', vmin=-5, vmax=5)
            fig.colorbar(im, ax=axs[:, :], location='right')
            #plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(fig_folder + '/' + 'lstruct.png')

            print('lstruct:')
            print(lstruct.shape)
            #print(lstruct)
            dump_transformed_input_pdbs(args.input_lstructs[i], 'lstruct_transformed.pdb', offset, l_centroid)

            with mrcfile.new(mrc_folder + '/' + 'lstruct.mrc', overwrite=True) as mrc:
                mrc.set_data(lstruct.astype(np.float32))
                mrc.header.cella.x = angs_per_vox * num_voxels
                mrc.header.cella.y = angs_per_vox * num_voxels
                mrc.header.cella.z = angs_per_vox * num_voxels
                #mrc.print_header()
            #print('lstruct_fft:')
            #print(lstruct_fft.shape)
            #print(lstruct_fft)
            #sys.exit()

            # Rotate ligand FFT and calculate IFFT
            ## TODO: Make GPU-compatible!!
            rot_angles_random = random.random(3) * num_rots
            print('rot_angles_random:')
            print(rot_angles_random)
            sys.exit()
            for roll in range(0, 360, num_rots): # rotate ligand FFT grid along x,y,z axes
                for pitch in range(0, 360, num_rots):
                    for yaw in range(0, 360, num_rots):
                        rot_angles = [roll + rot_angles_random[0], pitch + rot_angles_random[1], yaw + rot_angles_random[2]]
                        # https://www.cnblogs.com/xtl9/p/5445353.html
                        # https://scikit-learn.org/stable/modules/linear_model.html
                        # https://mp.weixin.qq.com/s?__biz=MzU0MTc2Njg4Nw==&mid=2247495464&idx=1&sn=ede61488a64604304308fff11e3d7861&chksm=fb264aa0cc51c3b6af11d534ca9fe11052f3b9578f420c57fab62d016e8f4ba04cfde4317fbf&scene=27
                        # https://aegis4048.github.io/mutiple_linear_regression_and_visualization_in_python
                        r = R.from_euler('zyx', rot_angles, degrees=True)
                        new_pocket_plane_params_ligand = r.apply(pocket_plane_params_ligand[0:3])
                        cos_theta_pocket_planes = np.dot(new_pocket_plane_params_ligand, pocket_plane_params_receptor[0:3])
                        cos_theta_pocket_planes /= np.linalg.norm(new_pocket_plane_params_ligand) * np.linalg.norm(pocket_plane_params_receptor[0:3])
                        #if cos_theta_pocket_planes > np.cos(config.angle_pocket_thredshold / 180 * np.pi):
                        #    print('skip rotation ' + str(rot_angles))
                        #    continue
                        if pstruct_fft.shape != lstruct_fft.shape:
                            raise Exception("GRIDS DIFFERENT SIZES! THIS SHOULDN'T HAPPEN BUT WE HAVE A PROBLEM!")
                        lstruct = rotate_grid(lstruct, rot_angles)
                        lstruct_fft = calc_lstruct_fft(lstruct)
                        #lstruct_fft = rotate_grid(lstruct_fft, rot_angles)
                        
                        '''
                        fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
                        for iter in range(len(pstruct)):
                            ii = iter // fig_dim
                            ij = iter % fig_dim
                            im = axs[ii][ij].imshow(lstruct[iter], origin='lower', vmin=-5, vmax=5)
                        fig.colorbar(im, ax=axs[:, :], location='right')
                        plt.savefig(fig_folder + '/' + 'lstruct_rotation_' + str(roll) + '_' + str(pitch) + '_' + str(yaw) + '.png')

                        fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
                        for iter in range(len(lstruct_fft)):
                            ii = iter // fig_dim
                            ij = iter % fig_dim
                            im = axs[ii][ij].imshow(lstruct_fft.real[iter], origin='lower', vmin=-10, vmax=10)
                        fig.colorbar(im, ax=axs[:, :], location='right')
                        plt.savefig(fig_folder + '/' + 'lstruct_fft_real_rotation_' + str(roll) + '_' + str(pitch) + '_' + str(yaw) + '.png')
                        fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
                        for iter in range(len(lstruct_fft)):
                            ii = iter // fig_dim
                            ij = iter % fig_dim
                            im = axs[ii][ij].imshow(lstruct_fft.imag[iter], origin='lower', vmin=-10, vmax=10)
                        fig.colorbar(im, ax=axs[:, :], location='right')
                        plt.savefig(fig_folder + '/' + 'lstruct_fft_imag_rotation_' + str(roll) + '_' + str(pitch) + '_' + str(yaw) + '.png')
                        '''

                        ifft = - calc_ifft(pstruct_fft,lstruct_fft)
                        #ifft = -lstruct_conv

                        flat = ifft.flatten()

                        #print('ifft:')
                        #print(ifft.shape)
                        #print(ifft)
                        #np.savetxt('ifft30.txt', ifft[30], delimiter=',', fmt='%1.2e')
                        #plot_cube(ifft, num_voxels, 'ifft')
                        #print('flat:')
                        #print(flat.shape)
                        #print(flat)
                        #np.savetxt('flat.txt', flat, delimiter=',', fmt='%1.2e')
                        
                        #with mrcfile.new(mrc_folder + '/' + 'ifft_rotation_' + str(roll) + '_' + str(pitch) + '_' + str(yaw) + '.mrc', overwrite=True) as mrc:
                        #    mrc.set_data(ifft.real.astype('float32'))
                        #    mrc.header.cella.x = angs_per_vox * num_voxels
                        #    mrc.header.cella.y = angs_per_vox * num_voxels
                        #    mrc.header.cella.z = angs_per_vox * num_voxels

                        '''
                        fig, axs = plt.subplots(fig_dim, fig_dim, sharex=True, sharey=True, figsize=[45, 40])
                        for iter in range(len(ifft)):
                            ii = iter // fig_dim
                            ij = iter % fig_dim
                            im = axs[ii][ij].imshow(ifft.real[iter], origin='lower', vmin=-10000, vmax=0)
                        fig.colorbar(im, ax=axs[:, :], location='right')
                        plt.savefig(fig_folder + '/' + 'ifft_rotation_' + str(roll) + '_' + str(pitch) + '_' + str(yaw) + '.png')
                        '''
                        
                        indices = np.argpartition(flat, nstruct)[:nstruct]
                        indices = indices[np.argsort(flat[indices])]
                        indices = np.unravel_index(indices, ifft.shape)
                        #print('len(sd.keys()):')
                        #print(len(sd.keys()))
                        if len(sd.keys()) < nstruct:
                            score_cutoff = ifft[indices[0][-1]][indices[1][-1]][indices[2][-1]]
                        else:
                            score_cutoff = sd.peekitem(-1)[0]
                        #score_cutoff = -10000
                        score_cutoff = fft_thredshold
                        indices = np.where(ifft < score_cutoff)
                        print('score_cutoff:')
                        print(score_cutoff)
                        print('indices:')
                        print(indices)
                        print("Roll %d pitch %d yaw %d had %d scores above %0.2f cutoff to add to final output"
                              % (roll,pitch,yaw, len(indices[0]), score_cutoff))
                        for zvox,yvox,xvox in zip(indices[0],indices[1],indices[2]):
                            new_pocket_com_ligand = r.apply(pocket_com_ligand - l_centroid) + np.array([xvox * angs_per_vox, yvox * angs_per_vox, zvox * angs_per_vox])
                            dis_pockets = np.linalg.norm(new_pocket_com_ligand - pocket_com_recptor + p_centroid - np.array([new_grid_dim * angs_per_vox / 2, new_grid_dim * angs_per_vox / 2, new_grid_dim * angs_per_vox / 2]))
                            #print('dis_pockets:')
                            #print(dis_pockets)
                            if dis_pockets > config.dist_pocket_thredshold:
                                continue

                            vox_val = round(ifft.real[zvox][yvox][xvox], 2)
                            key = str("%d %d %d %d %d %d %.2f") % (xvox,yvox,zvox,roll,pitch,yaw,dis_pockets)
                            sd.update({vox_val: key})
                            dict_posi_rota_score.update({key: vox_val})

                        #print('sd:')
                        #print(sd)
                        l_sd = sd.items()[:nstruct]
                        print('l_sd:')
                        print(l_sd)
                        #print(l_sd[0])
                    
                        '''
                        for n in range(0,nstruct):
                            euler_angles = [str(l_sd[n][1]).split()[3],str(l_sd[n][1]).split()[4],str(l_sd[n][1]).split()[5]]
                            #print('euler_angles:')
                            #print(euler_angles)
                            l_offset = [float(str(l_sd[n][1]).split()[2]),float(str(l_sd[n][1]).split()[1]),float(str(l_sd[n][1]).split()[0])]
                            #print('l_offset:')
                            #print(l_offset)
                            l_offset = [(l_offset[i] - num_voxels / 2) * angs_per_vox for i in range(0,len(l_offset))]
                            #print('l_offset:')
                            #print(l_offset)
                            dump_transformed_input_pdbs2(args.input_lstructs[i], pdb_folder + '/' + 'lstruct_docked_' + str(n) + '.pdb', offset, p_centroid, l_offset, l_centroid, euler_angles)
                        '''
                        print(str(len(dict_posi_rota_score.items())) + ' poses kept so far.')
                        #sys.exit()

            #sd = sd.items()[:nstruct]
            #print('sd:')
            #print(sd)
            #print('dict_posi_rota_score:')
            #print('after sort:')
            #print(sorted(dict_posi_rota_score.items(), key = lambda kv:(kv[1], kv[0])))
            dict_posi_rota_score = sorted(dict_posi_rota_score.items(), key = lambda kv:(kv[1], kv[0]))[:nstruct]
            if len(dict_posi_rota_score) < nstruct:
                nstruct = len(dict_posi_rota_score)
            print(str(nstruct) + ' poses will be saved.')
            for n in range(0,nstruct):
                print('Rank ' + str(n) + ':')
                outfile = str("%s_docked_final_%d.pdb" % (ligand_name, n))
                l_offset = [float(str(dict_posi_rota_score[n][0]).split()[0]),float(str(dict_posi_rota_score[n][0]).split()[1]),float(str(dict_posi_rota_score[n][0]).split()[2])]
                #l_offset = [(num_voxels-l_offset[i])*angs_per_vox for i in range(0,len(l_offset))]
                l_offset = [(l_offset[i])*angs_per_vox for i in range(0,len(l_offset))]
                print('l_offset:')
                print(l_offset)
                euler_angles = [str(dict_posi_rota_score[n][0]).split()[3],str(dict_posi_rota_score[n][0]).split()[4],str(dict_posi_rota_score[n][0]).split()[5]]
                print('euler_angles:')
                print(euler_angles)
                print('dis_pockets:')
                print(str(dict_posi_rota_score[n][0]).split()[6])
                dump_transformed_input_pdbs2(args.input_lstructs[i], pdb_folder + '/' + outfile, offset, p_centroid, l_offset, l_centroid, euler_angles)
            #dump_transformed_input_pdbs2(args.input_lstructs[i], pdb_folder + '/' + 'blah.pdb', offset, p_centroid, )

            """
            #if args.top_ifft_voxels:
            #    ifft = get_max_ifft_voxels(ifft, args.top_ifft_voxels)
            ## Get top ligand orientation and output back into transformed xyz
            if args.nstruct: 
                if 1 <= args.nstruct <= num_rots**3 :
                    centroids = get_max_ifft_voxels(ifft, args.top_ifft_voxels)
                    get_top_poses(lstruct, args.nstruct, centroids, offset, ligand_name)
                else:
                    raise Exception("-nstruct flag requires positive integer between 1 and %d" % (num_rots**3))
            """
        """
        if args.plots:
            plots = ['pstruct', 'lstruct', 'pstruct_fft', 'lstruct_fft',
                     'ifft']  ## 'pstruct_fft' needs to be abs(Pconj) so handling this a little differently.
            for outname in plots:
                if outname in args.plots:
                    if 'pstruct_fft' in args.plots:
                        Pconj_abs = np.abs(pstruct)
                        Pconj_shifted = np.fft.fftshift(Pconj_abs)
                        pstruct_fft = Pconj_shifted
                    struct = locals()[outname]
                    plot_cube(struct, num_voxels, outname)
                    #make_3dplots(struct, num_voxels, outname)
        """

    return 0

def commandlineparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-p", dest="input_pstruct", required=True)
    parser.add_argument(
        "-l", dest="input_lstructs", required=False, nargs='+') ## TODO: make this accept multiple ligands
    parser.add_argument(
        "-dist", dest="dist", type=float, default="5",
        help="Maximum distance for Gaussian distribution calculations."
        "AKA atoms will see each other from 5A away.")
    parser.add_argument(
        "-angs_per_vox", dest="angs_per_vox", type=float, default="1",
        help="Distance of each voxel--0.25A is nice, but slow. "
        "Try 1 or higher if things are just too slow. Default is 1.0.")
    parser.add_argument(
        "-rot_angle", dest="rot_angle", type=float, default="360",
        help="Angle for ligand rotation. E.g. if -rot_angle 30, ligand would rotate"
             "30 degrees, requiring 360/30 = 20 rounds of sampling in roll, pitch and yaw. Default is 360 for now AKA not rotating.")
    parser.add_argument(
        "-nstruct", dest="nstruct", type=int, default="1", help="Number of top scoring output docked ligand poses to output for each ligand."
    )
    parser.add_argument(
        "-top_ifft_voxels", dest="top_ifft_voxels", type=int, default="10",
        help="Number of top correlating IFFT voxels to output to ifft1.mrc (Good for debugging and visualizing IFFT 'hotspots').")
    parser.add_argument(
        "-opencl", action="store_true",
        help="Use the GPU to speed up the calculation")
    parser.add_argument("-plots", dest="plots", nargs='+', required=False, help="Choose any/all of the following"
                                                            ": pstruct, lstruct, pstruct_fft, lstruct_fft, ifft")
    parser.add_argument("-output_inputpdbs", dest="output_inputpdbs", action='store_true',
        help="Output transformed PDBs that you input to align these with output grid points. Otherwise input PDBs do not match up with output grids.")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
