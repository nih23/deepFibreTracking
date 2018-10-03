from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere, HemiSphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.io import read_bvals_bvecs
from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines, length
from dipy.tracking import metrics
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

from joblib import Parallel, delayed
import multiprocessing
import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

def normalize_dwi(weights, b0):
    """ Normalize dwi by the first b0.
    source: https://github.com/ppoulin91/learn2track/blob/miccai2017_submission/learn2track/neurotools.py
    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : ndarray of shape (X, Y, Z)
        B0 image.
    Returns
    -------
    ndarray
        Diffusion weights normalized by the B0.
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    nb_erroneous_voxels = np.sum(weights > b0)
    if nb_erroneous_voxels != 0:
        print ("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed


def get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=8, smooth=0.006, first=False):
    """ Compute coefficients of the spherical harmonics basis.
    source: https://github.com/ppoulin91/learn2track/blob/miccai2017_submission/learn2track/neurotools.py
    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    Returns
    -------
    sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number of
        coeffs depends on `sh_order`.
    """
    bvals = np.asarray(bvals)
    bvecs = np.asarray(bvecs)
    dwi_weights = dwi.astype("float32")

    # Exract the averaged b0.
    #b0_idx = bvals == 0
    b0_idx = bvals < 10 # 10/02/18 NH: changed to work with HCP data
    b0 = dwi_weights[..., b0_idx].mean(axis=3)

    # Extract diffusion weights and normalize by the b0.
    bvecs = bvecs[np.logical_not(b0_idx)]
    weights = dwi_weights[..., np.logical_not(b0_idx)]
    weights = normalize_dwi(weights, b0)

    # Assuming all directions are on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(weights, invB.T)
    return data_sh, weights, b0





def convertIntoSphericalCoordsAndNormalize(train_prevDirection, train_nextDirection):
    train_nextDirection_sph = convAllToSphCoords(train_nextDirection)
    train_prevDirection_sph = convAllToSphCoords(train_prevDirection)
    train_nextDirection_sph[:,1] = (train_nextDirection_sph[:,1] + np.pi) / (2 * np.pi)
    train_prevDirection_sph[:,1] = (train_prevDirection_sph[:,1] + np.pi) / (2 * np.pi)
    return train_prevDirection_sph, train_nextDirection_sph

def loadDataset(path):
    bvals, bvecs = read_bvals_bvecs(path + '/bvals', path + '/bvecs')
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    img = nb.load(path + '/data.nii.gz')
    dwi = img.get_data()
    aff = img.affine
    img = nb.load(path + '/T1w_acpc_dc_restore_1.25.nii.gz')
    t1 = img.get_data()
    binarymask, options = nrrd.read(path + '/nodif_brain_mask.nrrd')

    return bvals,bvecs,gtab,dwi,aff,t1,binarymask


def cropDatsetToBValue(bthresh, bvals, bvecs, dwi):
    singleShellIndices = np.where( np.abs(bvals - bthresh)  < 100)[0]
    dwi_subset = dwi[:,:,:,singleShellIndices]
    gtab_subset = gradient_table(bvals=bvals[singleShellIndices], bvecs=bvecs[singleShellIndices,])
    return dwi_subset, gtab_subset


def convAllFromSphToEuclCoords(data):
    euclCoords = np.zeros([len(data),3])
    #print('projecting from spherical to euclidean space. assuming all vector to be of unit lengths.')
    for i in range(len(data)):
        x,y,z = spheToEuclidean(1, data[i,0],data[i,1])
        euclCoords[i,0] = x
        euclCoords[i,1] = y
        euclCoords[i,2] = z
    return euclCoords



def convAllToSphCoords(data):
    sphCoords = np.zeros([len(data),2])
    #print('converting from euclidean to spherical space. assuming all vector to be of unit lengths.')
    #TODO: check for unit lengths or normalize vectors
    for i in range(len(data)):
        r,t,p = euclToSpherical(data[i,0],data[i,1],data[i,2])
        sphCoords[i,0] = t
        sphCoords[i,1] = p
    return sphCoords


def euclToSpherical(x,y,z):
    r = np.sqrt(x**2 + y ** 2 + z ** 2)
    theta = np.nan_to_num(np.arccos(z / r))
    psi = np.nan_to_num(np.arctan2(y,x))
    return r,theta,psi


def spheToEuclidean(r,theta,psi):
    xx = r * np.sin(theta) * np.cos(psi)
    yy = r * np.sin(theta) * np.sin(psi)
    zz = r * np.cos(theta)
    return xx,yy,zz


def _interpolateVolume(j):
    global data
    global coordVecs
    
    return vfu.interpolate_scalar_3d(data[:,:,:,j],coordVecs)[0]


def interpolatePartialDWIVolume(dwi, centerPosition, x_,y_,z_, noX = 8, noY = 8, noZ = 8, coordinateScaling = 1,):
    szDWI = dwi.shape
    coordVecs = np.vstack(np.meshgrid(x_,y_,z_, indexing='ij')).reshape(3,-1).T + centerPosition   
    x = np.zeros([noX,noY,noZ,szDWI[-1]])
    #PARALLEL BUT SLOWER VERSION ...
    #global data
    #global coordVecs
    #num_cores = multiprocessing.cpu_count()
    #data = dwi
    #results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(0,szDWI[-1]))    
    #pool = multiprocessing.Pool(processes=num_cores)
    #results = pool.map(_interpolateVolume, range(0,szDWI[-1]))
    #pool.close()
    #results = np.swapaxes(results,0,1)
    #return np.reshape(np.array(results),[noX,noY,noZ,szDWI[-1]])
    
    #return results
    
    for i in range(0,szDWI[-1]):
        x[:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],coordVecs)[0], [noX,noY,noZ])
    return x


def visSphere(sphere):
    '''
    Visualize sphere
    '''
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.point(sphere.vertices, window.colors.red, point_radius=0.05))
    window.show(ren)

def visTwoSetsOfStreamlines(streamlines,streamlines2, volume, vol_slice_idx = 40, vol_slice_idx2 = 40):
    '''
    Visualize streamline using vtk
    '''
    # Prepare the display objects.
    #color = line_colors(streamlines)

    if window.have_vtk:
        vol_actor = actor.slicer(volume)

        vol_actor.display(y=vol_slice_idx)
        vol_actor2 = vol_actor.copy()
        vol_actor2.display(z=vol_slice_idx2)
        
        hue = [0, 0.5]  # red only
        saturation = [0.0, 1.0]  # black to white
        lut_cmap = actor.colormap_lookup_table(
        scale_range=(40, 200),
        hue_range=hue,
        saturation_range=saturation)

        streamlines_actor = actor.line(streamlines, (125,125,125))
        streamlines_actor2 = actor.line(streamlines2, lookup_colormap= lut_cmap) #(1., 0.5, 0))

        # Create the 3D display.
        r = window.Renderer()
        r.add(streamlines_actor)
        r.add(streamlines_actor2)
        r.add(vol_actor)
        r.add(vol_actor2)
        #window.record(r, n_frames=1, out_path='deterministic.png', size=(800, 800))
        window.show(r)
    else:
        print('we need VTK for proper visualisation of our fibres.')
    
    
def visStreamlines(streamlines, volume, vol_slice_idx = 40, vol_slice_idx2 = 40):
    '''
    Visualize streamline using vtk
    '''
    # Prepare the display objects.
    #color = line_colors(streamlines)

    if window.have_vtk:
        vol_actor = actor.slicer(volume)

        vol_actor.display(y=vol_slice_idx)
        vol_actor2 = vol_actor.copy()
        vol_actor2.display(z=vol_slice_idx2)
        
        streamlines_actor = actor.line(streamlines, line_colors(streamlines, cmap = 'rgb_standard'))
        #streamlines_actor = actor.line(streamlines, (125,125,125))

        # Create the 3D display.
        r = window.Renderer()
        r.add(streamlines_actor)
        #r.add(vol_actor)
        r.add(vol_actor2)
        window.record(r, n_frames=1, out_path='deterministic.png', size=(800, 800))
        window.show(r)
    else:
        print('we need VTK for proper visualisation of our fibres.')
        
def filterStreamlinesByLength(streamlines, minimumLength = 80):
    '''
    Removes streamlines that are shorter (in mm) than minimumLength
    '''
    return [x for x in streamlines if metrics.length(x) > minimumLength]



def generateSimpleTraindataFromStreamlines(streamlines, dwi, rec_level_sphere = 3):
    '''
    deprecated, dont use
    '''
    warnings.warn("deprecated", DeprecationWarning)
    sfa = np.asarray(streamlines)
    sph = subdivide_octahedron.create_unit_sphere(recursion_level=rec_level_sphere) # create unit sphere with 4 ** recursion_level + 2 vertices
    #visSphere(sph)
    dx,dy,dz,dw = dwi.shape
    noStreamlines = min(len(sfa), 1000) # FIXME: spares us some time right now
    train_X = []
    train_Y = []
    
    for streamlineIndex in range(0,noStreamlines):
        print('Streamline ' + str(streamlineIndex) + '/' + str(noStreamlines))
        lengthStreamline = len(sfa[streamlineIndex])
        for streamlineElementIndex in range(0,lengthStreamline-1):
            # center sphere around current streamline position
            sph2 = sph.vertices + sfa[streamlineIndex][streamlineElementIndex]
            sph2 = np.vstack((sph2,sfa[streamlineIndex][streamlineElementIndex]))
            
            # interpolate data given these coordinates for each channel
            x = np.zeros([4**rec_level_sphere+2+1,dw])
            for i in range(0,dw):
                x[:,i] = vfu.interpolate_scalar_3d(dwi[:,:,:,i],sph2)[0]
            train_X.append(x)
            
            old_y = sfa[streamlineIndex][streamlineElementIndex+1]
            nextStreamlineDirection = sfa[streamlineIndex][streamlineElementIndex] - sfa[streamlineIndex][streamlineElementIndex+1]
            
            train_Y.append(nextStreamlineDirection) # dont store absolute value but relative displacement
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)
    return train_X, train_Y


def generateGridSimpleTraindataFromStreamlines(streamlines, dwi, rec_level_sphere = 3, noX=3, noY=3,noZ=3,coordinateScaling = 1):
    '''
    deprecated, dont use
    '''
    warnings.warn("deprecated", DeprecationWarning)
    sfa = np.asarray(streamlines)
    dx,dy,dz,dw = dwi.shape
    noStreamlines = min(len(sfa), 1000) # FIXME: spares us some time right now
    train_X = []
    train_Y = []
    
    for streamlineIndex in range(0,noStreamlines):
        if((streamlineIndex % 10000) == 0):
            print('Streamline ' + str(streamlineIndex) + '/' + str(noStreamlines))
        lengthStreamline = int(len(sfa[streamlineIndex]) / 10)
        for streamlineElementIndex in range(0,lengthStreamline-1):           
            x_ = coordinateScaling * np.linspace(-1., 1., noX)
            y_ = coordinateScaling * np.linspace(-1., 1., noY)
            z_ = coordinateScaling * np.linspace(-1., 1., noZ)
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + sfa[streamlineIndex][streamlineElementIndex]
            
            # interpolate data given these coordinates for each channel
            x = np.zeros([noX,noY,noZ,dw])
            for i in range(0,dw):
                x[:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],coordVecs)[0], [noX,noY,noZ])
            train_X.append(x)
            old_y = sfa[streamlineIndex][streamlineElementIndex+1]
            nextStreamlineDirection = sfa[streamlineIndex][streamlineElementIndex] - sfa[streamlineIndex][streamlineElementIndex+1]
            
            train_Y.append(nextStreamlineDirection) # dont store absolute value but relative displacement
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)
    return train_X, train_Y

def getCoordinateGrid(noX,noY,noZ,coordinateScaling):
    #print("using " + str([noX,noY,noZ]) + "px interpolation grid with coordinateScaling " + str(coordinateScaling))
    x_ = coordinateScaling * np.linspace(-4., 4, noX)
    y_ = coordinateScaling * np.linspace(-4., 4., noY)
    z_ = coordinateScaling * np.linspace(-4., 4., noZ)
    
    x_ = coordinateScaling * np.linspace(-1 * noX/2, noX/2, noX)
    y_ = coordinateScaling * np.linspace(-1 * noX/2, noY/2, noY)
    z_ = coordinateScaling * np.linspace(-1 * noX/2, noZ/2, noZ)
    
    # dirty hack ...
    if(noX == 1):
        x_ = [0]
        
    if(noY == 1):
        y_ = [0]
        
    if(noZ == 1):
        z_ = [0]
    
    return x_,y_,z_

def generatePredictionNetworkTrainingDataFromStreamlines(streamlines, dwi, rec_level_sphere = 3, noX=3, noY=3,noZ=3,coordinateScaling = 1, noCrossings = 3, distToNeighbours = 0.5, maximumNumberOfNearbyStreamlinePoints = 3, affine = np.eye(4,4)):
    '''
    
    '''
    sfa = np.asarray(streamlines)
    dx,dy,dz,dw = dwi.shape
    noStreamlines = min(len(sfa), 1000) # FIXME: spares us some time right now
    noNeighbours = 2*noCrossings + 1
    sl_pos = sfa[0]
    # Build kd-tree of streamline positions. This significantly increases subsequent lookup times.
    for streamlineIndex in range(0,noStreamlines):
        lengthStreamline = int(len(sfa[streamlineIndex]) / 10)
        sl_pos = np.concatenate([sl_pos, sfa[streamlineIndex][0:lengthStreamline]], axis=0) # dont store absolute value but relative displacement
    
    kdt = KDTree(sl_pos)
    
    print('Generating training data')
   
    # define spacing of the 3D grid
    x_,y_,z_ = getCoordinateGrid(noX,noY,noZ,coordinateScaling)
    
    # initialize our supervised training data
    train_Y_1 = np.zeros([len(sl_pos),2*noCrossings,3]) # likely next streamline directions
    train_Y_2 = np.zeros([len(sl_pos),3]) # next direction
    train_X_2 = np.zeros([len(sl_pos),3]) # previous direction
    train_X = np.zeros([len(sl_pos),noX,noY,noZ,dw]) # interpolated dwi dataset for each streamline position
    
    for streamlineIndex in range(0,len(sl_pos)):
        if((streamlineIndex % 1000) == 0):
            print(str(streamlineIndex) + "/" + str(len(sl_pos)))

        streamlinevec = sl_pos[streamlineIndex]
        streamlinevec_next = sl_pos[min((streamlineIndex+1,len(sl_pos)-1))]
        streamlinevec_prev = sl_pos[max((streamlineIndex-1,0))]
        #d = np.sum( (streamlinevec - streamlinevec_next)**2)
        #i = kdt.query_ball_point(streamlinevec,distToNeighbours) # acquire neighbours within a certain distance
        #i = i[0:2*noCrossings] # limit length to the nearest neighbours
        #n_slv = kdt.data[i] - streamlinevec # direction to nearest streamline position
        #n_slv = np.unique(n_slv, axis = 0) # remove duplicats from our search query
        #noPoints,dimPoints = n_slv.shape
        #train_Y_1[streamlineIndex,0:noPoints,] = n_slv
        train_Y_2[streamlineIndex,] = streamlinevec_next - streamlinevec
        train_Y_2[streamlineIndex,] = np.nan_to_num(train_Y_2[streamlineIndex,] / np.sqrt(np.sum(train_Y_2[streamlineIndex,] ** 2))) # make unit vector
        train_X_2[streamlineIndex,] = streamlinevec_prev - streamlinevec
        train_X_2[streamlineIndex,] = np.nan_to_num(train_X_2[streamlineIndex,] / np.sqrt(np.sum(train_X_2[streamlineIndex,] ** 2))) # make unit vector
        train_X[streamlineIndex,] = interpolatePartialDWIVolume(dwi,streamlinevec, noX = noX, noY = noY, noZ = noZ, coordinateScaling = coordinateScaling,x_ = x_,y_ = y_,z_ = z_)

        
    return train_X, train_X_2, train_Y_1, train_Y_2