import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere, HemiSphere
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.io import read_bvals_bvecs
from dipy.viz import window, actor
from dipy.tracking import metrics
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

import time
import nrrd
import nibabel as nb
import numpy as np
import dipy

from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate

#from src.state import TractographyInformation

import math

import vtk
from scipy.interpolate import griddata

from dipy.data import get_sphere

from dipy.align.reslice import reslice
from sklearn.neighbors import NearestNeighbors


def discretizeTangents(bvecs, tangents):
    if(len(tangents) == 1):
        return np.argmin( np.sum((bvecs - tangents)**2, axis = 1) )
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(bvecs)
    distances, indices = nbrs.kneighbors(tangents)
    runtime = time.time() - start_time
    print('Mean discretization error: ' + str(np.mean(distances)))
    print('Runtime ' + str(runtime) + ' s')
    return indices


def projectDiscretizedTangentsBack(bvecs, label):
    return [bvecs[i,] for i in label]


def projectIntoAppropriateSpace(myState, dwi):
    #print(myState.repr)
    if(myState.repr == 'sh'):
        #print('Spherical Harmonics (ours)')
        start_time = time.time()
        tracking_data = get_spherical_harmonics_coefficients(bvals=myState.bvals,bvecs=myState.bvecs,sh_order=myState.shOrder, dwi=dwi, b0 = None) # assuming normnalized dwi data
        runtime = time.time() - start_time
        #print('Runtime ' + str(runtime) + 's')
    elif(myState.use2DProjection):
        #print('2D projection')
        start_time = time.time()
        tracking_data, resamplingSphere = resample_dwi_2D(dwi, myState.b0, myState.bvals, myState.bvecs, sh_order=myState.shOrder, smooth=0, mean_centering=False)
        runtime = time.time() - start_time
        #print('Runtime ' + str(runtime) + 's')
        #print('Shape: ' + str(tracking_data.shape))
    elif(myState.repr == 'res100'):
        #print('Resampling to 100 directions')
        start_time = time.time()
        tracking_data, resamplingSphere = resample_dwi(dwi, myState.b0, myState.bvals, myState.bvecs, sh_order=myState.shOrder, smooth=0, mean_centering=False)
        runtime = time.time() - start_time
        #print('Runtime ' + str(runtime) + 's')
    elif(myState.repr == 'raw'):
        tracking_data = dwi
    elif(myState.repr == '2Dcnn'):
        dwi = np.squeeze(dwi)
        tracking_data = dwi[..., np.newaxis]
    else:
        print('[ERROR] no data representation specified (raw, sh, res100, 2D')
        return None

    return tracking_data


def loadVTKstreamlines(pStreamlines, reportProgress = True):
    isPDR = False
    if(pStreamlines[-4:] == '.vtk'):
        isPDR = True
        reader = vtk.vtkPolyDataReader()
    elif (pStreamlines[-4:] == '.vtp'):
        isPDR = False
        reader = vtk.vtkXMLPolyDataReader()
    
    reader.SetFileName(pStreamlines)
    if(isPDR):
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
    reader.Update()

    polydata = reader.GetOutput()
    streamlines = []
    
    for i in range(polydata.GetNumberOfCells()):
    #for i in range(0,100):
        if(reportProgress and ((i % 10000) == 0)):
            print(str(i) + "/" + str(polydata.GetNumberOfCells()))
        c = polydata.GetCell(i)
        p = c.GetPoints()
        streamlines.append(np.array(p.GetData()))

    return streamlines


def saveVTKstreamlines(streamlines, pStreamlines):
    polydata = vtk.vtkPolyData()

    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    
    ptCtr = 0
       
    for i in range(0,len(streamlines)):
        if((i % 10000) == 0):
                print(str(i) + "/" + str(len(streamlines)))
        
        
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(len(streamlines[i]))
        for j in range(0,len(streamlines[i])):
            points.InsertNextPoint(streamlines[i][j])
            linePts = line.GetPointIds()
            linePts.SetId(j,ptCtr)
            
            ptCtr += 1
            
        lines.InsertNextCell(line)
                   
    print(str(i) + "/" + str(len(streamlines)))
            
    polydata.SetLines(lines)
    polydata.SetPoints(points)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(pStreamlines)
    writer.SetInputData(polydata)
    writer.Write()
    
    print("Wrote streamlines to " + writer.GetFileName())


def saveVTKstreamlinesWithPointdata(streamlines, pStreamlines, pointdata, normalizationFactor = 1):
    polydata = vtk.vtkPolyData()

    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    
    ptCtr = 0
    
    
    scalars = vtk.vtkFloatArray()
    scalars.SetName("empirical uncertainty")

    
       
    for i in range(0,len(streamlines)):
        if((i % 10000) == 0):
                print(str(i) + "/" + str(len(streamlines)))
        
        
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(len(streamlines[i]))
        for j in range(0,len(streamlines[i])):
            points.InsertNextPoint(np.nan_to_num(streamlines[i][j]))
            linePts = line.GetPointIds()
            linePts.SetId(j,ptCtr)
            
            scalars.InsertNextValue(pointdata[i][j,0] / normalizationFactor)
            
            ptCtr += 1
            
        lines.InsertNextCell(line)
                   
    polydata.SetLines(lines)
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(scalars)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(pStreamlines)
    writer.SetInputData(polydata)
    writer.Write()
    
    print("Wrote streamlines to " + writer.GetFileName())
    
    
def resample_dwi(dwi, b0, bvals, bvecs, directions=None, sh_order=8, smooth=0.006, mean_centering=True):
    """ Resamples a diffusion signal according to a set of directions using spherical harmonics.
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
    directions : `dipy.core.sphere.Sphere` object, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be used.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    mean_centering : bool
        If True, signal will have zero mean in each direction for all nonzero voxels
    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    data_sh = get_spherical_harmonics_coefficients(dwi, b0=b0, bvals=bvals, bvecs=bvecs, sh_order=sh_order, smooth=smooth)

    sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
       
    if directions is not None:
        sphere = Sphere(xyz=directions)
        #sphere = Sphere(xyz=bvecs[1:]) #TODO LOOK INTO THAT LINE!!!  WHY DO WE OMIT THE FIRST BVEC?!?!?!?!?!?!
        #print('WARNING THERE MIGHT BE A BUG HERE!!!')

    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)

    if mean_centering:
        # Normalization in each direction (zero mean)
        idx = data_resampled.sum(axis=-1).nonzero()
        means = data_resampled[idx].mean(axis=0)
        data_resampled[idx] -= means

    return data_resampled, sphere


def resample_dwi_2D(dwi, b0, bvals, bvecs, directions=None, sh_order=8, smooth=0.006, mean_centering=True, noThetas = 8, noPhis = 8):
    """ Resamples a diffusion signal according to a set of directions using spherical harmonics.
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
    directions : `dipy.core.sphere.Sphere` object, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be used.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    mean_centering : bool
        If True, signal will have zero mean in each direction for all nonzero voxels
    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    data_sh = get_spherical_harmonics_coefficients(dwi, bvals=bvals, bvecs=bvecs, sh_order=sh_order, smooth=smooth)

    # sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
    
    xi = np.arange(0,np.pi, (np.pi) / noThetas) # theta
    yi = np.arange(-np.pi,np.pi,2 * (np.pi) / noPhis) # phi

    orderedBasis = np.ones([len(xi)*len(yi,), 2])

    ctr = 0
    for i in range(len(xi)):
        for j in range(len(yi)):
            orderedBasis[ctr,] = [xi[i], yi[j]]
            ctr+=1
    
    sphere = Sphere(theta=orderedBasis[:,0],phi=orderedBasis[:,1])
    #sphere = get_sphere('repulsion100')
    
    if directions is not None:
        sphere = directions

    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)

    if mean_centering:
        # Normalization in each direction (zero mean)
        idx = data_resampled.sum(axis=-1).nonzero()
        means = data_resampled[idx].mean(axis=0)
        data_resampled[idx] -= means

    return data_resampled, sphere
    

def normalize_dwi(weights, b0):
    """ Normalize dwi by average b0 data
    adapted from: https://github.com/ppoulin91/learn2track/blob/miccai2017_submission/learn2track/neurotools.py
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
    print('Normalizing DWI signals')

    if(b0 is None):
        print('[W] no normalisation')
        return weights
    
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    nb_erroneous_voxels = np.sum(weights > b0)
    nb_valid_voxels = np.sum(weights <= b0)
    if nb_erroneous_voxels != 0:
        print ("Percentage erroneous voxels: %.2f" % ( 100 * nb_erroneous_voxels / (nb_erroneous_voxels + nb_valid_voxels) ) )
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = (weights / b0)
    
    # take the log (Fan's idea)
    #weights_normed = -1 * np.log(weights_normed)
    
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed


def get_spherical_harmonics_coefficients(dwi, b0, bvecs, sh_order=8, smooth=0.006, bvals=None):
    """ Compute coefficients of the spherical harmonics basis.
    adapted from: https://github.com/ppoulin91/learn2track/blob/miccai2017_submission/learn2track/neurotools.py
    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
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
    bvecs = np.asarray(bvecs)
    dwi_weights = dwi.astype("float32")

    # normalize by the b0.
### never normalize DWI signals..
#    if(not b0 is None):
#        dwi_weights = normalize_dwi(dwi_weights, b0)

    # Assuming all directions lie on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(dwi_weights, invB.T)

    return data_sh


def projectGradientsOntoGrid(sphere, z, factor = 0.25):
    xi = np.arange(np.min(sphere.theta),np.max(sphere.theta),np.max(sphere.theta) / (factor * len(sphere.theta))) # theta
    yi = np.arange(np.min(sphere.phi),np.max(sphere.phi),(np.abs(np.min(sphere.phi)) + np.max(sphere.phi)) / (factor * len(sphere.phi))) # phi
    xi,yi = np.meshgrid(xi,yi)
    zi = griddata((sphere.theta,sphere.phi),np.squeeze(z),(xi,yi),method='linear')
    zi = np.nan_to_num(zi)
    dx,dy = zi.shape
    if(dx == 25):
        zi = zi[1:,]
    
    return zi


def convertIntoSphericalCoordsAndNormalize(train_prevDirection, train_nextDirection):
    '''
    convert euclidean directional vectors of our training set into spherical coordinates and normalize to [0,1]
    '''
    train_nextDirection_sph = convAllToSphCoords(train_nextDirection)
    train_prevDirection_sph = convAllToSphCoords(train_prevDirection)
    train_nextDirection_sph[:,1] = (train_nextDirection_sph[:,1] + np.pi) / (2 * np.pi)
    train_prevDirection_sph[:,1] = (train_prevDirection_sph[:,1] + np.pi) / (2 * np.pi)
    return train_prevDirection_sph, train_nextDirection_sph

def transformISMRMDatsetToHCPCoordinates(dwi):
    I = np.eye(4)
    I[1,1] = -1
    I[1,3] = dwi.shape[1]
    dx,dy,dz,dw = dwi.shape
    dwiT = dwi.copy()
    print(str(I))
    for i in range(dw):
        dwiT[:,:,:,i] = dipy.align.reslice.affine_transform(dwi[:,:,:,i], I)
        
    return dwiT,I


def loadISMRMDataArtifactFree(path, resliceToHCPDimensions=True, denoiseData=False):
    '''
    import HCP dataset
    '''
    print('loading ISMRM artifact free data')
    bvals, bvecs = read_bvals_bvecs(path + '/NoArtifacts_Relaxation.bvals', path + '/NoArtifacts_Relaxation.bvecs')
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    img = nb.load(path + '/NoArtifacts_Relaxation.nii.gz')
    dwi = img.get_data()
    print(dwi.shape)
    
    #dwi,I = transformISMRMDatsetToHCPCoordinates(dwi)
    aff = img.affine
    #aff = I @ aff
    
    if(denoiseData):
        print("Denoising")
        t = time.time()
        sigma = pca_noise_estimate(dwi, gtab, correct_bias=True, smooth=3)
        print("Sigma estimation time", time.time() - t)

        dwi = localpca(dwi, sigma=sigma, patch_radius=2)

        print("Time taken for local PCA (slow)", time.time() - t)
    
    
    zooms = img.header.get_zooms()[:3]
    img = nb.load(path + '/T1.nii.gz')
    t1 = img.get_data()
    if(resliceToHCPDimensions):
        print("Reslicing to 1.25 mm^3")
        new_zooms = (1.25, 1.25, 1.25) # similar to HCP
        dwi, aff = reslice(dwi, aff, zooms, new_zooms)
    
    return bvals,bvecs,gtab,dwi,aff,t1



def loadISMRMData(path, resliceToHCPDimensions=True, denoiseData=False):
    '''
    import HCP dataset
    '''
    bvals, bvecs = read_bvals_bvecs(path + '/Diffusion.bvals', path + '/Diffusion.bvecs')
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    #img = nb.load(path + '/Diffusion.nii.gz') # raw data
    img = nb.load(path + '/ismrm_denoised_preproc_mrtrix.nii.gz') # denoised and motion corrected data
    print('Loading ismrm_denoised_preproc_mrtrix.nii.gz')
    dwi = img.get_data()
    print(dwi.shape)
    
    #dwi,I = transformISMRMDatsetToHCPCoordinates(dwi)
    aff = img.affine
    #aff = I @ aff
    
    if(denoiseData):
        print("Denoising")
        t = time.time()
        sigma = pca_noise_estimate(dwi, gtab, correct_bias=True, smooth=3)
        print("Sigma estimation time", time.time() - t)

        dwi = localpca(dwi, sigma=sigma, patch_radius=2)

        print("Time taken for local PCA (slow)", time.time() - t)
    
    
    zooms = img.header.get_zooms()[:3]
    img = nb.load(path + '/T1.nii.gz')
    t1 = img.get_data()
    if(resliceToHCPDimensions):
        print("Reslicing to 1.25 mm^3")
        new_zooms = (1.25, 1.25, 1.25) # similar to HCP
        dwi, aff = reslice(dwi, aff, zooms, new_zooms)
    
    return bvals,bvecs,gtab,dwi,aff,t1


def loadHCPData(path):
    '''
    import HCP dataset
    '''
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
    '''
    filter specific b-value measurements from a multi-shell dataset
    '''
    singleShellIndices = np.where( np.abs(bvals - bthresh)  < 100)[0]
    dwi_subset = dwi[:,:,:,singleShellIndices]
    bvals_subset = bvals[singleShellIndices]
    bvecs_subset = bvecs[singleShellIndices,]
    gtab_subset = gradient_table(bvals=bvals_subset, bvecs=bvecs_subset)
    return dwi_subset, gtab_subset, bvals_subset, bvecs_subset


def convAllFromSphToEuclCoords(data):
    '''
    convert array of vectors from spherical (polar) coordinate system into euclidean cs
    assume that all vectors are unit length
    '''
    euclCoords = np.zeros([len(data),3])
    #print('projecting from spherical to euclidean space. assuming all vector to be of unit lengths.')
    for i in range(len(data)):
        x,y,z = _spheToEuclidean(1, data[i,0],data[i,1])
        euclCoords[i,0] = x
        euclCoords[i,1] = y
        euclCoords[i,2] = z
    return euclCoords


def convAllToSphCoords(data):
    '''
    convert array of vectors from euclidean coordinate system into spherical (polar) coordinates
    assume that all vectors are unit length
    '''
    #TODO: check for unit lengths or normalize vectors
    sphCoords = np.zeros([len(data),2])
    for i in range(len(data)):
        r,t,p = _euclToSpherical(data[i,0],data[i,1],data[i,2])
        sphCoords[i,0] = t
        sphCoords[i,1] = p
    return sphCoords


def _euclToSpherical(x,y,z):
    '''
    convert euclidean coordinates into spherical coordinates
    '''
    r = np.sqrt(x**2 + y ** 2 + z ** 2)
    theta = np.nan_to_num(np.arccos(z / r))
    psi = np.nan_to_num(np.arctan2(y,x))
    return r,theta,psi


def _spheToEuclidean(r,theta,psi):
    '''
    convert spherical coordinates into euclidean coordinates
    '''
    xx = r * np.sin(theta) * np.cos(psi)
    yy = r * np.sin(theta) * np.sin(psi)
    zz = r * np.cos(theta)
    return xx,yy,zz


def getSizeOfDataRepresentation(myState):
    reprSize = -1
    if (myState.repr == 'sh'):
        if(myState.shOrder == 4):
            return 15
        elif(myState.shOrder == 8):
            return 45
        else:
            return -1
    elif(myState.repr == 'res100'):
        return 100

    return len(myState.bvals)


def _getCoordinateGrid(state):
    '''
    generate a grid of provided spatial extent which is used to interpolate data
    '''
    x_ = state.gridSpacing * np.linspace(-1 * state.dim[0], state.dim[0], state.dim[0])
    y_ = state.gridSpacing * np.linspace(-1 * state.dim[1], state.dim[1], state.dim[1])
    z_ = state.gridSpacing * np.linspace(-1, state.dim[2]-2, state.dim[2])

    # x_ = state.gridSpacing * np.linspace(-1 * state.dim[0], 0, state.dim[0])
    # y_ = state.gridSpacing * np.linspace(-1 * state.dim[1], 0, state.dim[1])
    # z_ = state.gridSpacing * np.linspace(-1 * state.dim[2], 0, state.dim[2])

    #x_ = state.gridSpacing * np.linspace(0, state.dim[0], state.dim[0])
    # y_ = state.gridSpacing * np.linspace(0, state.dim[1], state.dim[1])
    # z_ = state.gridSpacing * np.linspace(0, state.dim[2], state.dim[2])

    # dirty hack ...
    if (state.dim[0] == 1):
        x_ = [0]

    if (state.dim[1] == 1):
        y_ = [0]

    if (state.dim[2] == 1):
        z_ = [0]

    return x_, y_, z_


def interpolateDWIVolume(myState, dwi, positions, x_,y_,z_, rotations = None):
    # positions: noPoints x 3
    szDWI = dwi.shape
    noPositions = len(positions)
    noElem = myState.dim[0] * myState.dim[1] * myState.dim[2]
    cvF = np.ones([noPositions*noElem,3])
    grid = np.array(np.meshgrid(x_,y_,z_)).reshape(3,-1)
    for j in range(0,noPositions):
        grid_rotated = grid
        if(rotations is not None):
            grid_rotated = rotateByMatrix(grid,rotations[j,])
        coordVecs = (grid_rotated + positions[j,:,None]).T
        il = j * noElem
        ir = (j+1) * noElem 
        cvF[il:ir] = coordVecs

    x = np.zeros([noPositions,myState.dim[0],myState.dim[1],myState.dim[2],szDWI[-1]])

    for i in range(0,szDWI[-1]):
        interpRes = vfu.interpolate_scalar_3d(dwi[:,:,:,i],cvF)[0]
        x[:,:,:,:,i] = np.reshape(interpRes, [noPositions,myState.dim[0],myState.dim[1],myState.dim[2]])

    if(rotations is None):
        return x

    # resample the diffusion gradients wrt. to the rotation matrix
    for j in range(noPositions):
        bvecs_rot = np.dot(rotations[j,], myState.bvecs.T).T
        x_rot, _ = resample_dwi(x[j,], b0=myState.b0, bvals=myState.bvals, bvecs=myState.bvecs, directions=bvecs_rot, mean_centering=False)
        x[j, ] = x_rot

    return x


def rotateByMatrix(vectorsToRotate,rotationMatrix,rotationCenterVector = np.array([0,0,0])):    
    return np.dot(rotationMatrix,vectorsToRotate - rotationCenterVector[:,None]) + rotationCenterVector[:,None]


def interpolatePartialDWIVolume(dwi, centerPosition, x_,y_,z_, state):
    '''
    interpolate a dwi volume at some center position and provided spatial extent
    '''
    #print("rot " + str(rotations[j,].shape))
    szDWI = dwi.shape
    coordVecs = np.vstack(np.meshgrid(x_,y_,z_, indexing='ij')).reshape(3,-1).T + centerPosition   
    x = np.zeros([state.dim[0],state.dim[1],state.dim[2],szDWI[-1]])
    
    for i in range(0,szDWI[-1]):
        x[:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],coordVecs)[0], [state.dim[0],state.dim[1],state.dim[2]])
    return x


def visSphere(sphere):
    '''
    visualize sphere
    '''
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.point(sphere.vertices, window.colors.red, point_radius=0.05))
    window.show(ren)

    
def visTwoSetsOfStreamlines(streamlines,streamlines2, volume, vol_slice_idx = 40, vol_slice_idx2 = 40):
    '''
    visualize two sets of streamlines with different colors using vtk
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

        streamlines_actor = actor.line(streamlines, np.ones([len(streamlines)]))  # red
        streamlines_actor2 = actor.line(streamlines2, 0.5 * np.ones([len(streamlines2)])) # green

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
    
    
def visStreamlines(streamlines, volume=None, vol_slice_idx = 40, vol_slice_idx2 = 40):
    '''
    visualize streamline using vtk
    '''
    # Prepare the display objects.
    #color = line_colors(streamlines)
    
    print("Lsl:" + str(len(streamlines)))
    
    if window.have_vtk:
        r = window.Renderer()
        
        if volume is not None:
            vol_actor = actor.slicer(volume)
            vol_actor.display(y=vol_slice_idx)
            vol_actor2 = vol_actor.copy()
            vol_actor2.display(z=vol_slice_idx2)
            r.add(vol_actor)
            r.add(vol_actor2)
        
        #streamlines_actor = actor.line(streamlines, line_colors(streamlines))
        streamlines_actor = actor.line(streamlines)

        # Create the 3D display.
        
        r.add(streamlines_actor)

        window.show(r)
    else:
        print('we need VTK for proper visualisation of our fibres.')
        
        
def curvatureOfStreamlineExceedsThreshold(sl, theta):
    max_theta = np.deg2rad(theta)

    ls = sl[2:,:] - sl[1:-1,:]
    bls = sl[1:-1, :] - sl[0:-2,:]

    lsn = np.sqrt(np.sum(ls ** 2, axis = 1))
    ls = ls / lsn[:,None]

    blsn = np.sqrt(np.sum(bls ** 2, axis = 1))
    bls = bls / blsn[:,None]

    #angle1 = np.sum(np.multiply(ls, bls), axis = 1)
    #maxAngle = np.arccos(np.max(abs(angle1)))
    
    angles2 = np.arccos(np.sum(np.multiply(ls, bls), axis = 1))
    
    maxAngle = np.max(abs(angles2))
    
    #if(maxAngle > np.pi):
    #    maxAngle -= np.pi
        
    #if(maxAngle < -np.pi):
     #   maxAngle += np.pi
        
    #maxAngle = np.arccos(maxAngle)
    
    return maxAngle > max_theta or math.isnan(np.max(angles2))
        

def getMaxCurvatureOfStreamline(sl):
    ls = sl[2:,:] - sl[1:-1,:]
    bls = sl[1:-1, :] - sl[0:-2,:]

    lsn = np.sqrt(np.sum(ls ** 2, axis = 1))
    ls = ls / lsn[:,None]

    blsn = np.sqrt(np.sum(bls ** 2, axis = 1))
    bls = bls / blsn[:,None]

    angles2 = np.arccos(np.sum(np.multiply(ls, bls), axis = 1))
    return np.max(angles2)
    
    
def getCurvaturesForEachStreamline(streamlines):
    
    return [getMaxCurvatureOfStreamline(x) for x in streamlines]

def filterStreamlinesByCurvature(streamlines, maximumCurvature = 20):
    '''
    removes streamlines whose curvature is larger than maximumCurvature (in degree)
    '''
       
    return [ x for x in streamlines if curvatureOfStreamlineExceedsThreshold(x, maximumCurvature) == False]
        

        
def filterStreamlinesByLength(streamlines, minimumLength = 80):
    '''
    removes streamlines that are shorter than minimumLength (in mm)
    '''
    return [x for x in streamlines if metrics.length(x) > minimumLength]


def filterStreamlinesByMaxLength(streamlines, maxLength = 200):
    '''
    removes streamlines that are shorter than minimumLength (in mm)
    '''
    return [x for x in streamlines if metrics.length(x) < maxLength]

def generateTrainingData(streamlines, dwi, affine, state, generateRandomData = False):
    '''

    '''
    sfa = np.asarray(streamlines)
    dx,dy,dz,_ = dwi.shape
    dw = getSizeOfDataRepresentation(state)
    noNeighbours = 2*state.noCrossingFibres + 1
    sl_pos = sfa[0]
    noStreamlines = len(streamlines)
    
    # Build kd-tree of streamline positions. This significantly decreases subsequent lookup times.
    for streamlineIndex in range(1,noStreamlines):
        lengthStreamline = len(sfa[streamlineIndex])
        sl_pos = np.concatenate([sl_pos, sfa[streamlineIndex][0:lengthStreamline]], axis=0) # dont store absolute value but relative displacement
    
    #kdt = KDTree(sl_pos)
    
    #print('Processing streamlines')
   
    # define spacing of the 3D grid
    x_,y_,z_ = _getCoordinateGrid(state)
    
    # initialize our supervised training data
    #directionsToAdjacentStreamlines = np.zeros([len(sl_pos),2*noCrossings,3]) # likely next streamline directions
    directionToNextStreamlinePoint = np.zeros([len(sl_pos),3]) # next direction
    directionToPreviousStreamlinePoint = np.zeros([len(sl_pos),3]) # previous direction
    interpolatedDWISubvolume = np.zeros([len(sl_pos),state.dim[0],state.dim[1],state.dim[2],dw]) # interpolated dwi dataset for each streamline position
    #interpolatedDWISubvolumePast = np.zeros([len(sl_pos),noX,noY,noZ,dw]) # interpolated dwi dataset for each streamline position

    
    # projections
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    ctr = 0
    
    slOffset =  np.zeros([len(sl_pos)])
    
#    if(rotateTrainingData):
#        print('[I] the training data is being rotated wrt. each points tangent.')
    
    for streamlineIndex in range(0,noStreamlines):
        slOffset[streamlineIndex] = ctr
        if(((streamlineIndex) % 10000) == 0 and streamlineIndex > 0):
            print(str(streamlineIndex) + "/" + str(noStreamlines))
                
        streamlinevec = streamlines[streamlineIndex]
        noPoints = len(streamlinevec)
        streamlinevec_ijk = (M.dot(streamlinevec.T) + abc).T
        
        streamlinevec_all_next = streamlines[streamlineIndex][1:]

        dNextAll = np.concatenate(  (streamlinevec_all_next - streamlinevec[0:-1,], np.array([[0,0,0]])))
        dPrevAll = np.concatenate( (np.array([[0,0,0]]), -1 * dNextAll[0:-1,]) )
        
        rot = None

        if(state.rotateData):
            # reference orientation
            vv = state.getReferenceOrientation()
            
            # compute tangents
            tangents = streamlinevec_ijk[0:-1] - streamlinevec_ijk[1:] # tangents represents the tangents starting from the 2nd streamline position previousPosition - currentPosition

            # compute rotation matrices
            rot = np.zeros([noPoints,3,3])
            rot[0,:] = np.eye(3)

            for k in range(1,noPoints):
                R_2vect(rot[k, :], vector_orig=vv, vector_fin=tangents[k - 1,])
                #dNextAll[k,] = np.dot(rot[k, :], dNextAll[k,].T).T

                if(generateRandomData):
                    rot[k,:] += np.random.rand(3,3) - 0.5 # randomely rotate our data (and hope that there no other streamline coming from that direction)

                
        # interpolate
        interp_slv_ijk = interpolateDWIVolume(state,dwi,streamlinevec_ijk, rotations=rot, x_ = x_,y_ = y_,z_ = z_)
        interp_slv_ijk = projectIntoAppropriateSpace(state, interp_slv_ijk)
        #print(str(interp_slv_ijk))

        if(generateRandomData):
            dNextAll = np.zeros([noPoints,3])
            dPrevAll = np.zeros([noPoints,3])
        
        streamlinevecPast_ijk = (M.dot(streamlinevec.T) + abc).T
        
        interpolatedDWISubvolume[ctr:ctr+noPoints,] = interp_slv_ijk
        #interpolatedDWISubvolumePast[ctr+1:ctr+noPoints,] = interp_slv_ijk[0:-1,]
        directionToNextStreamlinePoint[ctr:ctr+noPoints,] = dNextAll
        directionToPreviousStreamlinePoint[ctr:ctr+noPoints,] = dPrevAll
        
#        interpolatedDWISubvolumePastAggregated[ctr,] = interp_slv_ijk[0,]
        
#        for j in range(1,noPoints):
#            interpolatedDWISubvolumePastAggregated[ctr+j,] = np.mean(interp_slv_ijk[0:j,], axis = 0)
        
        ctr += noPoints
        
    if(state.unitTangent):
        directionToNextStreamlinePoint = np.nan_to_num(directionToNextStreamlinePoint // np.sqrt(np.sum(directionToNextStreamlinePoint ** 2, axis = 1))) # unit vector   
        directionToPreviousStreamlinePoint = np.nan_to_num(directionToPreviousStreamlinePoint // np.sqrt(np.sum(directionToPreviousStreamlinePoint ** 2, axis = 1))) # unit vector

    return interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint #, interpolatedDWISubvolumePast

def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
