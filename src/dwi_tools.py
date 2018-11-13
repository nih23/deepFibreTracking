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

import vtk
from scipy.interpolate import griddata

from dipy.data import get_sphere


def loadVTKstreamlines(pStreamlines):
    
    if(pStreamlines[-4:] == '.vtk'):
        print('PDR')
        reader = vtk.vtkPolyDataReader()
    elif (pStreamlines[-4:] == '.vtp'):
        print('xmlPDR')
        reader = vtk.vtkXMLPolyDataReader()
    
    reader.SetFileName(pStreamlines)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    polydata = reader.GetOutput()
    streamlines = []
    
    for i in range(polydata.GetNumberOfCells()):
    #for i in range(0,100):
        if((i % 10000) == 0):
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
        if((i % 1000) == 0):
                print(str(i) + "/" + str(len(streamlines)))
        
        
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(len(streamlines[i]))
        for j in range(0,len(streamlines[i])):
            points.InsertNextPoint(np.nan_to_num(streamlines[i][j]))
            linePts = line.GetPointIds()
            linePts.SetId(j,ptCtr)
            
            ptCtr += 1
            
        lines.InsertNextCell(line)
                   
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
        if((i % 1000) == 0):
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
    data_sh = get_spherical_harmonics_coefficients(dwi, b0, bvals, bvecs, sh_order=sh_order, smooth=smooth)

    sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
    
    if directions is not None:
        sphere = Sphere(xyz=bvecs[1:])

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


def get_spherical_harmonics_coefficients(dwi, b0, bvals, bvecs, sh_order=8, smooth=0.006):
    """ Compute coefficients of the spherical harmonics basis.
    adapted from: https://github.com/ppoulin91/learn2track/blob/miccai2017_submission/learn2track/neurotools.py
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

    # normalize by the b0.
    dwi_weights = normalize_dwi(dwi_weights, b0)

    # Assuming all directions lie on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs) ### CHANGED 11/09/2018 FROM HEMISPHERE TO SPHERE AS OUR HCP DATA DOESNT JUST HAVE GRADIENTS ON A SPHERE

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


def interpolateDWIVolume(dwi, positions, x_,y_,z_, noX = 8, noY = 8, noZ = 8):
    
    szDWI = dwi.shape
    noPositions = len(positions)
    #print('pos: ' + str(positions.shape))
    start_time = time.time()
    cvF = np.ones([noPositions*noX*noY*noZ,3])
    #cvF = (np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + positions[0,])
    #print('cvf: ' + str(cvF.shape))
    noElem = noX * noZ * noY
    grid = np.array(np.meshgrid(x_,y_,z_)).reshape(3,-1).T
    for j in range(0,noPositions):
        coordVecs = grid + positions[j,]
        
        #print('coordVecs: ' + str(coordVecs.shape))
        il = j * noElem
        ir = (j+1) * noElem 
        cvF[il:ir] = coordVecs
        #cvF = np.concatenate([cvF, coordVecs])
    x = np.zeros([noPositions,noX,noY,noZ,szDWI[-1]])
    
    for i in range(0,szDWI[-1]):
        x[:,:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],cvF)[0], [noPositions,noX,noY,noZ])
        
    return x


def interpolatePartialDWIVolume(dwi, centerPosition, x_,y_,z_, noX = 8, noY = 8, noZ = 8):
    '''
    interpolate a dwi volume at some center position and provided spatial extent
    '''
    szDWI = dwi.shape
    coordVecs = np.vstack(np.meshgrid(x_,y_,z_, indexing='ij')).reshape(3,-1).T + centerPosition   
    x = np.zeros([noX,noY,noZ,szDWI[-1]])
    
    for i in range(0,szDWI[-1]):
        x[:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],coordVecs)[0], [noX,noY,noZ])
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
    
    
def visStreamlines(streamlines, volume, vol_slice_idx = 40, vol_slice_idx2 = 40):
    '''
    visualize streamline using vtk
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
        
        #streamlines_actor = actor.line(streamlines, np.ones([len(streamlines)]))  # red

        # Create the 3D display.
        r = window.Renderer()
        r.add(streamlines_actor)
        r.add(vol_actor)
        r.add(vol_actor2)
        #window.record(r, n_frames=1, out_path='deterministic.png', size=(800, 800))
        window.show(r)
    else:
        print('we need VTK for proper visualisation of our fibres.')

        
def filterStreamlinesByLength(streamlines, minimumLength = 80):
    '''
    removes streamlines that are shorter than minimumLength (in mm)
    '''
    return [x for x in streamlines if metrics.length(x) > minimumLength]


def _getCoordinateGrid(noX,noY,noZ,coordinateScaling):
    '''
    generate a grid of provided spatial extent which is used to interpolate data
    '''
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

def generateTrainingData(streamlines, dwi, affine, rec_level_sphere = 3, noX=3, noY=3,noZ=3,coordinateScaling = 1, noCrossings = 3, distToNeighbours = 0.5, maximumNumberOfNearbyStreamlinePoints = 3, step = 1):
    '''
    
    '''
    sfa = np.asarray(streamlines)
    dx,dy,dz,dw = dwi.shape
    noNeighbours = 2*noCrossings + 1
    sl_pos = sfa[0]
    noStreamlines = len(streamlines)
    
    # Build kd-tree of streamline positions. This significantly decreases subsequent lookup times.
    for streamlineIndex in range(1,noStreamlines):
        lengthStreamline = len(sfa[streamlineIndex])
        sl_pos = np.concatenate([sl_pos, sfa[streamlineIndex][0:lengthStreamline]], axis=0) # dont store absolute value but relative displacement
    
    #kdt = KDTree(sl_pos)
    
    print('Processing streamlines')
   
    # define spacing of the 3D grid
    x_,y_,z_ = _getCoordinateGrid(noX,noY,noZ,coordinateScaling)
    
    # initialize our supervised training data
    #directionsToAdjacentStreamlines = np.zeros([len(sl_pos),2*noCrossings,3]) # likely next streamline directions
    directionToNextStreamlinePoint = np.zeros([len(sl_pos),3]) # next direction
    directionToPreviousStreamlinePoint = np.zeros([len(sl_pos),3]) # previous direction
    interpolatedDWISubvolume = np.zeros([len(sl_pos),noX,noY,noZ,dw]) # interpolated dwi dataset for each streamline position
    
    # projections
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    ctr = 0
    for streamlineIndex in range(0,noStreamlines):
        if((streamlineIndex % 100) == 0):
                print(str(streamlineIndex) + "/" + str(noStreamlines))
        lengthStreamline = len(streamlines[streamlineIndex])
        for streamlinePoint in range(0,lengthStreamline,step):

            streamlinevec = streamlines[streamlineIndex][streamlinePoint]
            streamlinevec_next = streamlines[streamlineIndex][min(streamlinePoint+1,lengthStreamline-1)]
            streamlinevec_prev = streamlines[streamlineIndex][max(streamlinePoint-1,0)]


            #streamlinevec_next = sl_pos[min((streamlineIndex+1,len(sl_pos)-1))]
            #streamlinevec_prev = sl_pos[max((streamlineIndex-1,0))]

            ### COMMENTED OUT AS ITS CURRENTLY NOT IN USE AND SLOWS DOWN DATASET CREATION
            #d = np.sum( (streamlinevec - streamlinevec_next)**2)
            #i = kdt.query_ball_point(streamlinevec,distToNeighbours) # acquire neighbours within a certain distance
            #i = i[0:2*noCrossings] # limit length to the nearest neighbours
            #n_slv = kdt.data[i] - streamlinevec # direction to nearest streamline position
            #n_slv = np.unique(n_slv, axis = 0) # remove duplicats from our search query
            #noPoints,dimPoints = n_slv.shape
            #directionsToAdjacentStreamlines[streamlineIndex,0:noPoints,] = n_slv

            directionToNextStreamlinePoint[ctr,] = streamlinevec_next - streamlinevec
            directionToNextStreamlinePoint[ctr,] = np.nan_to_num(directionToNextStreamlinePoint[ctr,] / np.sqrt(np.sum(directionToNextStreamlinePoint[ctr,] ** 2))) # unit vector

            directionToPreviousStreamlinePoint[ctr,] = streamlinevec_prev - streamlinevec
            directionToPreviousStreamlinePoint[ctr,] = np.nan_to_num(directionToPreviousStreamlinePoint[ctr,] / np.sqrt(np.sum(directionToPreviousStreamlinePoint[ctr,] ** 2))) # unit vector

            #DEBUG project from RAS to image coordinate system
            curStreamlinePos_ras = streamlinevec
            curStreamlinePos_ras = curStreamlinePos_ras[:,None]
            curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T

            interpolatedDWISubvolume[ctr,] = interpolatePartialDWIVolume(dwi,curStreamlinePos_ijk, noX = noX, noY = noY, noZ = noZ,x_ = x_,y_ = y_,z_ = z_)
            
            ctr += 1

    print("-> " + str(ctr))
    return interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint


def generate2DUnrolledTrainingData(streamlines, dwi, affine, resamplingSphere, coordinateScaling = 1, noCrossings = 3, distToNeighbours = 0.5, maximumNumberOfNearbyStreamlinePoints = 3, step = 0.6):
    '''
    
    '''
    sfa = np.asarray(streamlines)
    dx,dy,dz,dw = dwi.shape
    noNeighbours = 2*noCrossings + 1
    sl_pos = sfa[0]
    noStreamlines = len(streamlines)
    
    # Build kd-tree of streamline positions. This significantly decreases subsequent lookup times.
    for streamlineIndex in range(1,noStreamlines):
        lengthStreamline = len(sfa[streamlineIndex])
        sl_pos = np.concatenate([sl_pos, sfa[streamlineIndex][0:lengthStreamline]], axis=0) # dont store absolute value but relative displacement
    
    #kdt = KDTree(sl_pos)
    
    print('Processing streamlines')
   
    # define spacing of the 3D grid
    x_,y_,z_ = _getCoordinateGrid(1,1,1,coordinateScaling)
    
    # initialize our supervised training data
    #directionsToAdjacentStreamlines = np.zeros([len(sl_pos),2*noCrossings,3]) # likely next streamline directions
    directionToNextStreamlinePoint = np.zeros([len(sl_pos),3]) # next direction
    directionToPreviousStreamlinePoint = np.zeros([len(sl_pos),3]) # previous direction
    interpolatedDWISubvolume = np.zeros([len(sl_pos),24,24]) # interpolated dwi dataset for each streamline position
    
    # projections
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    ctr = 0
    for streamlineIndex in range(0,noStreamlines):
        if((streamlineIndex % 100) == 0):
                print(str(streamlineIndex) + "/" + str(noStreamlines))
        lengthStreamline = len(streamlines[streamlineIndex])
        for streamlinePoint in range(0,lengthStreamline,step):

            streamlinevec = streamlines[streamlineIndex][streamlinePoint]
            streamlinevec_next = streamlines[streamlineIndex][min(streamlinePoint+1,lengthStreamline-1)]
            streamlinevec_prev = streamlines[streamlineIndex][max(streamlinePoint-1,0)]


            #streamlinevec_next = sl_pos[min((streamlineIndex+1,len(sl_pos)-1))]
            #streamlinevec_prev = sl_pos[max((streamlineIndex-1,0))]

            ### COMMENTED OUT AS ITS CURRENTLY NOT IN USE AND SLOWS DOWN DATASET CREATION
            #d = np.sum( (streamlinevec - streamlinevec_next)**2)
            #i = kdt.query_ball_point(streamlinevec,distToNeighbours) # acquire neighbours within a certain distance
            #i = i[0:2*noCrossings] # limit length to the nearest neighbours
            #n_slv = kdt.data[i] - streamlinevec # direction to nearest streamline position
            #n_slv = np.unique(n_slv, axis = 0) # remove duplicats from our search query
            #noPoints,dimPoints = n_slv.shape
            #directionsToAdjacentStreamlines[streamlineIndex,0:noPoints,] = n_slv

            directionToNextStreamlinePoint[ctr,] = streamlinevec_next - streamlinevec
            directionToNextStreamlinePoint[ctr,] = np.nan_to_num(directionToNextStreamlinePoint[ctr,] / np.sqrt(np.sum(directionToNextStreamlinePoint[ctr,] ** 2))) # unit vector

            directionToPreviousStreamlinePoint[ctr,] = streamlinevec_prev - streamlinevec
            directionToPreviousStreamlinePoint[ctr,] = np.nan_to_num(directionToPreviousStreamlinePoint[ctr,] / np.sqrt(np.sum(directionToPreviousStreamlinePoint[ctr,] ** 2))) # unit vector

            #DEBUG project from RAS to image coordinate system
            curStreamlinePos_ras = streamlinevec
            curStreamlinePos_ras = curStreamlinePos_ras[:,None]
            curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T

            interpolatedDWISubvolume[ctr,] = projectGradientsOntoGrid(resamplingSphere, interpolatePartialDWIVolume(dwi,curStreamlinePos_ijk, noX = 1, noY = 1, noZ = 1,x_ = x_,y_ = y_,z_ = z_))
            
            ctr += 1

    print("-> " + str(ctr))
    return interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint
