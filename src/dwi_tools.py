from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 

from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines, length
from dipy.tracking import metrics

import numpy as np
import warnings

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
        r.add(vol_actor)
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
    print("using " + str([noX,noY,noZ]) + "px interpolation grid with coordinateScaling " + str(coordinateScaling))
    x_ = coordinateScaling * np.linspace(-4., 4, noX)
    y_ = coordinateScaling * np.linspace(-4., 4., noY)
    z_ = coordinateScaling * np.linspace(-4., 4., noZ)
    return x_,y_,z_

def generatePredictionNetworkTrainingDataFromStreamlines(streamlines, dwi, rec_level_sphere = 3, noX=3, noY=3,noZ=3,coordinateScaling = 1, noCrossings = 3, distToNeighbours = 0.5, maximumNumberOfNearbyStreamlinePoints = 3, affine = np.eye(4,4)):
    '''
    
    '''
    # project seed positions into DWI space
    
    
    sfa = np.asarray(streamlines)
    #np.random.shuffle(sfa)
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
    ctr = 0
    
    # define spacing of the 3D grid
    x_,y_,z_ = getCoordinateGrid(noX,noY,noZ,coordinateScaling)
    
    # initialize our supervised training data
    train_Y_1 = np.zeros([len(sl_pos),2*noCrossings,3]) # likely next streamline directions
    train_Y_2 = np.zeros([len(sl_pos),3]) # next streamline position
    train_X = np.zeros([len(sl_pos),noX,noY,noZ,dw]) # interpolated dwi dataset for each streamline position
    train_X_2 = np.zeros([len(sl_pos),3]) # actual position
    for streamlineIndex in range(0,len(sl_pos)):
        streamlinevec = sl_pos[streamlineIndex]
        streamlinevec_next = sl_pos[min((streamlineIndex+1,len(sl_pos)-1))]
        streamlinevec_prev = sl_pos[max((streamlineIndex-1,0))]
        d = np.sum( (streamlinevec - streamlinevec_next)**2)
        i = kdt.query_ball_point(streamlinevec,distToNeighbours) # acquire neighbours within a certain distance
        i = i[0:2*noCrossings] # limit length to the nearest neighbours
        n_slv = kdt.data[i] - streamlinevec # direction to nearest streamline position
        n_slv = np.unique(n_slv, axis = 0) # remove duplicats from our search query
        noPoints,dimPoints = n_slv.shape
        train_Y_1[ctr,0:noPoints,] = n_slv
        train_Y_2[ctr,] = streamlinevec_next - streamlinevec
        train_Y_2[ctr,] = np.nan_to_num(train_Y_2[ctr,] / np.sqrt(np.sum(train_Y_2[ctr,] ** 2))) # make unit vector
        train_X_2[ctr,] = streamlinevec_prev - streamlinevec
        train_X_2[ctr,] = np.nan_to_num(train_X_2[ctr,] / np.sqrt(np.sum(train_X_2[ctr,] ** 2))) # make unit vector
        
        
        if((ctr % 1000) == 0):
            print(str(ctr) + "/" + str(len(sl_pos)))
        coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + streamlinevec
        for i in range(0,dw):
            train_X[ctr,:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(dwi[:,:,:,i],coordVecs)[0], [noX,noY,noZ])
#        if(ctr<1):
#            print("Streamline: " + str(streamlinevec))
#            print("Streamline_next: " + str(streamlinevec_next))
#            print("dir_next: " + str(train_Y_2[ctr,]))
#            print('unit_dir_next: ' + str(train_Y_2[ctr,] / np.sqrt(np.sum(train_Y_2[ctr,] ** 2))))
#            print("streamline_neighbours: " + str(train_Y_1[ctr,0:noPoints,]))

        ctr += 1
        
    return train_X, train_X_2, train_Y_1, train_Y_2