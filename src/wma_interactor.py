import os
import xml.etree.ElementTree
import src.dwi_tools as dwi_tools
import glob
import SimpleITK as sitk
import numpy as np
from dipy.tracking.life import transform_streamlines

def loadStreamlinesFromMRML(pMRML):
    e = xml.etree.ElementTree.parse(pMRML).getroot()
    pathSep = os.path.dirname(pMRML)

    streamlines = []

    for atype in e.findall('FiberBundleStorage'):
        streamlines.extend(dwi_tools.loadVTKstreamlines(pathSep + os.sep + atype.get('fileName'), reportProgress=False))
        
    return streamlines


def computeNumberOfStreamlinesForEachCluster(pMRML):
    e = xml.etree.ElementTree.parse(pMRML).getroot()
    pathSep = os.path.dirname(pMRML)

    my_dict = {}

    for atype in e.findall('FiberBundleStorage'):
        noFibres = len(dwi_tools.loadVTKstreamlines(pathSep + os.sep + atype.get('fileName'), reportProgress=False))
        cID = atype.get('fileName').replace('.vtp','')
        my_dict[cID] = noFibres
        print(cID + " -> " + str(noFibres))
        
    return my_dict


def computeBinaryMaskForEachCluster(pMRML, shapeVolume, affine, affineRough, affineFine):
    e = xml.etree.ElementTree.parse(pMRML).getroot()
    pathSep = os.path.dirname(pMRML)

    my_dict = {}

    for atype in e.findall('FiberBundleStorage'):
        streamlines_atlas = dwi_tools.loadVTKstreamlines(pathSep + os.sep + atype.get('fileName'), reportProgress=False)
        cID = atype.get('fileName').replace('.vtp', '')
        print(cID)
        new_voxel_data = np.zeros(shapeVolume)
        sl2 = transform_streamlines(streamlines_atlas, np.linalg.inv(affineFine))  # fine transform: ATLAS -> ATLAS_ROUGH
        sl_final = transform_streamlines(sl2, np.linalg.inv(affineRough))  # rough transform: ATLAS_ROUGH -> RAS
        streamlines_ijk = transform_streamlines(sl_final, np.linalg.inv(affine))  # RAS -> IJK

        for sl in streamlines_ijk:
            for streamlinePoint in sl:
                sl_r = np.rint(streamlinePoint).astype(np.int32)
                new_voxel_data[sl_r[0], sl_r[1], sl_r[2]] = 1
        my_dict[cID] = new_voxel_data


    return my_dict

def computeClusterwiseIOU(gt_tracts_dict, pred_tracts_dic):
    my_dict = {}

    for k in gt_tracts_dict:
        my_dict[k] = tract_intersection_over_union(gt_tracts_dict[k],pred_tracts_dic[k])
        print(k + " - > " + str(my_dict[k][0]))

    return my_dict

def tract_intersection_over_union(gt_tracts, pred_tracts):
    gt_tracts = gt_tracts.astype(np.int32)
    pred_tracts = pred_tracts.astype(np.int32)

    eps = 1e-6 # minor term to prevent division by 0 issues

    intersection = (pred_tracts & gt_tracts).sum(( 0,1, 2))
    union = (pred_tracts | gt_tracts).sum((0, 1, 2))

    iou = (intersection + eps) / (union + eps)

    # return the intersection over union value
    return iou, intersection, union

def loadAllMRMLStreamlinesFromDirectory(pDirectory):
    streamlines = []
    for file in glob.glob(pDirectory + os.sep + "*.mrml"):
        print(file)
        streamlines.extend(loadStreamlinesFromMRML(file))
    return streamlines

def loadTransform(pTrans):
    t = sitk.ReadTransform(pTrans)
    myVersorTransform = sitk.AffineTransform(t)
    part1 = np.vstack( (np.array(myVersorTransform.GetMatrix()).reshape([3,3]).T, myVersorTransform.GetTranslation()) ).T
    affFine = np.vstack( (part1, [0,0,0,1]))
    return affFine