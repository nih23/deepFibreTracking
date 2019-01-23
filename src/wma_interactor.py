import os
import xml.etree.ElementTree
import src.dwi_tools as dwi_tools
import glob
import SimpleITK as sitk
import numpy as np

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