import os
import xml.etree.ElementTree
import src.dwi_tools as dwi_tools
import glob

def loadStreamlinesFromMRML(pMRML):
    e = xml.etree.ElementTree.parse(pMRML).getroot()
    pathSep = os.path.dirname(pMRML)

    streamlines = []

    for atype in e.findall('FiberBundleStorage'):
        streamlines.extend(dwi_tools.loadVTKstreamlines(pathSep + os.sep + atype.get('fileName'), reportProgress=False))
        
    return streamlines


def loadAllMRMLStreamlinesFromDirectory(pDirectory):
    streamlines = []
    for file in glob.glob(pDirectory + os.sep + "*.mrml"):
        print(file)
        streamlines.extend(loadStreamlinesFromMRML(file))
    return streamlines