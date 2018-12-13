from dipy.io.streamline import load_trk
import src.dwi_tools as dwi_tools
import argparse
from dipy.tracking.streamline import Streamlines

def main():
    parser = argparse.ArgumentParser(description='Tract Converter')
    parser.add_argument('-i', '--input', help='path to input streamline (tck)')
    parser.add_argument('-o', '--output', help='path to output streamline (vtk)')
    args = parser.parse_args()
    print(args)
    if(args.input is None):
        parser.print_help()
        return -1
    
    fname = args.input
    if(args.output is None):
        fname2 = fname.replace('.tck','.vtk')
    else:
        fname2 = args.output
    streams, hdr = load_trk(fname)
    streamlines = Streamlines(streams)
    dwi_tools.saveVTKstreamlines(streamlines=streamlines, pStreamlines=fname2)

if __name__ == "__main__":
    main()
