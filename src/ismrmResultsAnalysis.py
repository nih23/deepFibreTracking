from dipy.io.streamline import load_trk
import argparse
import csv
import glob
import os
import src.dwi_tools as dwi_tools

def main():
    parser = argparse.ArgumentParser(description='Tract Converter')
    parser.add_argument('IBdir', help='path to directory with invalid bundles (tck)')
    args = parser.parse_args()
    print(args)

    pDirectory = args.IBdir
    pResults = 'results.csv'

    csvfile = open(pResults, 'a')

    spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)


    for file in glob.glob(pDirectory + os.sep + "*_IB*.tck"):

        streams, hdr = load_trk(file)
        fname = os.path.split(file)[-1]

        print(fname)
        spamwriter.writerow([str(fname)] + [str(len(streams))])

        dwi_tools.saveVTKstreamlines(streamlines=streams, pStreamlines=file.replace('.tck','.vtk'))

    csvfile.close()

if __name__ == "__main__":
    main()