#!/bin/bash
echo "Starting with LSTMs"

python 03_ismrm2015_fa.py --faMask -fa 0.05 &
python 03_ismrm2015_fa.py --faMask -fa 0.1  &
python 03_ismrm2015_fa.py --faMask -fa 0.15 &
python 03_ismrm2015_fa.py --faMask -fa 0.2  &

wait

echo "Starting with MLPs"
python 03_ismrm2015_fa.py --faMask -fa 0.05 --useMLP &
python 03_ismrm2015_fa.py --faMask -fa 0.1  --useMLP &
python 03_ismrm2015_fa.py --faMask -fa 0.15 --useMLP &
python 03_ismrm2015_fa.py --faMask -fa 0.2  --useMLP &

wait

echo "Finished"
