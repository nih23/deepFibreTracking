#!/bin/bash
echo "Starting with LSTMs"

python 03_ismrm2015_fa.py --faMask -threshold 0.05 &
python 03_ismrm2015_fa.py --faMask -threshold 0.1  &
python 03_ismrm2015_fa.py --faMask -threshold 0.15 &
python 03_ismrm2015_fa.py --faMask -threshold 0.2  &
python 03_ismrm2015_fa.py --faMask -threshold 0.25 &
python 03_ismrm2015_fa.py --faMask -threshold 0.3  &
python 03_ismrm2015_fa.py --faMask -threshold 0.35 &
python 03_ismrm2015_fa.py --faMask -threshold 0.4  &
python 03_ismrm2015_fa.py --faMask -threshold 0.45 &
python 03_ismrm2015_fa.py --faMask -threshold 0.5  &
python 03_ismrm2015_fa.py --faMask -threshold 0.55 &
python 03_ismrm2015_fa.py --faMask -threshold 0.6  &
python 03_ismrm2015_fa.py --faMask -threshold 0.65 &
python 03_ismrm2015_fa.py --faMask -threshold 0.7  &
python 03_ismrm2015_fa.py --faMask -threshold 0.75 &
python 03_ismrm2015_fa.py --faMask -threshold 0.8  &
python 03_ismrm2015_fa.py --faMask -threshold 0.85 &
python 03_ismrm2015_fa.py --faMask -threshold 0.9  &
python 03_ismrm2015_fa.py --faMask -threshold 0.95 &

wait

echo "Starting with MLPs"
python 03_ismrm2015_fa.py --faMask -threshold 0.05 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.1  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.15 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.2  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.25 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.3  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.35 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.4  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.45 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.5  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.55 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.6  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.65 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.7  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.75 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.8  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.85 --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.9  --useMLP &
python 03_ismrm2015_fa.py --faMask -threshold 0.95 --useMLP &

wait

echo "Finished"
