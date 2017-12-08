#!/bin/bash
wget https://www.dropbox.com/s/vxt3nvoihwopmv7/best.hdf5?dl=1
mv best.hdf5?dl=1 best.hdf5
python3 test.py $1 $2
