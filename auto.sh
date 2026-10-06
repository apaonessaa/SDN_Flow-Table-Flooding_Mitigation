#!/bin/bash

cd feature_collector
sudo python3 simulation.py
./build_datasets.sh

cd ..
cd training
python3 mfr_main.py
cd ..

