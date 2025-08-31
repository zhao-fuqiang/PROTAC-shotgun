#!/bin/bash

#for current_step in {0..99}
for current_step in {0..9}
do

    echo "current_step is:"
    echo ${current_step}
    sed -i "s/cluster_X/cluster_${current_step}/g" config.py
    python main.py
    sed -i "s/cluster_${current_step}/cluster_X/g" config.py
    
    sleep 5s

done

