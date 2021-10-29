#!/bin/bash

#for dist_config in banana half_banana funnel cauchy
for dist_config in funnel cauchy
do
    echo "Dist ${dist_config}"
    python experiments/complex_geometry.py \
        configs/complex_geom.yaml \
        --dist_config configs/dists/${dist_config}.yaml
done
