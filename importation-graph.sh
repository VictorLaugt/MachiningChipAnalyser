#!/bin/bash
PROJECT_DIR=$(dirname $(realpath $0))

eval "$(conda shell.bash hook)"
conda activate base

which inclusionmap
if [ $? == 0 ]; then
    conda activate base
    inclusionmap ${PROJECT_DIR}/ChipAnalyser -l python --display-algorithm dot -s -i tests
    conda deactivate
else
    echo "Should install InclusionMap to generate the graph: pip install InclusionMap"
fi
