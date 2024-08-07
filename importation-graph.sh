#!/bin/bash
PROJECT_DIR=$(dirname $(realpath $0))

which inclusionmap
if [ $? == 0 ]; then
    inclusionmap ${PROJECT_DIR}/ChipAnalyser -l python --display-algorithm dot -s -i tests
else
    echo "Should install InclusionMap to generate the graph: pip install InclusionMap"
fi
