#!/bin/bash
PROJECT_DIR=$(dirname $(realpath $0))

printf "===== line count =====\n"
find . -name '*.py' ! -path '*/tests/*' ! -path '*/__pycache__/*' | xargs wc -l

printf "\n===== importation graph =====\n"
which inclusionmap > /dev/null
if [ $? == 0 ]; then
    inclusionmap ${PROJECT_DIR}/ChipAnalyser -s -l python -g '.*test.*' --display-algorithm dot
else
    echo "Should install InclusionMap to generate the graph: pip install InclusionMap"
fi
