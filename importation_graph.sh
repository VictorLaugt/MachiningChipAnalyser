#!/bin/bash
PROJECT_DIR=$(dirname $(realpath $0))

printf "===== line count =====\n"
find . -name '*.py' ! -path '*/tests/*' ! -path '*/__pycache__/*' | xargs wc -l

printf "\n===== importation graph =====\n"
which inclusionmap > /dev/null
if [ $? == 0 ]; then
    inclusionmap ${PROJECT_DIR}/ChipAnalyser -l python --display-algorithm dot -s -i tests
else
    echo "Should install InclusionMap to generate the graph: pip install InclusionMap"
fi
