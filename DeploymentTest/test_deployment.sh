#!/bin/bash

TEST_NAME="$(basename $(realpath $0))"
TEST_NAME="${TEST_NAME%.*}"

DEPLOYMENT_TEST_DIR="$(dirname $(realpath $0))"
INPUT_DIR="$DEPLOYMENT_TEST_DIR/inputs"
OUTPUT_DIR="$DEPLOYMENT_TEST_DIR/outputs"
RESULT_FILE="$DEPLOYMENT_TEST_DIR/results.txt"

PROJECT_DIR="$(realpath $DEPLOYMENT_TEST_DIR/..)"
PROGRAM_DIR="$PROJECT_DIR/ChipAnalyser"

PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12")


function log {
    echo "<$TEST_NAME> $*"
}

function out {
    echo -e $* >> $RESULT_FILE
}

function debug {
    echo "<DEBUG> $*"
}

function enter_environment { local version=$1
    log "CREATING TEST VIRTUAL ENVIRONMENT $version"
    conda create -y -n "ChipAnalyser-deploy-test-$version" python=$version > /dev/null
    conda activate "ChipAnalyser-deploy-test-$version"
}

function exit_environment { local version=$1
    log "REMOVING TEST VIRTUAL ENVIRONMENT $version"
    while [[ "$CONDA_DEFAULT_ENV" != "" ]]; do conda deactivate; done
    conda env remove -y -n "ChipAnalyser-deploy-test-$version" > /dev/null
}

function install_dependencies {
    log "INSTALLING DEPENDENCIES"
    pip install -q -r "$PROJECT_DIR/requirements.txt" > /dev/null 2>&1
}

function test_command { local cmd=$1; local msg=$2
    eval "$cmd" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        out "$msg: SUCCESS"
    else
        out "$msg: FAIL"
    fi
}

function test_without_graphical_rendering {
    log "TESTING WITHOUT GRAPHICAL RENDERING"
    out "\twith rendering disabled:"
    find "$INPUT_DIR" -mindepth 1 -maxdepth 1 ! -name .gitkeep | while read -r input_path; do
        rm -rf "$OUTPUT_DIR"
        cmd="python \"$PROGRAM_DIR\" -i \"$input_path\" -o \"$OUTPUT_DIR\""
        test_command "$cmd" "\t\tusing input $(basename $input_path)"
    done
}

function test_with_graphical_rendering {
    log "TESTING WITH GRAPHICAL RENDERING"
    out "\twith rendering enabled:"
    find "$INPUT_DIR" -mindepth 1 -maxdepth 1 ! -name .gitkeep | while read -r input_path; do
        rm -rf "$OUTPUT_DIR"
        cmd="python \"$PROGRAM_DIR\" -i \"$input_path\" -o \"$OUTPUT_DIR\" -r"
        test_command "$cmd" "\t\tusing input $(basename $input_path)"
    done
}

eval "$(conda shell.bash hook)"
cat /dev/null > $RESULT_FILE
for version in "${PYTHON_VERSIONS[@]}"; do
    log "========= DEPLOYING ON PYTHON VERSION $version ========="
    out "\npython $version:"

    enter_environment $version

    install_dependencies
    if [ $? -eq 0 ]; then
        out "\tinstall: SUCCESS"
        test_without_graphical_rendering
        test_with_graphical_rendering
    else
        out "\tinstall: FAIL"
    fi

    exit_environment $version
done

log "RESULTS WRITTEN IN THE FILE $RESULT_FILE"
cat $RESULT_FILE
