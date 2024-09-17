#!/bin/bash
eval "$(conda shell.bash hook)"

conda create -y -n deploy-test-3.8 python=3.8
conda create -y -n deploy-test-3.9 python=3.9
conda create -y -n deploy-test-3.10 python=3.10
conda create -y -n deploy-test-3.11 python=3.11
conda create -y -n deploy-test-3.12 python=3.12

results="deploy-test-results.txt"
input_images=("bande_noire" "diagonal" "diagonal2" "diagonal3" "diagonal_flip" "diagonal_video.avi" "vertical" "vertical_flip")
environments=("deploy-test-3.8" "deploy-test-3.9" "deploy-test-3.10" "deploy-test-3.11" "deploy-test-3.12")

cat /dev/null > "$results"

for env_name in "${environments[@]}"
do
    printf "\n\n\n==========\DEPLOYING ON $env_name\n==========\n\n"
    printf "\n$env_name:\n" >> "$results"
    conda activate "$env_name"

    # install the dependencies
    echo "INSTALLING DEPENDENCIES"
    pip install -q -r requirements.txt
    if [ $? -ne 0 ]; then
        printf "* install: FAIL\n" >> "$results"
        conda deactivate
        continue
    else
        printf "* install: SUCCESS\n" >> "$results"
    fi

    # run without the rendering option -r
    for inp in "${input_images[@]}"
    do
        rm -rf outputs
        cmd="python3 ChipAnalyser -i \"imgs/$inp\" -o outputs"
        echo "RUNNING: $cmd"
        eval "$cmd"
        if [ $? -ne 0 ]; then
            printf "* classic : FAIL (${cmd})\n" >> "$results"
            break -1
        fi
    done
    if [ $? -eq 0 ]; then
        printf "* classic : SUCCESS\n" >> "$results"
    fi

    # run with the rendering option -r
    for inp in "${input_images[@]}"
    do
        rm -rf outputs
        cmd="python3 ChipAnalyser -i \"imgs/$inp\" -o outputs -r"
        echo "RUNNING: $cmd"
        eval "$cmd"
        if [ $? -ne 0 ]; then
            printf "* rendering : FAIL (${cmd})\n" >> "$results"
            break -1
        fi
    done
    if [ $? -eq 0 ]; then
        printf "* rendering : SUCCESS\n" >> "$results"
    fi

    conda deactivate
done

conda env remove -y -n deploy-test-3.8
conda env remove -y -n deploy-test-3.9
conda env remove -y -n deploy-test-3.10
conda env remove -y -n deploy-test-3.11
conda env remove -y -n deploy-test-3.12
