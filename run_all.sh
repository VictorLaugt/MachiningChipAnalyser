set -e

inputs=()
inputs+=("vertical")
inputs+=("vertical_flip")
inputs+=("bande_noire")
inputs+=("diagonal")
inputs+=("diagonal2")
inputs+=("diagonal3")
inputs+=("diagonal_flip")
inputs+=("diagonal_video.avi")

rm -rf outputs_*
for input_imgs in "${inputs[@]}"; do
    cmd="python3 ChipAnalyser -i \"imgs/$input_imgs\" -o \"outputs_$input_imgs\" -r"
    echo $cmd
    eval $cmd
    echo
done
