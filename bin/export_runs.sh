# Enable the "errexit" option which causes the script to exit immediately if any command fails
set -e

# Define the model sizes
model_sizes=("12b" "6.9b" "1.4b" "410m" "160m" "70m")


# Iterate over each model size and run the export_runs.py script
for size in "${model_sizes[@]}"; do
    echo Running export_runs.py with model size $size
    poetry run python ./scripts/export_runs.py --model_size "$size"
done
