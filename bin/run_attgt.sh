# Enable the "errexit" option which causes the script to exit immediately if any command fails
set -e


# Define the model sizes
model_sizes=("12b" "6.9b" "1.4b" "410m" "160m" "70m")
metric=acc_seq  # sup_seq, avg_rank, entr_seq

# Iterate over each model size and run the export_runs.py script
for size in "${model_sizes[@]}"; do
        
    echo Estimating causal effect with bootstrap confidence intervals $size
    poetry run python ./scripts/compute_attgt.py --model_size $size --target_col $metric --bootstrap

done