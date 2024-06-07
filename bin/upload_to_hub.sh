# Define the start and end of the loop
start=0
end=143000
increment=1000
model_size=6.9b

# Loop from start to end with the given increment
for ((i=start; i<=end; i+=increment)); do
    
    # Define the step variable for the command
    echo *step$i.parquet
    
    # Run the huggingface-cli command
    huggingface-cli upload \
        pietrolesci/pythia-deduped-stats \
        ./outputs/multirun/$model_size/ \
        ./$model_size \
        --repo-type dataset \
        --include "*step$i.parquet"
done
