# Enable the "errexit" option which causes the script to exit immediately if any command fails
set -e


# We follow the instructions reported in the [pythia](https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset)
data_path="./data/pile-deduped-train"

echo Downloading the Pile training set and saving it into $data_path
mkdir -p $data_path

echo Downloading shards from HuggingFace
poetry run ./scripts/download_pile.py --data_path $data_path

echo Creating a memory mapped dataset
poetry run python utils/unshard_mmap.py \
    --num_shards 21 \
    --input_file $data_path/pile-deduped-preshuffled/document-00000-of-00020.bin \
    --output_dir $data_path

echo Copy document.idx file to $data_path
cp $data_path/pile-deduped-preshuffled/document.idx $data_path


# Optionally process the validation set
file="./data/pile-deduped-validation/raw/pile_val.jsonl"

if [ -e "$file" ]; then
    echo "Validation file $file exists. Processing the Pile validation split"
    poetry run ./scripts/process_pile_validation.py
fi