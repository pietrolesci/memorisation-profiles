# Enable the "errexit" option which causes the script to exit immediately if any command fails
set -e

echo Downloading the Pile training set
poetry run ./scripts/sample_pile.py