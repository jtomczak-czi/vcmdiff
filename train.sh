#!/bin/bash
# Check if the correct number of arguments is provided
echo "Activating Python environment and navigating to $1" >$1/logs
# Redirect stdout and stderr to log files
cd /mnt/czi-sci-ai/intrinsic-variation-gene-ex/project_gene_regulation/scDiff
# Activate the Python environment
source /mnt/czi-sci-ai/intrinsic-variation-gene-ex/project_gene_regulation/.venv4/bin/activate
which python3 
python /mnt/czi-sci-ai/intrinsic-variation-gene-ex/project_gene_regulation/scDiff/train.py 
