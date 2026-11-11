#!/usr/bin/env bash
# Run every notebook in order. Stops at the first error.
set -e
cd "$(dirname "$0")/notebooks"
for nb in 01_*.ipynb 02_*.ipynb 03a_*.ipynb 03b_*.ipynb 03c_*.ipynb 04_*.ipynb 05_*.ipynb; do
  echo "Running $nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
echo "Done."
