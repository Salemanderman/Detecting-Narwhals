Take raw .wav audiofiles and place in a directory (e.g. datadictionary) besides the other directories. 

open terminal and navigate to the Detecting-Narwhals directory.

setup a conda environment with the packages specified in the environment.yml file
This will create a conda environment called "narwhal_env":

conda env create -f environment.yml 

then activate the newly created environment:

conda activate narwhal_env

Then use preprocessing/run_extraction_noref.py with the specified --input-root --output-root 
and optional subset-len to take subset of audiofiles.
for example using the first 20 audiofiles only running on a macbook:

  Python preprocessing/run_extraction_noref.py \
      --input-root datadictionary \
      --output-root processedDataNPZFiles \
      --subset-len 20

This will produce npz files and metadata which can be used for different purposes.
  
