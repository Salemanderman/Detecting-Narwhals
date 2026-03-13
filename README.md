# Detecting Narwhals

Preprocess raw .wav audio into .npz feature files and metadata 

## Guide for using some of the implementations

### 1) Put your input audio files in a folder

Create a directory for raw .wav files (example: datadictionary) in the project folder:


Detecting-Narwhals/  
----datadictionary/  
--------audiofile1.wav  
--------audiofile2.wav  
--------...  
--------audiofileN.wav  
----preprocessing/  
----analysis/  
----utilities/  
----environment.yml  
----README.md  

### 2) Open a terminal in the project root


\$cd /Users/johan/uni/bachelor/Detecting-Narwhals


### 3) Create and activate the Conda environment

This uses the packages in environment.yml and creates narwhal_env:


\$conda env create -f environment.yml  
\$conda activate narwhal_env


### 4) Run preprocessing / feature extraction

Use preprocessing/run_extraction_noref.py with required arguments:

- --input-root: folder with .wav files  
- --output-root: destination for generated .npz + metadata  
- --subset-len (optional): process only first N files

Example (first 20 files on a macbook):


python preprocessing/run_extraction_noref.py \
  --input-root datadictionary \
  --output-root processedDataNPZFiles \
  --subset-len 20


## Output

After running extraction, the output directory contains:

- .npz feature files
- metadata files for the .npz files

These outputs can be used later on


