# Detecting Narwhals

Preprocess raw .wav audio into .npz feature files and metadata 

## Guide for preprocessing .wav files to .npz files

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

```bash
$ cd /Users/johan/uni/bachelor/Detecting-Narwhals
```

### 3) Create and activate the Conda environment

This uses the packages in environment.yml and creates narwhal_env:

```bash
$ conda env create -f environment.yml  
$ conda activate narwhal_env
```

### 4) Run preprocessing / feature extraction

Use preprocessing/run_extraction_noref.py with required arguments:

- --input-root: folder with .wav files  
- --output-root: destination for generated .npz + metadata  
- --subset-len (optional): process only first N files

Example (first 20 files on a macbook):

```bash
$ python preprocessing/run_extraction_noref.py \
  --input-root datadictionary \
  --output-root processedDataNPZFiles \
  --subset-len 20
```

### Output

After running extraction, the output directory contains:

- .npz feature files
- metadata files for the .npz files called feature_index.csv  

feature_index.csv contains for each processed file the metadata:
 - source_path: path to original .wav audio file
 - feature_path: path to generated .npz files
 - sr: sample rate
 - shape: shape of 'feature' in the npz file

These outputs can be used later on


## produce pca with the npz files

with the npz files in processedDataNPZFiles we can do pca where each file is divided into individual 5-sec segments  

First go to the project directory in the teminal if not already there

```bash
$ cd PATH/TO/Detecting-Narwhals
```

Then run the pca:  
```bash
$ python analysis/pca_sliding_window.py \
        --input-root  processedDataNPZFiles \
        --output-root analysis/pca_output \
        --window-secs 5 \
        --stride-secs 2.5 \
        --mel-start 9 --mel-end 61 \
        --n-components 50
```
If just a single file, add the --single-file and provide the filename

Now it produces 

