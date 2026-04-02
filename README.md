# Detecting Narwhals

Preprocess raw .wav audio into .npz feature files and metadata 

## Guide for preprocessing .wav files to .npz files
### 0) Navigate to your desired working directory and clone repository
```bash
$ git clone https://github.com/Salemanderman/Detecting-Narwhals.git 
```
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
For Windows:  
```bash
$ python analysis\pca_sliding_window.py --input-root processedDataNPZFiles --output-root analysis\pca_output --window-secs 5 --stride-secs 2.5 --mel-start 9 --mel-end 61 --n-components 50
```
If just a single file, add the --single-file and provide the filename

## Find outliers with the produced output from the PCA
With the pca part done we can now find the outliers, which is different from the noise.

Run the script analysis/finding_outliers.py with the necessary flags:

pca root is the relative path from the project root directory (Detecting-Narwhals) and to the file named 'pca_results.npz' produced by the previous PCA step.  
npz root is the path to the produced npz files from the extraction step. 

```bash
python analysis/finding_outliers.py \
        --pca-root subsetWithValidatedCalls/pca_output \
        --npz-root subsetWithValidatedCalls/npzFiles \
        --distance-metric mahalanobis \
        --threshold-std 3 \
        --plots-root analysis/outlier_plots \
        --save-csv \
        --mel-start 9 \
        --mel-end 61
```


## Running the full pipeline for outlier detection with pca with Standard Config

The `run_outlier_pipeline.py` file runs the complete outlier detection pipeline: 
  extraction from .wav to .npz files → PCA → find outliers 


### Setup

The default configuration is stored in `pipeline_config.py` at the project root.  
Customize the file `get_pipeline_config()` to adjust to your setup.  

Default configuration includes:
- **Input/output paths**: where to read audio files and save results
- **Window parameters**: window size, stride, mel bins
- **PCA settings**: n_components, feature extraction method
- **Outlier detection**: distance metric, threshold
- **Skip flags**: control which steps to run and which to skip

### Running with Default Config

When `pipeline_config.py` is set up with desired defaults, run by executing:

```bash
$ python run_outlier_pipeline.py
```

This will use all the default values from `pipeline_config.py`.

### Overriding Specific Parameters

You can override any default parameter using command-line flags:

```bash
$ python run_outlier_pipeline.py --threshold-std 4.0 --no-plot
```

### Skipping Pipeline Steps

If some of the steps is already executed, such as extracting from .wav to .npz, these can be skipped:

```bash
$ python run_outlier_pipeline.py --skip-extraction --skip-pca
```

### Example: Full Pipeline Run

```bash
# First run with custom parameters
$ python run_outlier_pipeline.py \
    --audio-root data/subsetWithValidatedCalls \
    --output-root output/pipeline_results \
    --window-secs 5 \
    --mel-start 9 --mel-end 61 \
    --n-components 20 \
    --threshold-std 3

# Re-run outlier detection with different threshold (skip extraction and PCA)
$ python run_outlier_pipeline.py --skip-extraction --skip-pca --threshold-std 4
```

### Output

The pipeline creates three subdirectories in the output root:
- `npz/`: Extracted mel spectrogram features
- `pca/`: PCA results and visualizations
- `outliers/`: Outlier detections, plots, audio clips, and CSV files


