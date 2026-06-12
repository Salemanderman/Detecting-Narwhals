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
----clustering/  
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

- --audio-root: folder with .wav files  
- --output-root: destination for generated .npz + metadata  
- --subset-len (optional): process only first N files

Example (first 20 files on a macbook):

```bash
$ python preprocessing/run_extraction_noref.py \
  --audio-root datadictionary \
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
        --npz-root  processedDataNPZFiles \
        --output-root analysis/pca_output \
        --window-secs 5 \
        --stride-secs 5.0 \
        --mel-start 11 --mel-end 128 \
        --n-components 20
```
For Windows:  
```bash
$ python analysis\pca_sliding_window.py --npz-root processedDataNPZFiles --output-root analysis\pca_output --window-secs 5 --stride-secs 5.0 --mel-start 11 --mel-end 128 --n-components 20
```
If just a single file, add the --single-file and provide the filename

## Find outliers with the produced output from the PCA
With the pca part done we can now find the outliers, which is different from the noise.

Run the script analysis/finding_outliers.py with the necessary flags:

pca root is the relative path from the project root directory (Detecting-Narwhals) and to the file named 'pca_results.npz' produced by the previous PCA step.  
npz root is the path to the produced npz files from the extraction step. 

```bash
python analysis/finding_outliers.py \
        --pca-root analysis/pca_output \
        --npz-root processedDataNPZFiles \
        --distance-metric mahalanobis \
        --threshold-percentile 95 \
        --output-root analysis/outlier_plots \
        --mel-start 11 \
        --mel-end 128
```

The outliers.csv and a report are saved by default. Add `--plot` to also save spectrogram plots, and `--audio-root <path-to-wav-files>` to save audio clips of each outlier.


## Cluster the outliers

Once you have detected outliers, you can cluster them to group similar types together:

```bash
python clustering/cluster.py \
        --pca-root analysis/pca_output \
        --outliers-csv analysis/outlier_plots/outliers.csv \
        --output-root analysis/clusters \
        --npz-root processedDataNPZFiles \
        --algorithm kmeans --n-clusters 5
```

This writes `clusters.csv` (each outlier with its cluster), a scatter plot, and a spectrogram grid per cluster.

You can adjust:
- `--algorithm`: `kmeans`, `hdbscan`, or `dpmm`
- `--n-clusters`: number of clusters for k-means (default: 7)


## Review clusters and label them

To step through each cluster, look at its spectrogram grid, and label it keep or remove (and optionally a call type such as clicks or tonal):

```bash
python clustering/interactive_cluster_review.py \
        --pca-root analysis/pca_output \
        --npz-root processedDataNPZFiles \
        --output-root analysis/review \
        --strategy d --mel-start 11 --mel-end 128
```

Labels are saved as you go and can be reused to train a classifier.


## Train a type classifier and classify windows

Train a random forest on the labelled clusters:

```bash
python clustering/train_type_classifier.py \
        --annotations-csv analysis/review/type_annotations.csv \
        --npz-root processedDataNPZFiles \
        --output-root evaluation
```

Then predict the type of any window list containing `File` and `Start Time (s)` columns:

```bash
python clustering/classify_windows.py \
        --windows-csv analysis/outlier_plots/outliers.csv \
        --npz-root processedDataNPZFiles \
        --output-root analysis/classified
```

And save spectrogram grids grouped by predicted type to check them:

```bash
python clustering/plot_classified_grids.py \
        --classified-csv analysis/classified/clusters_classified.csv \
        --npz-root processedDataNPZFiles \
        --output-root analysis/classified_grids \
        --types clicks tonal
```


## Running the full pipeline for outlier detection with pca with Standard Config

The `run_outlier_pipeline.py` file runs the complete outlier detection pipeline:   
1. extraction from .wav to .npz files → 
2. PCA → 
3. find outliers in pca


### Setup

The default configuration is stored in `utilities/configs.py`.  
Customize `get_pipeline_config()` to adjust to your setup.  

Default configuration includes:
- **Input/output paths**: where to read audio files and save results
- **Window parameters**: window size, stride, mel bins
- **PCA settings**: n_components, feature extraction method
- **Outlier detection**: distance metric, threshold
- **Skip flags**: control which steps to run and which to skip

### Running with Default Config

When `utilities/configs.py` is set up with desired defaults, run by executing:

```bash
$ python run_outlier_pipeline.py
```

This will use all the default values from `utilities/configs.py`.

### Overriding Specific Parameters

You can override any default parameter using command-line flags:

```bash
$ python run_outlier_pipeline.py --threshold-percentile 95 --no-plot
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
    --mel-start 11 --mel-end 128 \
    --n-components 20 \
    --threshold-percentile 95

# Re-run outlier detection with different threshold (skip extraction and PCA)
$ python run_outlier_pipeline.py --skip-extraction --skip-pca --threshold-percentile 97
```

### Output

The pipeline creates three subdirectories in the output root:
- `npz/`: Extracted mel spectrogram features
- `pca/`: PCA results and visualizations
- `outliers/`: Outlier detections, plots, audio clips, and CSV files


## Evaluate against the validated calls

Compare detected outliers against the hand-labelled calls in `evaluation/validatedChristerCalls.csv`:

```bash
python evaluation/compareChristerCalls.py \
        --outliers-csv analysis/outlier_plots/outliers.csv \
        --validation-csv evaluation/validatedChristerCalls.csv
```

This prints recall, precision, and F1, plus which calls were matched and missed.


## Autoencoder anomaly detection

An alternative detector based on a convolutional autoencoder lives in the notebook `analysis/autoencoder_anomaly_detection.ipynb`. It trains on the spectrogram windows and flags the ones with high reconstruction error.

