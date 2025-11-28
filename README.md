# HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing benchmark
Official Code for HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing Benchmark. [MDPI J. Imaging](https://www.mdpi.com/2313-433X/11/12/422), [Preprints.org](https://www.preprints.org/manuscript/202510.1565).

## Code
The code is available in the [main branch](https://github.com/iitpvisionlab/hyperhazeoff).

- [x] Benchmarking pipeline on RGB and HSI versions of RRealHyperPDID *24.10.2025*
- [x] Visualization alghorithms, models weights trained on HyperDehazing, requirements.txt *27.10.2025*
- [x] Field delineation quality assessment *29.10.2025*


## Datasets 

**Remote sensing Real-world Hyperspectral Paired Dehazing Image Dataset (RealHyperPDID)**: [HuggingFace](https://huggingface.co/datasets/nikos74/RRealHyperPDID).

**Remote sensing Synthetic Hyperspectral Paired Dehazing Image Dataset (RSyntHyperPDID)**: [HuggingFace](https://huggingface.co/datasets/nikos74/RSyntHyperPDID).

## Getting started

### Installation 

1. Clone the repository:
```
git clone https://github.com/iitpvisionlab/hyperhazeoff.git
```
2. Change to the project directory:
```
cd hyperhazeoff 
```
3. Create and activate a Python virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate 
```
4. Install the required Python dependencies:
```
pip install -r requirements.txt
```

### Data

1. Create the data directory (if it does not exist):
```
mkdir -p data/source
```
2. Install the Hugging Face Hub Python client:
```
pip install huggingface_hub
``` 
3. Download the RRealHyperPDID dataset from Hugging Face directly into data/source:
```
huggingface-cli download nikos74/RRealHyperPDID --repo-type dataset --local-dir data/source
```
4. Unzip all ZIP archives in the data/source directory:
```
cd data/source
unzip '*.zip'
```

### Inference 

#### A. Inference for dehazing methods trained by RSyntHyperPDID on RRealHyperPDID

From the root directory *hyperhazeoff*, run:

```
python3 inference.py --config configs/hsi/rrhpdid_aacnet.yaml	# Alternatives: rrhpdid_aid.yaml | rrhpdid_hdmba.yaml
```

#### B. Inference for dehazing methods on the RGB part of RRealHyperPDID
```
python3 inference.py --config configs/rgb/rrhpdid_dcp.yaml	# Alternatives: rrhpdid_cadcp.yaml | rrhpdid_dehazeformer.yaml
```

#### C. Inference for dehazing methods trained by HyperDehazing on RRealHyperPDID

1. Perform spectral harmonization first:
```
python3 utils/data/harmonization.py --input-dir ./data/source/RRealHyperPDID/HSI \
--wl-src ./meta/wls/wavelengths_realhyper.npy \
--wl-tgt ./meta/wls/wavelengths_hyperdehazing.npy \
--output-dir ./data/source/RRealHyperPDID/interHSI
```
2. Then run inference:

```
python3 inference.py --config configs/hsi/hd_aacnet.yaml	# Alternatives: hd_aid.yaml | hd_hdmba.yaml
```

### Benchmarking

#### A. For HyperSpectral Images:
Run the benchmarking script for hyperspectral image results:
```
python3 benchmarking.py --benchmark-dir ./data/source/RRealHyperPDID/HSI \
--dehazed-dir ./data/output/... \
--model-name ... \
--out-dir ... \
--device cuda \
--format-data npy
```
Replace the `...` fields with your dehazed output path, model name identifier, and desired output path as needed.

#### B. For RGB Images:
Run the benchmarking script for RGB image results:
```
python3 benchmarking.py --benchmark-dir ./data/source/RRealHyperPDID/CSNC \
--dehazed-dir ./data/output/... \
--model-name ... \
--out-dir ... \
--device cpu \
--format-data png
```
Again, fill in the placeholder values for your specific results.


### Calculate Metrics Value for Subset of Closest Pairs
To compute evaluation metrics for a predefined subset of image pairs:
```
python3 calculate_subset_metrics.py --metrics-csv <PATH_TO_METRICS_CSV> \
--subset-csv <PATH_TO_SUBSET_CSV> \
--output-json <PATH_TO_OUTPUT_JSON>
# Replace angle brackets with your file locations

```

### Hyperspectral Image Visualization 

This script generates RGB visualizations from hyperspectral data using different synthesis methods.
```
python3 utils/data/visualization.py --input-dir <PATH_TO_HSI_DATA> \
--wl <PATH_TO_WAVELENGTHS_FILE> \
--output-dir <PATH_TO_OUTPUT_DIR> \
--method [csnc|csso|both]
```
Replace <PATH_TO_HSI_DATA> with the input hyperspectral images directory, <PATH_TO_WAVELENGTHS_FILE> with your wavelength file (`hyperhazeoff/meta/wls/wavelengths_realhyper.npy`), and <PATH_TO_OUTPUT_DIR> with the folder for saving visualizations. Set --method to csnc, csso, or both depending on the visualization method you want.

## License
HyperHazeOff is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Citations


```
@Article{jimaging11120422,
AUTHOR = {Nikonorov, Artem and Sidorchuk, Dmitry and Odinets, Nikita and Volkov, Vladislav and Sarycheva, Anastasia and Dudenko, Ekaterina and Zhidkov, Mikhail and Nikolaev, Dmitry},
TITLE = {HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing Benchmark},
JOURNAL = {Journal of Imaging},
VOLUME = {11},
YEAR = {2025},
NUMBER = {12},
ARTICLE-NUMBER = {422},
URL = {https://www.mdpi.com/2313-433X/11/12/422},
ISSN = {2313-433X},
ABSTRACT = {Hyperspectral remote sensing images (HSIs) provide invaluable information for environmental and agricultural monitoring, yet they are often degraded by atmospheric haze, which distorts spatial and spectral content and hinders downstream analysis. Progress in hyperspectral dehazing has been limited by the absence of paired real-haze benchmarks; most prior studies rely on synthetic haze or unpaired data, restricting fair evaluation and generalization. We present HyperHazeOff, the first comprehensive benchmark for hyperspectral dehazing that unifies data, tasks, and evaluation protocols. It comprises (i) RRealHyperPDID, 110 scenes with paired real-haze and haze-free HSIs (plus RGB images), and (ii) RSyntHyperPDID, 2616 paired samples generated using a physically grounded haze formation model. The benchmark also provides agricultural field delineation and land classification annotations for downstream task quality assessment, standardized train/validation/test splits, preprocessing pipelines, baseline implementations, pretrained weights, and evaluation tools. Across six state-of-the-art methods (three RGB-based and three HSI-specific), we find that hyperspectral models trained on the widely used HyperDehazing dataset fail to generalize to real haze, while training on RSyntHyperPDID enables significant real-haze restoration by AACNet. HyperHazeOff establishes reproducible baselines and is openly available to advance research in hyperspectral dehazing.},
DOI = {10.3390/jimaging11120422}
}
```
