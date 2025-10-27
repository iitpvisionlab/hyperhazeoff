# HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing benchmark
Official Code for HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing Benchmark. [Preprints.org](https://www.preprints.org/manuscript/202510.1565)

## Code
The code is available in the [dev branch](https://github.com/iitpvisionlab/hyperhazeoff/tree/dev).

- [x] Benchmarking pipeline on RGB and HSI versions of RRealHyperPDID *24.10.2025*
- [ ] Visualization alghorithms, models weights trained on HyperDehazing, requirements.txt *27.10.2025*
- [ ] Field delineation quality assessment *29.10.2025*


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
--wl_src ./meta/wls/wavelengths_realhyper.npy \
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

## License
HyperHazeOff is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Citations


```
@article{202510.1565,
	doi = {10.20944/preprints202510.1565.v1},
	url = {https://doi.org/10.20944/preprints202510.1565.v1},
	year = 2025,
	month = {October},
	publisher = {Preprints},
	author = {Artem Nikonorov and Dmitry Sidorchuk and Nikita Odinets and Vladislav Volkov and Anastasia Sarycheva and Ekaterina Dudenko and Mikhail Zhidkov and Dmitry Nikolaev},
	title = {HyperHazeOff: Hyperspectral Remote Sensing Image Dehazing Benchmark},
	journal = {Preprints}
}
```
