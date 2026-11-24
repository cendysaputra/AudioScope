# AudioScope

Classify environmental sounds by turning audio into spectrograms and training a convolutional network on them.

## Overview

AudioScope is a learning and portfolio project. It takes short recordings of everyday sounds, converts each clip into a time–frequency image, and trains a convolutional neural network (CNN) to predict the sound class. The point of the project is to make that pipeline visible: from the waveform, through a spectrogram, to a prediction and a map of the time and frequency regions that drove it.

The work sits next to two other projects by the same author: [VisionClassify](https://github.com/cendysaputra/VisionClassify), a CNN image classifier trained on CIFAR-10, and [VecScope](https://github.com/cendysaputra/VecScope), an explorer for tokenization and embeddings. AudioScope applies the same kind of convolutional classifier to sound, using the spectrogram as the image.

The primary dataset is [ESC-50](https://github.com/karolpiczak/ESC-50). An optional second dataset is [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html).

**Status.** The pipeline, the evaluation harness, and the Streamlit demo are implemented. The 5-fold evaluation runs are in progress; [Results](#results) will list only numbers from run directories written by `eval/crossval.py`. Task status and the evaluation rules are in [PLAN.md](PLAN.md).

## How it works

Each clip follows one path from audio to a class prediction.

1. **Load.** Read the clip and its sample rate.
2. **Resample.** Convert the audio to the rate the model was trained on, so every clip yields a comparable spectrogram.
3. **Spectrogram.** Compute a short-time Fourier transform (STFT), pass the magnitudes through a mel filterbank, and take the logarithm. The result is a log-mel spectrogram: time on one axis, frequency on the other. Mel-frequency cepstral coefficients (MFCCs) are an optional extra representation for comparison.
4. **CNN.** A convolutional network reads the spectrogram as a single-channel image and outputs a score for each class.
5. **Prediction.** Softmax turns those scores into confidences. The demo shows the top-k classes. Grad-CAM highlights the time–frequency regions that contributed most to the chosen class.

![AudioScope pipeline](docs/pipeline.svg)

Training follows the same front end. A small CNN is the baseline. Later runs add spectrogram and waveform augmentation, then compare transfer from an ImageNet-pretrained network and from pretrained audio models. The full training plan is in [PLAN.md](PLAN.md).

## Features

- Build STFT and log-mel spectrograms with torchaudio, with MFCCs available as an optional feature.
- Train a small CNN, then the same model with augmentation, then a transfer-learning comparison.
- Evaluate with ESC-50's official 5-fold cross-validation and report mean and standard deviation across folds.
- Demo in Streamlit: upload a file or record from the microphone, view the waveform and spectrogram, and read the top-k classes with confidence.
- Overlay Grad-CAM on the spectrogram so the time and frequency evidence for a prediction is visible.

## Datasets

Audio files are downloaded locally and are not part of this repository. Each dataset keeps its own license. The MIT license of this code does not cover the recordings.

### ESC-50 (primary)

[ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015) contains 2,000 five-second clips, 50 classes, and 40 clips per class. Clips are mono and sampled at 44.1 kHz. The classes fall into five groups: animals, natural soundscapes and water, human non-speech sounds, interior and domestic sounds, and exterior and urban noises. The release defines five cross-validation folds. Clips that come from the same source recording stay in the same fold. The official citation and the dataset license are on the project page. The paper is listed in [PLAN.md](PLAN.md).

### UrbanSound8K (optional extension)

[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) (Salamon, Jacoby, and Bello, 2014) is a larger urban-sound set with 10 classes and 10 predefined folds. It is a later extension for checking whether the same pipeline transfers to a second label set. It is not required for the first milestone.

## Tech stack

| Piece | Role |
| --- | --- |
| Python | Language for training, evaluation, and the demo |
| PyTorch | Models, loss, and the training loop |
| torchaudio | Resampling, STFT, and mel spectrograms |
| soundfile | Reading WAV/FLAC/OGG/MP3 (librosa as fallback decoder) |
| librosa | Reference check against torchaudio, mel-band center frequencies |
| matplotlib | Saved figures for waveforms, spectrograms, and confusion matrices |
| Streamlit | Upload, record, visualize, and predict in the browser |

Accuracy, macro F1, per-class accuracy, and confusion matrices are computed in `src/audioscope/eval/metrics.py` (checked against scikit-learn in the tests). Each run saves its package versions in `environment.txt`.

## Project structure

```text
AudioScope/
├── README.md
├── PLAN.md
├── LICENSE
├── pyproject.toml
├── docs/                    # pipeline figure
├── configs/                 # one YAML file per experiment (baseline, augment-*, transfer-*)
├── data/                    # local datasets and pretrained weights (not committed)
├── notebooks/               # 01 folds + spectrograms, 02 torchaudio-vs-librosa check
├── src/audioscope/
│   ├── data/                # download, metadata, official folds, cached waveforms
│   ├── features/            # loading, resampling, log-mel, MFCC, normalization
│   ├── models/              # baseline CNN, ImageNet ResNet, PANNs Cnn14, Grad-CAM
│   ├── train/               # training loop and augmentation
│   ├── eval/                # metrics, reports, cross-validation driver
│   ├── inference.py         # load a checkpoint and score audio with its own front end
│   └── viz.py               # waveform / spectrogram figures in seconds and hertz
├── eval/                    # crossval.py, feature_check.py, results_table.py
│   └── runs/                # one directory per run (not committed)
├── app/streamlit_app.py     # Streamlit demo
└── tests/
```

`src/audioscope/` is the library. `eval/` holds the scripts that produce the numbers this README is allowed to quote. A score without a matching run in `eval/runs/` does not get written up.

## Getting started

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[app,dev]"
```

### 2. Download the data

```bash
python -m audioscope.data.download --dataset esc50 --dest data/esc50
python -m audioscope.data.download --weights resnet18 cnn14   # only for the transfer-learning configs
```

The ESC-50 clips can also be taken from the [official repository](https://github.com/karolpiczak/ESC-50). Keep them under `data/` and out of git.

### 3. Evaluate with 5-fold cross-validation

```bash
python eval/crossval.py --config configs/baseline.yaml --out eval/runs/baseline-v1
```

For each official fold the script trains on the other four (with a validation split taken from them), selects the checkpoint on that validation split, and scores the held-out fold once. It writes `config.yaml`, `train.log`, `environment.txt`, `metrics.json` (per-fold and mean ± std accuracy and macro F1, per-class accuracy, git revision), confusion matrices as CSV and PNG, `predictions.csv`, and one checkpoint per fold. A run is marked `"reportable": true` only if it covers every fold from a clean git tree.

```bash
python eval/results_table.py          # Markdown table of reportable runs
python eval/feature_check.py --n 64   # torchaudio vs librosa tolerance check
pytest                                # folds, features, augmentation, models, a tiny overfit
```

To train a single checkpoint (for the demo) without scoring its held-out fold:

```bash
python -m audioscope.train --config configs/baseline.yaml --holdout-fold 1 --out checkpoints/baseline.pt
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app lists checkpoints under `eval/runs/*/fold*/model.pt` and `checkpoints/*.pt` and rebuilds the front end from the config saved in the selected checkpoint. It accepts an upload (WAV, FLAC, OGG, MP3) or a microphone recording and shows the waveform, the log-mel spectrogram with mel-band center frequencies on the y axis, the top-k classes, and a Grad-CAM overlay. Without a checkpoint it still draws the waveform and spectrogram and says that predictions need a trained model. Checkpoints are not committed; create one with step 3.

## Results

No model has been trained, and no evaluation has been run. This section will report accuracy and macro F1 as the mean and standard deviation across the five official ESC-50 folds once those figures exist.

A figure belongs here only when a script in `eval/` produced it and saved both a log and a metrics JSON file. That JSON records the model, the config, and the code version of the run. Scores that were not produced this way will not be added to this file.

## Roadmap

The work is sequenced so that a reproducible baseline exists before augmentation, transfer learning, or the demo claim a gain. Detail, task lists, and the evaluation rules are in [PLAN.md](PLAN.md).

1. Repository skeleton, data download, and official fold splits.
2. Log-mel features, checked against a reference implementation.
3. Baseline CNN and the `eval/` harness on 5-fold cross-validation.
4. Augmentation: SpecAugment, mixup, time shift, and noise.
5. Transfer learning: an ImageNet-pretrained ResNet on spectrograms, and a pretrained audio model (PANNs or AST) as a comparison.
6. Streamlit demo with waveform, spectrogram, top-k predictions, and Grad-CAM.
7. Write the measured results back into this README from the saved metrics files.

## Related projects

- [VisionClassify](https://github.com/cendysaputra/VisionClassify) — CNN image classification on CIFAR-10.
- [VecScope](https://github.com/cendysaputra/VecScope) — exploration of tokenization and embeddings.

## License

The code is released under the [MIT License](LICENSE). Dataset recordings stay under the licenses of their publishers (ESC-50: CC BY-NC) and are not redistributed here. Pretrained weights keep their own licenses: torchvision ResNet-18 (BSD-3-Clause) and PANNs Cnn14 (CC BY 4.0, Zenodo record 3987831).
