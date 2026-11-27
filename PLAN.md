# AudioScope project plan

This document is the working plan for AudioScope. The pipeline, evaluation harness, and demo are implemented; [Milestones](#milestones) tracks which tasks are done. [README.md](README.md) is the public summary and will receive measured results only after the evaluation rules in this file are met.

## Goals

- Build a complete, readable path from a short audio clip to a class prediction: resample, spectrogram, CNN, top-k probabilities.
- Train and evaluate on ESC-50 using the dataset's official five folds. Report the mean and the standard deviation across folds.
- Start with a small CNN trained from scratch, then add augmentation, then compare transfer learning, so each stage has a measured baseline behind it.
- Ship a Streamlit demo that shows the waveform, the spectrogram, the top-k classes with confidence, and a Grad-CAM map on the spectrogram.
- Keep every reported number reproducible from a script in `eval/`, with the log, the metrics JSON, and the model and config versions saved beside it.
- Leave the code clear enough to serve as a portfolio project next to VisionClassify and VecScope.

## Non-goals

- A production audio-monitoring service, a real-time product, or a hosted deployment as part of the first completed version.
- Speech recognition, speaker identification, music tagging, or source separation.
- Multi-label tagging. ESC-50 clips are treated as single-label.
- Training an audio foundation model from unlabeled audio.
- A requirement to beat a published leaderboard. Published numbers are comparison points, cited from their papers.
- Redistributing ESC-50 or UrbanSound8K audio. Those files stay local and keep their own licenses.
- Tuning any hyperparameter on the fold that is being reported as the test fold.

## Background

These are the ideas the feature pipeline depends on. The implementation will follow the definitions here so training and the demo compute the same representation.

### Sampling rate

The sampling rate is the number of audio samples taken per second, in hertz. A signal sampled at 44.1 kHz, which is the native rate of ESC-50, can represent frequencies up to half that rate (the Nyquist frequency), so 22.05 kHz. Models usually resample every clip to one fixed rate before the spectrogram step. A lower rate, such as 16 kHz or 22.05 kHz, shortens the spectrogram and drops content above the new Nyquist limit. The chosen rate is a config value. It is stored with the run and applied to every fold the same way.

### Short-time Fourier transform (STFT)

A clip is a one-dimensional waveform. A CNN wants a local pattern on a grid. The STFT builds that grid: cut the waveform into overlapping frames, multiply each frame by a window (Hann, by default), and take the Fourier transform of the frame. The magnitude of those transforms is a spectrogram. One axis is time, advancing by the hop length. The other is frequency, in linear hertz bins. Frame length and hop length set the tradeoff between time detail and frequency detail, and both are config values.

### Mel scale

Human hearing resolves low frequencies more finely than high ones. A mel filterbank groups linear frequency bins into bands spaced so that equal steps along the mel scale are closer in hertz at the bottom of the spectrum and wider at the top. Summing energy inside each band produces a mel spectrogram with a chosen number of bands, often 64 or 128.

### Log-mel spectrogram

Sound energy covers several orders of magnitude. Taking the logarithm of the mel energies, with a small positive offset so silence stays finite, compresses that range. The log-mel spectrogram is the main input to the CNN: a single-channel image of shape `(n_mels, time)`. It keeps the time–frequency layout that convolution can use.

### MFCC

Mel-frequency cepstral coefficients are a further compression of the log-mel spectrogram. A discrete cosine transform decorrelates the mel bands, and the lower coefficients are kept. MFCCs are a compact summary and a useful comparison feature. They are optional in this project. They are a weaker primary input for a CNN, because the transform removes the local time–frequency structure the convolutions are meant to see. The ESC-50 paper's own classical baselines were built on MFCCs and zero-crossing rate, which is one reason to keep the option available.

## Dataset and splits

### Primary dataset: ESC-50

ESC-50 is the dataset the project stands on.

- 2,000 clips, 50 classes, 40 clips per class, so the classes are balanced.
- Each clip is 5 seconds, mono, 44.1 kHz.
- Five groups of ten classes: animals, natural soundscapes and water, human non-speech sounds, interior and domestic sounds, exterior and urban noises. The class list lives in the [official repository](https://github.com/karolpiczak/ESC-50).
- The release includes a metadata table (`esc50.csv`) with a `fold` column valued 1 through 5. Folds are the same size. Within a class, clips cut from the same source recording are placed in a single fold, which limits one form of leakage. Four source ids (131943, 134049, 209698, 234879) appear in two folds, each time under two different classes (clock_tick/clock_alarm, hen/rooster); `tests/test_folds.py` pins that fact.
- License: the dataset is distributed by its author under a Creative Commons non-commercial license. This repository will not commit the audio. Confirm the current terms on the dataset page before downloading.

Source: Karol J. Piczak, "ESC: Dataset for Environmental Sound Classification," *Proceedings of the 23rd ACM International Conference on Multimedia*, pp. 1015–1018, 2015. DOI: [10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390).

### Protocol

Use the official five folds. Do not reshuffle clips into new folds.

For fold *k* in 1..5:

- Test set: fold *k* only.
- Training set: the other four folds.
- Validation set, when early stopping or model selection needs one: a split taken from the training folds only.

The test fold is scored once, after training choices are fixed. Learning rate, epoch count, augmentation strength, and architecture choices are made on the training folds. Looking at the test fold and then changing the setup is a protocol violation, and that run is not a reportable result.

The headline numbers are the mean and the standard deviation of each metric across the five test folds. Per-fold scores are saved as well. A single fold is never presented as the result of the project.

### Published reference points

These figures are from Piczak (2015). They are not AudioScope results. They are the comparison points the evaluation write-up will cite.

The paper measured human accuracy with a crowdsourced listening task. Listeners chose one of the 50 class names. The reported average accuracy on ESC-50 is **81.3%**. The paper describes the figure as a rough estimate: on the order of two judgments per clip, with the usual limits of an untrained crowd asked to hold 50 categories in mind. It is the right human reference for this dataset, and it is not a controlled laboratory ceiling.

The same paper reports classical baselines on clip-level means and standard deviations of the first 12 MFCCs (dropping the 0th) plus zero-crossing rate, under 5-fold cross-validation:

| Classifier | ESC-50 accuracy reported by Piczak (2015) |
| --- | --- |
| Random forest | 44.3% |
| Linear SVM | 39.6% |
| k-nearest neighbors | 32.2% |

Later published systems, including convolutional models, PANNs, and the Audio Spectrogram Transformer, will be added to the comparison only by copying the number from the paper that reported it, with the citation beside it. A number remembered from a leaderboard, or read off someone else's README without the source, is not used.

### Optional dataset: UrbanSound8K

UrbanSound8K is an extension, taken up after the ESC-50 pipeline is reproducible. It has 10 urban-sound classes and 10 predefined folds (Salamon, Jacoby, and Bello, 2014). The same rules apply: official folds, no tuning on the test fold, mean and standard deviation across folds, and a separate metrics file. Results on UrbanSound8K are not substituted for ESC-50 results.

Dataset page: [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html).

## Model plan

Three stages, in order. A later stage starts only after the previous stage has a full 5-fold run saved under `eval/runs/`.

### Stage 1 — Baseline CNN

A small network trained from scratch on log-mel spectrograms.

- Input: one log-mel spectrogram per clip, normalized with statistics computed on the training folds of that run.
- Body: a few convolutional blocks (convolution, batch normalization, nonlinearity, pooling), then global pooling, dropout, and a linear layer with 50 outputs.
- Loss: cross-entropy. Optimizer: AdamW. Schedule and epoch budget live in `configs/baseline.yaml`.
- Purpose: a floor that later stages have to clear, and a model small enough to debug on a CPU.

The baseline is deliberately ordinary. Depth, width, and regularization get adjusted only through the training-fold validation split.

### Stage 2 — Augmentation

Add augmentation on top of the same baseline, one family at a time, so a change in the mean can be attributed.

- **SpecAugment.** Mask random time spans and frequency bands on the log-mel spectrogram during training (Park et al., 2019). Masks are not applied at validation or test time.
- **Mixup.** Mix pairs of training spectrograms and their one-hot labels with a coefficient drawn from a Beta distribution (Zhang et al., 2018).
- **Time shift.** Roll the waveform by a random offset before the spectrogram, training only.
- **Noise.** Add low-level noise to the training waveform.

Each family gets its own config under `configs/` and its own 5-fold run. A combined config comes after the separate runs, not before.

### Stage 3 — Transfer learning

Compare two uses of pretraining against the best model from stages 1 and 2. The fold protocol stays identical.

- **ImageNet ResNet.** Take a pretrained ResNet, repeat the log-mel spectrogram to three channels, resize it to the network's expected spatial size, and fine-tune. Earlier layers can stay frozen for the first epochs. This tests whether a vision backbone helps when the "image" is a spectrogram.
- **A pretrained audio model.** Compare one model that was trained on audio, such as PANNs (Kong et al., 2020) or AST (Gong, Chung, and Glass, 2021), if the weights and the license can be used and the input pipeline can be matched. The comparison is about representation and data scale, and the write-up will say which checkpoint was loaded.

Both runs record the source checkpoint name, the license note, and whether the backbone started frozen. Changing the sample rate or the mel settings away from what the pretrained model expects is a different experiment and gets its own config.

## Evaluation plan

### Metrics

On each test fold:

- **Accuracy.** Fraction of clips whose top class matches the label.
- **Macro F1.** Unweighted mean of per-class F1. With 50 balanced classes it should sit near accuracy, and a gap is a signal to inspect.
- **Confusion matrix.** Counts of true class against predicted class, saved as a table and as a figure.
- **Per-class accuracy.** One number per class per fold, so a weak class is visible even when the mean looks fine.

Across folds, report mean and standard deviation for accuracy and for macro F1. Average the confusion matrices for a summary figure, and keep the per-fold matrices in the run directory.

### What a reportable run contains

Every run that might be quoted lives in `eval/runs/<run-id>/`:

- `config.yaml` — the resolved config, including sample rate, STFT, mel settings, augmentation, and seed.
- `metrics.json` — per-fold accuracy and macro F1, per-class accuracy, the mean, and the standard deviation.
- `confusion_matrix.png` and the raw counts.
- `train.log` — the console log of the script that produced the JSON.
- `environment.txt` — Python, PyTorch, and torchaudio versions, plus the git revision.

`metrics.json` names the model, the config path, and the git revision, and sets `"reportable": true` only when every fold ran from a clean git tree. The script that writes it lives in `eval/` and calls library code under `src/audioscope/`. A hand-edited JSON file is not a result.

### Evidence rule

Every number in [README.md](README.md) comes from a script in `eval/` that was actually run, with the log and the metrics JSON saved, and with the model and config versions recorded in that JSON. A number that was not produced this way is not written into the README. That includes estimates, partial runs, validation-split scores presented as test scores, and figures copied from memory.

Published numbers from other papers may appear in this plan and, when a comparison is written, in the README next to their citation. They are labeled as external. They are never presented as an AudioScope measurement.

### Comparison write-up

When the first full baseline run exists, the README results section will state:

- the AudioScope mean ± standard deviation for accuracy and macro F1,
- the run id under `eval/runs/`,
- the cited human figure of 81.3% and the cited classical baselines from Piczak (2015),
- any later system whose published score was copied from its paper for that comparison.

The text will describe the gap in both directions without treating the human figure as a target the project is required to pass.

## Streamlit app

The demo is a single Streamlit app at `app/streamlit_app.py`. It calls the same feature code as training.

### Inputs

- Upload an audio file (WAV, FLAC, OGG, or MP3).
- Record from the microphone in the browser.
- Choose the checkpoint. The app reads the config saved with that checkpoint and uses it for resampling and the spectrogram, so the picture on screen is the picture the model scores.
- Choose *k* for the top-k list. Default is 5.

### What the page shows

- The waveform, with time in seconds.
- The log-mel spectrogram, with time in seconds and frequency in hertz (mel-band centers, not a linear axis pretending to be hertz).
- The top-k classes, each with its confidence, after a softmax over the 50 ESC-50 classes.
- Grad-CAM for the predicted class, drawn over the spectrogram, so a bright region means that time–frequency patch mattered for that class.
- The checkpoint name, the config name, and the dataset the model was trained on.

### Behavior before a model exists

The app still loads audio and draws the waveform and spectrogram when no checkpoint is available. It states that predictions and Grad-CAM need a trained checkpoint. Class names in the prediction list are the ESC-50 label set of the loaded model. A confidence is a score inside that set, not a claim that the clip is one of those sounds in an open-world sense.

### Grad-CAM

Grad-CAM (Selvaraju et al., 2017) is computed on a convolutional feature map of the selected class, then resized to the spectrogram grid and overlaid. The implementation will name which layer it reads. For a transfer model that reshapes the spectrogram to an ImageNet input size, the map is projected back onto the original time–frequency grid before display, so the overlay still reads as seconds and hertz.

## Repository structure

```text
AudioScope/
├── README.md                  # public summary; results only from eval runs
├── PLAN.md                    # this plan
├── docs/
│   ├── pipeline.svg           # static pipeline figure
│   └── pipeline.png
├── configs/                   # baseline.yaml, augment-*.yaml, transfer-*.yaml
├── data/                      # ESC-50 and optional UrbanSound8K (gitignored)
├── notebooks/                 # fold checks, spectrogram exploration
├── src/audioscope/
│   ├── data/                  # download, metadata, Dataset, official folds
│   ├── features/              # resample, STFT, log-mel, optional MFCC
│   ├── models/                # baseline CNN, ResNet wrapper, audio backbones
│   ├── train/                 # loop, schedules, augmentation
│   └── eval/                  # metric helpers shared with the scripts
├── eval/                      # entry points that write logs and metrics JSON
│   └── runs/                  # one directory per run (checkpoints gitignored)
├── app/
│   └── streamlit_app.py
└── tests/                     # folds, feature shapes, a tiny overfit check
```

`src/audioscope/eval/` holds functions. `eval/` at the repository root holds the scripts a person runs. The README may quote a number only when one of those scripts wrote it.

`data/` and checkpoints are in `.gitignore`. Run directories under `eval/runs/` are committed without their checkpoints, so each README number links to its metrics, log, and config. Checkpoints stay out of git unless a small demo checkpoint is deliberately added later, with its config and run id.

## Milestones

Checked tasks are done in the repository. A phase is finished when its tasks are done and, where a phase produces a measurement, the matching directory under `eval/runs/` exists.

### Phase 0 — Skeleton

- [x] Add a package layout matching the tree above, with empty modules and a `pyproject.toml`.
- [x] Add `.gitignore` for virtual environments, `data/`, checkpoints, and `eval/runs/`.
- [x] Add the MIT `LICENSE` file.
- [x] Pin the planned runtime in the project metadata: Python, PyTorch, torchaudio, librosa, matplotlib, Streamlit.

### Phase 1 — Data

- [x] Document and script the ESC-50 download into `data/esc50`.
- [x] Parse `esc50.csv` and build a dataset object keyed by the official fold id.
- [x] Test that the five folds are disjoint, cover every clip, and match the published fold column.
- [x] Notebook: one clip per class group, with waveform and log-mel side by side.

### Phase 2 — Features

- [x] Implement resample, STFT magnitude, mel filterbank, and log-mel in torchaudio.
- [x] Implement optional MFCC with the same front end.
- [x] Check a batch of clips against librosa within a stated tolerance, and save the comparison in the notebook.
- [x] Unit tests for output shape and for a fixed-seed spectrogram snapshot.

### Phase 3 — Baseline and the evaluation harness

- [x] Implement the small CNN and `configs/baseline.yaml`.
- [x] Implement the training loop with a training-fold validation split and a recorded seed.
- [x] Implement `eval/` cross-validation: five folds, no test-fold decisions, log plus `metrics.json`.
- [x] Run the baseline on all five folds and archive the run directory.
- [x] Confirm the README is still free of any score until that archive exists, then add the baseline from the JSON.

### Phase 4 — Augmentation

- [x] SpecAugment, training only, with its own config and 5-fold run.
- [x] Mixup, with its own config and 5-fold run.
- [x] Time shift and additive noise, each with its own config and 5-fold run.
- [x] One combined config, run only after the separate runs are archived.
- [x] Update the README from the new metrics files.

### Phase 5 — Transfer learning

- [x] ImageNet ResNet on three-channel log-mel inputs, with a recorded freeze schedule.
- [x] One pretrained audio model (PANNs or AST), input pipeline matched to that checkpoint. PANNs Cnn14 was run as a frozen feature extractor with a linear head (`configs/transfer-panns-cnn14-frozen.yaml`); full fine-tuning (`configs/transfer-panns-cnn14.yaml`) needs a GPU and was not run.
- [x] Five-fold runs for both, archived like the baseline.
- [x] Follow-up: the same `fc1` probe on two more PANNs checkpoints (Cnn14_16k, ResNet38) and a validation-weighted ensemble of the three (`eval/runs/ensemble-panns3-v1`, 94.8 ± 1.2%).
- [x] Write the comparison, citing external numbers from their papers. The comparison between AudioScope runs is in the README; published PANNs/AST ESC-50 numbers were not quoted because the papers could not be checked from the run environment.

### Phase 6 — Streamlit demo

- [x] Upload and microphone recording into the shared feature code.
- [x] Waveform and log-mel views with physical axis labels.
- [x] Top-k table bound to a selected checkpoint.
- [x] Grad-CAM overlay mapped back to the spectrogram axes.
- [x] Empty-checkpoint state that still renders the figures and explains why there is no prediction.

### Phase 7 — Optional UrbanSound8K

- [x] Loader for the official 10 folds.
- [ ] Repeat the best ESC-50 config under that protocol, as its own run.
- [ ] Report it separately from ESC-50.

Not run: the UrbanSound8K download (about 6 GB) and its 10-fold protocol do not fit the CPU-only budget this project was evaluated on. The loader (`load_urbansound8k`) and `configs/urbansound8k-baseline.yaml` are in place; no UrbanSound8K number is reported.

## Risks and mitigations

| Risk | What we will do |
| --- | --- |
| The test fold influences a design choice, and the score looks better than the protocol allows. | Validation comes only from the training folds. The test fold is scored by the `eval/` script after the config is committed. The run directory stores that config. |
| Official folds get replaced by a random split, and clips from one source recording land on both sides. | The dataset object reads the published `fold` column. A unit test checks disjointness and equality with that column. |
| Augmentation is left on at test time, or mixup is applied to the test batch. | Augmentation is inside the training loop, gated by the training flag. The feature tests cover the eval path with augmentation disabled. |
| Log-mel settings drift between training, evaluation, and the app, so the demo scores a different image than the model saw. | Sample rate, window, hop, mel bands, and normalization are fields of the checkpoint config. The app loads that config and does not keep a second copy of the constants. |
| torchaudio and librosa disagree, and a bug hides inside the difference. | Phase 2 records a tolerance check. Training uses torchaudio only. librosa is the reference and the drawing library. |
| A pretrained backbone expects a different rate, mel range, or normalization, and fine-tuning measures the mismatch. | The run record names the source checkpoint and the feature settings. A mismatched front end is a separate config, not a silent default. |
| Five folds of a larger model are slow on CPU and a partial run gets quoted. | A reportable result has all five folds in `metrics.json`. A partial run stays in its directory and out of the README. |
| Grad-CAM on a resized ImageNet input is drawn on the wrong axes. | The overlay is projected onto the original mel-by-time grid and labeled in seconds and hertz. The layer used for the map is fixed in config. |
| Similar mechanical classes dominate the errors and the mean hides it. | Per-class accuracy and the confusion matrix are required outputs, not optional plots. |
| Dataset audio or a checkpoint with a restricted license is committed under MIT. | `data/` and checkpoints are gitignored. The README states that dataset licenses stay with the datasets. A pretrained weight is documented before use. |
| A score is typed into the README from a notebook cell or from memory. | The evidence rule above is the gate. The results section names the run id that produced each AudioScope number. |

## References

- Karol J. Piczak. "ESC: Dataset for Environmental Sound Classification." *Proceedings of the 23rd ACM International Conference on Multimedia*, pp. 1015–1018, 2015. DOI: [10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390). Dataset: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50).
- Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. "A Dataset and Taxonomy for Urban Sound Research." *Proceedings of the 22nd ACM International Conference on Multimedia*, pp. 1041–1044, 2014. DOI: [10.1145/2647868.2655045](https://doi.org/10.1145/2647868.2655045). Dataset: [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html).
- Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech*, 2019.
- Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond Empirical Risk Minimization." *International Conference on Learning Representations*, 2018.
- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *IEEE International Conference on Computer Vision*, 2017.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 2016.
- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2020.
- Yuan Gong, Yu-An Chung, and James Glass. "AST: Audio Spectrogram Transformer." *Interspeech*, 2021.
