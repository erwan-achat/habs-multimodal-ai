# AGENTS.md

## 1. Development guidelines

- Never commit or push unless explicitly requested by the user.
- Avoid code comments, except when strictly necessary to explain non-obvious logic.
- Keep code minimal and compact. Prefer the simplest implementation that achieves the desired result.
- Code must remain readable and logically organized.
- Group process-related utilities (configuration, data loading, models, losses, etc.) in the root `disentanglement/` folder.
- Preserve configurability for hyperparameters in a `config.py` file.

## 2. Project objective

Implement a multimodal physiological signal disentanglement framework inspired by **MDNet (IJCAI 2024)** for emotion recognition.
Framework paper is available at `/docs/modality_and_subject_disentanglement.pdf`.

Our implementation will use four modalities:
- EEG: 62 electrodes.
- ECG: Electrocardiography.
- GSR: Galvanic Skin Response.
- Respiration.

The main objectives are:
- Learn modality-invariant and modality-specific representations.
- Separate subject-shared and subject-specific representations.
- Improve cross-subject emotion recognition.
- Preserve the possibility to add feature engineering, performing ablation studies, and explore temporal evolutions of emotions later on.

## 3. Dataset-specific rules

- Use `PREPROCESSED` signals only. The full dataset is not downloaded into the
  repository; local development uses `sample/`, while GPU jobs read `/data`.
- Load dataset metadata and labels before signal files, then build only a trial/window index.
- Implement lazy `Dataset` loading: open one participant recording, extract one
  10-second trial and its 3-second windows, then release or cache it according
  to memory limits. Never load the full 300+ GB dataset into memory.
- Exclude gaze data completely; it must not enter preprocessing, training or inference.
- Each 10-second video-viewing segment maps to one `final_emotion` target among 12 emotions, including neutral.
- Read trial annotations from `labels_train.csv`.
- Exclude recording trials flagged with `1` or the released `TESTSUBJECT` sentinel in `labels_train.csv`.
- Reconstruct segment boundaries from recording metadata and event/marker
  annotations; do not assume that trial order alone gives exact boundaries.
- Prefer exact BDF annotation boundaries: start marker `marker_id` and end marker
  `marker_id + 1`; use the following `*13` marker or metadata duration only as
  fallback. Never extend an interval past the physical BDF duration.

## 4. Temporal segmentation

Initial experimental configuration:
- Window duration: **3 seconds**.
- Overlap: **1 second**.
- Stride: **2 seconds**.

These parameters must remain configurable.

Each window constitutes an individual model input. Temporal aggregation across consecutive windows is outside the scope of the initial architecture but should remain possible in future extensions.

## 5. Modality encoders

Each modality must have its own feature extractor, adapted to its signal characteristics.

All encoders must output a feature vector of the same dimension:

`d = 128` (initial configuration).

### 5.1 EEG encoder

Use a 2D CNN to extract features from the EEG matrix:

- Input: `[62 electrodes, T time samples]`.
- Apply convolutions across both the electrode and temporal dimensions.
- Use temporal and spatial feature extraction.
- Apply pooling and a projection layer to obtain a 128-dimensional representation.

**Important:** Electrode ordering does not necessarily represent physical proximity. The initial CNN may exploit local channel-index relationships, but must not assume that these correspond to anatomical relationships. Explicit spatial geometry (10-10 electrode positions) is a potential future improvement.

### 5.2 ECG encoder

Use an independent 1D CNN.

- Input: ECG time series.
- Extract local temporal patterns.
- Apply pooling and projection to obtain a 128-dimensional representation.

### 5.3 GSR encoder

Use an independent 1D CNN.

- Input: GSR time series.
- Extract local temporal patterns.
- Apply pooling and projection to obtain a 128-dimensional representation.

### 5.4 Respiration encoder

Use an independent 1D CNN.

- Input: respiration time series.
- Extract local temporal patterns.
- Apply pooling and projection to obtain a 128-dimensional representation.

### 5.5 Encoder outputs

The four encoders produce:

- `u_eeg âˆˆ R^128`
- `u_ecg âˆˆ R^128`
- `u_gsr âˆˆ R^128`
- `u_resp âˆˆ R^128`

The encoders must remain independent. Their outputs are subsequently processed by the modality-level disentanglement module.

## 6. Modality-level disentanglement

Separate each modality representation into two complementary components:

- **Modality-invariant:** information shared across modalities.
- **Modality-specific:** information particular to each modality.

### 6.1 Encoders

Implement:

- One shared modality-invariant encoder, with shared parameters across all modalities.
- Four independent modality-specific encoders, one per modality.

For each modality `i`:

- `u_invariant_i = E_shared(u_i)`
- `u_specific_i = E_specific_i(u_i)`

Both representations must have dimension `d = 128`.

### 6.2 Losses

Implement the following constraints:

**Modality similarity loss (`L_m_sim`)**

Use Central Moment Discrepancy (CMD) to minimize distributional differences between modality-invariant representations.

**Modality difference loss (`L_m_diff`)**

Use orthogonality constraints to:
- Separate invariant and specific representations within each modality.
- Separate specific representations across different modalities.

**Modality reconstruction loss (`L_m_recon`)**

Use a decoder to reconstruct each original modality representation from the sum of its invariant and specific representations.

`u_reconstructed_i = D(u_invariant_i + u_specific_i)`

Minimize reconstruction error using MSE.

## 7. Multimodal fusion

Concatenate the invariant and specific representations of all modalities.

For four modalities, this produces eight feature tokens:

`[u_inv_eeg, u_spec_eeg, u_inv_ecg, u_spec_ecg, u_inv_gsr, u_spec_gsr, u_inv_resp, u_spec_resp]`

Apply Multi-Head Self-Attention to model interactions between these representations.

The resulting fused representation is the input to the subject-level disentanglement module.

## 8. Subject-level disentanglement

Separate the fused representation into:

- **Subject-shared:** information common across subjects.
- **Subject-private:** information specific to individual subjects.

### 8.1 Encoders

Implement:

- One subject-shared encoder with shared parameters across subjects.
- One independent subject-private encoder per training subject, with independent parameters.

Both branches must produce representations of the same dimension.

The subject-shared representation is used for emotion classification. Subject-private representations are used exclusively for disentanglement and reconstruction constraints.

The cross-subject inference process must support previously unseen subjects. Subject-private encoders must not be required for classification.

### 8.2 Losses

**Subject difference loss (`L_s_diff`)**

Use an orthogonality constraint to separate subject-shared and subject-private representations.

**Subject reconstruction loss (`L_s_recon`)**

Use a decoder to reconstruct the original fused representation from the sum of subject-shared and subject-private representations.

Minimize reconstruction error using MSE.

## 9. Classification and objective function

Use the subject-shared representation as input to the classification head.

The classification objective must support the emotion prediction targets defined by the experiment.

Use cross-entropy for classification.

The total training objective is:

`L = L_task + Î± L_m_sim + Î² L_m_diff + Î³ L_m_recon + Î´ L_s_diff + Îµ L_s_recon`

All loss weights must be configurable.

Initial reference weights from the MDNet paper:

- Î± = 0.002
- Î² = 0.1
- Î³ = 0.04
- Î´ = 0.5
- Îµ = 0.05

These values are just a starting point.

## 10. Evaluation and experimentation

Evaluate three distinct generalization capabilities using a subject-video split:

- **A â€” Subject generalization:** unseen subjects, known videos (`S_test Ã— V_train`).
- **B â€” Video generalization:** known subjects, unseen videos (`S_train Ã— V_test`).
- **C â€” Joint generalization:** unseen subjects, unseen videos (`S_test Ã— V_test`).

Training must exclusively use `S_train Ã— V_train`.

Requirements:
- As an initial configuration, split subjects and videos independently with an
  80/20 train/test proportion, and reserve a validation subset only from the
  80% training partition.
- Create splits from participant and video identifiers before indexing windows;
  do not use random window-level splits or shuffle windows across partitions.
- Treat each `video` identifier as an atomic unit; do not split by emotion
  category or by a normalized video name.
- Keep all windows and repeated recordings from a subject-video pair within the same partition.
- Reserve validation data exclusively from the training partitions for hyperparameter selection.
- Never use evaluation partitions A, B or C for model selection.
- Fit normalization statistics and learned preprocessing transformations exclusively on training data.
- Subject-private encoders must not be required for inference on unseen subjects.
- Report metrics separately for A, B and C.
- Support ablation studies to evaluate the contribution of each modality, disentanglement module and loss term.
- Keep experimental parameters configurable and results reproducible.

## 11. Signal sampling frequencies

Initial target sampling frequencies (`sfreq`):

| Modality | Target sfreq |
|---|---:|
| EEG | 256 Hz |
| ECG | 256 Hz |
| GSR | 32 Hz |
| Respiration | 32 Hz |

Requirements:
- Original sfreq in preprocessed data is 1kHz.
- Apply anti-aliasing filtering before downsampling.
- Preserve synchronization and temporal alignment across modalities.
- Keep target sampling frequencies configurable independently.

## 12. Signal normalization

Apply normalization independently to each modality.

Initial normalization strategy: channel-wise z-score standardization.

- **EEG:** Compute a separate mean and standard deviation for each electrode.
- **ECG:** Compute one mean and standard deviation.
- **GSR:** Compute one mean and standard deviation.
- **Respiration:** Compute one mean and standard deviation.

Requirements:
- Estimate normalization statistics exclusively from the training partition.
- Compute statistics from training recordings before windowing.
- Apply the same statistics to training, validation and test data.
- Never independently normalize validation or test data using their own statistics.
- Add a small numerical constant to standard deviations to prevent division by zero.
- Save normalization statistics with each experiment.
- Keep normalization configurable to support future comparisons and ablation studies.

Avoid per-window normalization in the initial implementation, as it may remove meaningful amplitude and baseline information.

## 13. Training and loss monitoring

### 12.1 Loss monitoring

Track and record the following losses at every epoch:

- Classification loss (`L_task`).
- Modality similarity loss (`L_m_sim`).
- Modality difference loss (`L_m_diff`).
- Modality reconstruction loss (`L_m_recon`).
- Subject difference loss (`L_s_diff`).
- Subject reconstruction loss (`L_s_recon`).
- Total loss.

Record training and validation classification losses separately. Disentanglement losses should also be monitored during training.

### 12.2 Validation

- Evaluate the validation partition after each epoch. It is drawn exclusively
  from `S_train Ã— V_train`.
- Do not build a nested validation A/B/C split; A/B/C are final evaluation
  partitions only and are evaluated once after training.
- Track at least accuracy, balanced accuracy, macro-precision, macro-recall and macro-F1.
- Use a configurable primary validation metric to select the best checkpoint.
- Initially, use validation macro-F1 as the primary selection criterion.
- Never use final test partitions for checkpoint selection or early stopping.

### 12.3 Early stopping and checkpoints

- Implement early stopping based on the primary validation metric.
- Make patience and minimum improvement (`min_delta`) configurable.
- Save the best model checkpoint whenever the primary validation metric improves.
- Restore the best checkpoint at the end of training.
- Save the final checkpoint as well, if useful for debugging and comparison.
- Support configurable maximum epochs and learning-rate scheduling.

### 12.4 Experiment outputs

Save all experiment artifacts inside a dedicated, configurable `out_dir`.

At minimum, save:
- Training and validation loss history.
- Individual loss-component histories.
- Validation metrics at each epoch, and final A/B/C metrics.
- Loss curves and validation metric curves.
- Validation subject/video pairs.
- Best model checkpoint.
- Experiment configuration and hyperparameters.
- Normalization statistics.
- Random seeds and subject/video split information.

Use a structured format such as CSV or JSON for numerical histories, alongside plots for visual inspection.

Each experiment must have its own output directory to avoid overwriting previous results. Experiments should be reproducible from their saved configuration and split information.

## 14. Running `process.py`

- Modify only [`process.py`](process.py) for the job entry point:
  `process(data_dir: Path, out_dir: Path) -> None`.
- Read the GPU-mounted, read-only dataset from `/data`; write every checkpoint,
  metric, plot and other artifact to the provided `out_dir`.
- Put extra dependencies in [`requirements.txt`](requirements.txt); CUDA and
  PyTorch are already available in the GPU image.