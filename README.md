# Transmission-Line-Fault-Detection
Comparative analysis of eight ML and deep-learning models for three-phase transmission-line fault detection. Includes MS4, MS4N, Mamba-2, LSTM, LRCN, KAN, ANN, and PSO-ELM, with leakage-controlled evaluation, noise robustness testing, false-alarm analysis, and cross-configuration transfer.
# A Comparative Analysis of Modern ML Models for Power Line Fault Detection in Natural Disasters

Benchmark of **eight architectures across four model families** for binary fault detection on a
four-configuration three-phase transmission-line dataset, under a **single shared training protocol**
and a **leakage-controlled evaluation split**.

Structured state-space models (MS4 / MS4N), an input-dependent state-space model (Mamba-2) and a
Kolmogorov–Arnold Network are applied to transmission-line fault waveforms here for the first time,
to the authors' knowledge.


---

## Headline result

At 0 dB SNR every architecture exceeds 0.97 accuracy and six of eight fall within 0.005 of one
another, so **the aggregate ranking carries almost no information**. The models only separate under
stress, and when they do, the ordering is close to the reverse of what parameter count would predict:

| Criterion | Winner | Value |
|---|---|---|
| In-distribution accuracy / F1 | **KAN** | 0.9943 / 0.9969 |
| Fewest false alarms (specificity) | **ANN** (8,936 params) | FAR 0.0071 |
| Lowest missed-detection rate | **PSO-ELM** | MDR 0.0017 (but specificity 0.705) |
| Noise robustness at −10 dB | **KAN** | 0.9657 macro-F1 |
| Cross-configuration transfer | **LRCN** | leads 3 of 4 held-out cases |
| Strongest overall in-distribution | **Mamba-2** | 0.9959 F1 |

No single architecture dominates. Architecture selection on this task has to be argued from stressed
conditions, not from a leaderboard.

---

## Dataset and leakage control

Four configurations of a three-phase distributed transmission line, six channels each (three phase
voltages, three phase currents):

- **TL-1 / TL-2** — one generating unit, one RLC load, long and short line respectively
- **TL-3 / TL-4** — two generating units, three RLC loads, faults at the load end and at the source end

Sampling is ≈8.3 kHz for TL-1/TL-3/TL-4 and 1.5 kHz for TL-2. Eleven original labels (one healthy,
ten fault types) are mapped to a binary target.

https://doi.org/10.6084/m9.figshare.30615710

**Data-integrity screening happens before any modelling**, because three properties of these records
silently inflate reported accuracy:

- The TL-3 and TL-4 *classification* files are the same simulation stored twice — they differ by at
  most 0.185 across all values, roughly 7×10⁻⁶ relative on a ±27.7 kV signal. Only the genuine
  load-end data in the TL-3 *detection* file is used.
- TL-1, TL-3 and TL-4 are **tiled**: each nominal 2000-sample block is ≈500 unique samples repeated
  four times. TL-1 holds 5,510 distinct rows out of 21,999.
- TL-3 carries 21 isolated corrupted labels, which are removed.

**Preprocessing and splitting:**

- Segmentation follows the fault-type blocks; no window may span a join between the ten independent
  simulations concatenated in each record.
- Tiling period `p` detected as the smallest shift with `|s[p:] − s[:−p]|∞ < ε`; only the first `p`
  samples are kept.
- Windows of length `L = 32` (≈1/5 cycle at the higher sampling rate) at stride 2 → `X ∈ ℝ^(N×32×6)`.
- **Positional** 60/15/25 % train/val/test split with an `L`-sample guard band between partitions, so
  no test window shares a single sample with a training window.
- AWGN injected at a specified SNR; per-channel `σc = (Pc / 10^(SNR/10))^(1/2)`.
- Per-configuration z-scoring using that configuration's training statistics only — necessary because
  TL-2 magnitudes differ from the rest by roughly a factor of fifty.

Noise is **necessary, not incidental**: on the noiseless records every architecture exceeds 0.998
accuracy, at which point the comparison carries no information.

---

## Training protocol

Identical for all neural models, so differences reflect architecture rather than tuning:

- AdamW, lr 1e-3 with cosine decay, weight decay 1e-4, batch size 128, gradient clipping at unit norm
- Class-weighted loss `w_c = n / (2 n_c)` — the test partition is ≈92 % faulted
- Early stopping on validation macro-F1, patience 8, best checkpoint restored before testing
- PSO-ELM: 20 particles × 20 iterations, `w = 0.72`, `c1 = c2 = 1.49`, searching a binary descriptor
  mask jointly with ELM hyper-parameters; each fitness evaluation is one closed-form ridge solve

Evaluation is on a held-out partition of **5,103 windows, 91.7 % faulted** — a classifier that always
predicts "fault" already attains 0.917 accuracy. Because of this, the two operationally decisive
quantities are reported alongside accuracy:

```
MDR = FN / (FN + TP)      missed-detection rate
FAR = FP / (FP + TN)      false-alarm rate
```

---

## Results

### Detection performance at 0 dB

![Fault detection performance at 0 dB](outputs/fig1_metric_comparison.png)

*Note the suppressed vertical scale — the full plot range spans 0.11, and every model except PSO-ELM
lies within 0.006 of the others on accuracy.*

| Model | Acc. | Prec. | Rec. | F1 | Spec. | MCC | MDR | FAR | Params |
|---|---|---|---|---|---|---|---|---|---|
| MS4N (S4D) | .9902 | .9970 | .9923 | .9946 | .9670 | .9375 | .0077 | .0330 | 21,506 |
| MS4 (S4D) | .9888 | .9981 | .9897 | .9939 | .9788 | .9307 | .0103 | .0212 | 21,378 |
| Mamba-2 | .9926 | .9945 | .9974 | .9959 | .9387 | .9505 | .0026 | .0613 | 36,882 |
| **KAN** | **.9943** | .9989 | .9949 | **.9969** | .9882 | **.9637** | .0051 | .0118 | 31,286 |
| ANN | .9939 | **.9994** | .9940 | .9967 | **.9929** | .9616 | .0060 | **.0071** | **8,936** |
| PSO-ELM | .9739 | .9739 | **.9983** | .9860 | .7052 | .8166 | **.0017** | .2948 | 8,190 |
| LSTM | .9906 | .9961 | .9936 | .9949 | .9575 | .9392 | .0064 | .0425 | 51,970 |
| LRCN | .9922 | .9970 | .9944 | .9957 | .9670 | .9493 | .0056 | .0330 | 45,410 |

**The PSO-ELM trap.** Its recall (0.9983) and MDR (0.0017) are the best of any model, yet its
specificity is 0.7052 — it flags 29.5 % of healthy windows as faulted. Predicting the majority class
is rewarded by accuracy and recall alike; only specificity and MCC (0.8166 vs 0.9637 for KAN) expose
it. For a relay that trip rate is unusable.

### Confusion matrices

![Confusion matrices](outputs/fig2_confusion_matrices.png)

The informative cell is the upper right — false alarms on healthy windows. ANN raises 3, KAN 5, and
PSO-ELM 125 of 424.

### ROC curves

![ROC curves](outputs/fig3_roc_curves.png)

All models exceed 0.993 AUC; the metric saturates and does not discriminate.

### Convergence

![Training curves](outputs/fig4_training_curves.png)

KAN and ANN converge fastest and to the lowest validation loss. Mamba-2's validation loss diverges
after epoch ~7 while its macro-F1 plateaus — early stopping on macro-F1 restores the best checkpoint.

### Accuracy–complexity trade-off

![Accuracy vs complexity](outputs/fig5_accuracy_complexity.png)

Capacity does not buy accuracy here. ANN — the smallest neural model at 8,936 parameters and 9.1 s of
training — is second on F1 and beats the LSTM, which is six times larger and five times slower. MS4N
matches the LSTM at 41 % of its parameters, consistent with the efficiency claim for diagonal SSMs,
but so does the far simpler ANN. The standing of the descriptor-based models suggests the engineered
features already encode most of what separates the classes, leaving limited headroom for temporal
modelling.

### Per-configuration behaviour

![Per-configuration accuracy](outputs/fig6_per_configuration.png)

TL-2 is solved by every model (≥0.9996) and TL-1 nearly so. **All the variance lives in TL-4**, the
source-end fault case, where accuracy ranges from 0.9462 (MS4) to 0.9964 (ANN) — a spread twenty
times wider than in the aggregate. Source-end faults produce a weaker signature at the measurement
point, and the descriptor-based models handle this regime markedly better than the sequence models.

### Multi-criteria comparison

![Radar comparison](outputs/fig7_radar.png)

Seven models overlap almost exactly; PSO-ELM collapses on specificity and MCC.

### PSO feature selection

![PSO convergence](outputs/fig8_pso_convergence.png)

The swarm converges to **39 of 91 descriptors**, with global-best fitness improving in discrete steps
and flat after iteration 17.

### Noise robustness

![Noise robustness](outputs/fig9_snr_sweep.png)

On clean data six of seven models score exactly 1.000 macro-F1 — the noiseless benchmark is saturated
and cannot rank architectures. Separation emerges below 10 dB and widens monotonically.

| Model | ∞ (clean) | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB |
|---|---|---|---|---|---|---|---|
| MS4N (S4D) | 1.0000 | 0.9994 | 0.9949 | 0.9710 | 0.9529 | 0.9322 | 0.9194 |
| Mamba-2 | 1.0000 | 1.0000 | 0.9918 | 0.9929 | 0.9746 | 0.9431 | 0.9270 |
| **KAN** | 1.0000 | 0.9994 | 0.9974 | 0.9942 | **0.9810** | **0.9687** | **0.9657** |
| ANN | 1.0000 | 1.0000 | 0.9962 | 0.9818 | 0.9647 | 0.9497 | 0.9445 |
| PSO-ELM | 0.9968 | 0.9794 | 0.9500 | 0.9188 | 0.8826 | 0.9086 | 0.8829 |
| LSTM | 1.0000 | 0.9968 | 0.9949 | 0.9750 | 0.9399 | 0.9062 | 0.8864 |
| LRCN | 0.9911 | 1.0000 | 0.9962 | 0.9923 | 0.9708 | 0.9481 | 0.9201 |

KAN degrades most gracefully (0.9657 at −10 dB), then ANN (0.9445); the sequence models cluster
between 0.9194 and 0.9270 and LSTM falls to 0.8864. **This ordering is almost the reverse of what
architectural complexity would predict, and it is invisible above 10 dB.**

### Cross-configuration transfer

![Cross-configuration transfer](outputs/fig10_transfer.png)

Each model is trained on three configurations and evaluated on the fourth, held out entirely.

| Held out | MS4N | Mamba-2 | KAN | ANN | LSTM | LRCN |
|---|---|---|---|---|---|---|
| TL-1 | 0.9361 | 0.9667 | 0.9392 | 0.9093 | 0.9640 | **0.9870** |
| TL-2 | 0.7251 | 0.7721 | 0.7415 | 0.8273 | 0.7273 | **0.8533** |
| TL-3 | 0.9323 | **0.9527** | 0.9096 | 0.8776 | 0.9149 | 0.9129 |
| TL-4 | 0.5575 | 0.5231 | 0.4824 | 0.4456 | 0.5118 | **0.6253** |

Withholding TL-1 or TL-3 costs little; withholding TL-2 is harder; withholding TL-4 is severe, with
the best model (LRCN) reaching 0.6253 and ANN falling to 0.4456. **The descriptor-based models that
lead in distribution transfer worst** — the clearest evidence here that in-distribution accuracy does
not predict behaviour on an unseen topology.

