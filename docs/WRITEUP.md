# Write-up: what worked, what did not, and why

Ten weeks (July–September 2026), several hundred logged experiments, 47 scored submissions. This is the condensed story; every number below was measured
with the official metric (local runs on full ground truth, or the leaderboard).

## 1. First pipeline: transductive per crop (public 0.15 → 0.689)

The first question was what distinguishes an *annotated* cell. Inside a single video the signal is strong (classifier AUC
0.86–0.97), but across videos it collapses (leave-one-crop-out AUC ~0.55, sometimes inverted). A global "which cell is
annotated" detector is therefore structurally impossible, so the first pipeline worked per video: classical peak
detection, a detection budget calibrated to the node count the metric rewards, and a Kalman linker with Mahalanobis
gating and gap closing. The calibrated budget was the largest single jump of that phase (0.508 → 0.689).

## 2. Learned detectors and the organizer's recipe (0.843 → 0.878)

My own detectors (U-Net heatmaps, a Cellpose-style flow layer, learned selectors) plateaued at 0.843. The organizers'
reference model (temporal 3D U-Net + node transformer, 402 epochs) with an ILP linker and pruning of tracks shorter than
6 frames scored 0.9057 on my bench and 0.878 on the leaderboard: the recipe is the pipeline, and my selector + Kalman
stack was retired.

## 3. Fusing with the strongest public pipeline (0.947)

The best public notebooks add a second model seed, flip TTA, a DeepCenter rescue detector, bidirectional edge fusion,
~93 hand-tuned repair constants and a HOCT consensus veto. I reproduced it exactly (0.947) and set the goal of *fusing*:
keep what is proven, replace hand-tuned constants with learned components, measure piece by piece.

- **Divisions as a learned model.** A gradient-boosted classifier replacing the six "safe division" constants raised
  division Jaccard from 0 to 0.12 on my bench (+0.0065 total). On the leaderboard every change to the division stage lost
  (0.925–0.941): the deployed constants are gates tuned for precision, and the extra divisions were false positives there.
- **Dominance rule inside the ILP** (`edge_weight = 0.5 − p` with a wider candidate fan): +0.008 on the bench,
  0.936 on the leaderboard (0.909 private). The bench could not see the junk links the wider fan let in.

## 4. A model without hand-tuned constants

I retrained the organizer's architecture with a loss aligned with the metric (`cellmot/score_loss.py`):

1. the exact false-positive rule of the metric (an edge counts only if it touches an annotated cell with a successor or
   predecessor);
2. optimal one-to-one matching at 7 µm, as in the metric, instead of greedy matching;
3. the count factor without a ReLU (under-emitting is rewarded as much as over-emitting is punished);
4. what is trained is what is delivered: each cell chooses a parent or "no parent" and the soft Jaccard is the
   expectation of the delivered Jaccard;
5. the division term inside the loss.

Plus 8 physical input features per cell and a 5-frame temporal window. Alone, without any repair constants, it reached
0.875 public / 0.848 private. An apparent tie with the full pipeline on my bench turned out to be an aggregation
artifact (simple mean over crops instead of the metric's weighted aggregation); after fixing the aggregation the gap
was 0.025.

**Lesson:** aggregate the bench exactly like the metric does, and report median and per-crop wins, not only the mean.

## 5. Best of both: my edges inside the public pipeline (0.940 public / 0.915 private)

Keeping the public pipeline's detection and repair and swapping only its edge model for mine scored 0.940 public — below
0.947, so on the public board it looked like a loss. On the private split it was **0.915 vs 0.912**, and averaging both
edge models gave **0.916**, the best of all 47 submissions.

## 6. The faithful bench, and where offline evaluation stops transferring

To stop guessing, I made the public notebook run its own validator on my held-out training crops and dump its raw
post-ILP graphs; the notebook's repair can then be re-run locally with any constant changed, in seconds, without GPU.

| change | faithful bench | public LB | private LB |
|---|---|---|---|
| switch off motion relinking | +0.020 | −0.007 | −0.003 |
| my model as a third detection seed | +0.0076 more | −0.017 | −0.024 |
| ILP disappearance cost 1.5 / 2.5, detection threshold 0.98 | −0.004 / 0 / −0.001 | not sent | — |

Three findings:

- The bench **transfers for detection and divisions** but **not for edge choice or node removal**. The training
  annotation is sparse and contains identity jumps; the hidden annotation is denser. Measured directly: the same pipeline
  scores 0.8956 against training annotation of the public videos and 0.947 on the leaderboard. Removing nodes the bench
  considers noise removes cells the hidden annotation counts.
- The **public leaderboard (4 videos) cannot resolve ±0.005**, but large penalties there are real: the third seed lost
  0.024 on the private split too.
- **Final selection by offline evidence beat selection by public rank** (0.915 vs 0.912).

## 7. Practices that paid off

- The official metric runs locally and is unit-tested; nothing was judged on a proxy.
- Every submission carried a pre-registered expected score; misses were analysed, not explained away.
- Minimum effect size: two identical training runs differ by ±0.002 (non-deterministic `index_add_`), so effects
  below ±0.005 from a single run were not pursued.
- Destroy-the-piece test: before crediting a component, remove it and check the output changes. This caught an edge
  head that had been inert at inference for weeks.
- GPU budget discipline: 10 h/week cap, fp16 (2.35×), one process per T4, fail-fast checks at kernel start.
