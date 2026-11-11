---
slug: ambient-flowmap
status: active
---

## ambient-flowmap

**Question.** Can we learn a flow map that learns from noisy data and learns to generate clean samples?

**Mechanism.** The instantiation of the flow map will be mean flow, and the noisy learning framework will be ambient diffusion (for now). These are the lowest-hanging fruit, but we hope to contribute more novel methods.

**Synthetic setup.** Our synthetic data setup consists of two target 2D distribution: 1) 2-armed spiral, 2) checkerboard. I want to setup a couple different corruption schemes for our training data. The  axes of variation are:
1. Type of corruption: Gaussian or non-Gaussian (e.g. masking)
2. Strength of corruption:

**First experiment.** Directly port ambient training to MeanFlow. We will use training data from the 2-armed spiral distribution. For the Gaussian noising case, to generate each datapoint, we will randomly sample a variance $\sigma^2 \in [0, 1]$ We can sweep the following directions as a first pass.
1. iMF; standard iMF training (no noise adjustment); Gaussian corruption
2. iMF: ambient training; noisy data; Gaussian corruption
3. iMF: ambient training: non-Gaussian corruption

