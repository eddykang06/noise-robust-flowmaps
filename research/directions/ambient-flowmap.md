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

**First experiment.** Directly port ambient training to MeanFlow. 

We will use training data from the 2-armed spiral distribution. For the data with Gaussian corruption, 33% of the data will be clean. the remaining 66% will randomly sample a variance $\sigma^2 \in [0,10]$ and add it to the clean training point. 

For the non-Gaussian corruption, we will keep 33% clean data, and the remaining samples will be added Uniform noise dont

We can sweep the following directions as a first pass.
	1. iMF; standard iMF training (no noise adjustment); clean data 
	2. iMF; standard iMF training (no noise adjustment); Gaussian corruption
	3. iMF; ambient training; noisy data; Gaussian corruption
	4. iMF; ambient training: non-Gaussian corruption

