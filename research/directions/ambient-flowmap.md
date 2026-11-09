---
slug: ambient-flowmap
status: active
---

## ambient-flowmap

**Question.** Can we learn a flow map that learns from noisy data and learns to generate clean samples?

**Mechanism.** The instantiation of the flow map will be mean flow, and the noisy learningframework will be ambient diffusion

**First experiment.** Simply train meanflow on a partially noised synthetic dataset and assess the quality of generated samples. Then, traing meanflow using an ambient objective and compare.

