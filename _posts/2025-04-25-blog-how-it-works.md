---
title: "How it works: why use KL divergence as policy constrait? An information theory perspective."
date: 2025-04-25
permalink: /posts/2025/04/how_it_works/
tags:
  - Reinforcement Learning
  - Information Theory
  - KL Divergence
---

The Kullback-Leibler (KL) divergencehas been long used as a policy constrait in the field of reinforcement learning (RL). For example, in online RL, where agents interacts with the environment to update its policy, KL divergence is adopted to limit the search steps. Actually, KL divergence are so widely in the RL that it has become the golden standard. However, it sounds magical to me: why we adopt KL divergence as the constrait of policies?

KL Divergence and Related Meaning in Informative Perspective
======
KL divergence are long be viewed as a kind of *distance* for distribution. However, the KL divergence is not a true metric (it is asymmetric and does not satisfy the triangle inequality), while it serves as a principled measure of discrepancy between probability distributions. In reinforcement learning (RL), policies are essentially distributions over actions given states, and KL divergence provides an information-theoretic tool to quantify how "far" a new policy deviates from an old one during updates. Let’s dissect its role through two lenses: information theory and practical optimization.



Why KL divergence is useful in policy constrait?
======

Motivation of policy constrait
------

Futher thoughts on KL divergence
======