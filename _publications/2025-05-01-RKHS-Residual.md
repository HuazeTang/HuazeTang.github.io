---
title: "Residual kernel policy network: Enhancing stability and robustness in rkhs-based reinforcement learning"
collection: publications
category: conferences
permalink: /publication/2025-RKHS-Residual
excerpt: 'This paper is about a residual method with advantage functions to stabilize the RKHS RL methods.'
date: 2025-05-01
venue: 'The Thirteenth International Conference on Learning Representations (ICLR)'
slidesurl: 
paperurl: https://openreview.net/forum?id=2vgcDW2blS
citation: 'Zhang Y, Tang H, Lin H, et al. Residual kernel policy network: Enhancing stability and robustness in rkhs-based reinforcement learning[C]//The Thirteenth International Conference on Learning Representations. 2025.'
---

Achieving optimal performance in reinforcement learning requires robust policies supported by training processes that ensure both sample efficiency and stability. Modeling the policy in reproducing kernel Hilbert space (RKHS) enables efficient exploration of local optimal solutions. However, the stability of existing RKHS-based methods is hindered by significant variance in gradients, while the robustness of the learned policies is often compromised due to the sensitivity of hyperparameters. In this work, we conduct a comprehensive analysis of the significant instability in RKHS policies and reveal that the variance of the policy gradient increases substantially when a wide-bandwidth kernel is employed. To address these challenges, we propose a novel RKHS policy learning method integrated with representation learning to dynamically process observations in complex environments, enhancing the robustness of RKHS policies. Furthermore, inspired by the advantage functions, we introduce a residual layer that further stabilizes the training process by significantly reducing gradient variance in RKHS. Our novel algorithm, the Residual Kernel Policy Network (ResKPN), demonstrates state-of-the-art performance, achieving a 30% improvement in episodic rewards across complex environments.