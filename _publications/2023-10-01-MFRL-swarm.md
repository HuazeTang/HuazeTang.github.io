---
title: "Autonomous Swarm Robot Coordination via Mean-Field Control Embedding Multi-Agent Reinforcement Learning"
collection: publications
category: conferences
permalink: /publication/2023-10-01-MFRL-swarm
excerpt: ''
date: 2023-10-01
venue: '2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)'
paperurl: 'This paper is about the application of mean-field reinforcement learning in swarm robotics.'
citation: 'Tang, Huaze, et al. "Autonomous Swarm Robot Coordination via Mean-Field Control Embedding Multi-Agent Reinforcement Learning." 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023.'
---

Mean-field theory offers a promising solution to the scalability issues encountered in multi-agent reinforcement learning (MARL) within large-scale stochastic systems. However, most existing MARL algorithms based on mean-field theory are typically constrained to discrete mean-field spaces with one-hot encoding support. In continuous mean-field spaces, one-hot encoding presentation becomes unattainable due to the infinite granularity of mean-field values. In this paper, we propose Moment-Embedded Mean-Field Reinforcement Learning (ME-MFRL) for continuous mean-field space, which embeds empirical action distribution into space spanned by multi-order moments. We analyze the convergence of our proposed method to Nash equilibrium and develop Moment-Embedded Mean-Field Proximal Policy Optimization (ME-MFPPO) and Moment-Embedded Mean-Field Q-Learning (ME-MFQ) algorithms. Moreover, our proposed algorithms can easily adapted to discrete mean-field space by utilizing action embedding mapping the mean-field to a compact continuous domain. Finally, we validate the efficacy of proposed algorithms through experiments of mixed cooperative-competitive environments on both continuous and discrete action spaces.
