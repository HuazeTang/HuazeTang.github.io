---
title: "How it works: why use KL divergence as policy constraint? An information theory perspective."
collection: posts
date: 2025-04-25
permalink: /posts/2025/04/how_it_works/
tags:
  - Reinforcement Learning
  - Information Theory
  - KL Divergence
---

The Kullback-Leibler (KL) divergencehas been long used as a policy constraint in the field of reinforcement learning (RL). For example, in online RL, where agents interacts with the environment to update its policy, KL divergence is adopted to limit the search steps. Actually, KL divergence are so widely in the RL that it has become the golden standard. However, it sounds magical to me: why we adopt KL divergence as the constraint of policies?

KL Divergence in Informative Perspective
======

Definition of KL divergence
------
The Kullback-Leibler (KL) divergence, denoted \\(D_{\text{KL}}(\mu\Vert \nu)\\), is a measure of how one probability distribution \\(\mu\\) diverges from a second probability distribution \\(\nu\\) over the same sample space. It is defined under the condition that \\(\mu\\) is absolutely continuous with respect to \\(\nu\\) ((denoted as \\(\mu\ll\nu\\)). meaning that if \\(\nu\\) assigns zero probability to an event, \\(\mu\\) must also assign zero probability to that event.
Formally, let \\(\mu\\) and \\(\nu\\) be probability measures on a measurable space \\((\Omega, \mathcal{F})\\). The KL divergence is defined as:

$$
D_{\text{KL}}(\mu\Vert \nu) = \int_\Omega \log\left(\frac{\mathrm{d}\mu}{\mathrm{d}\nu}\right)\mathrm{d}\mu,
$$

where \\(\mathrm{d}\mu/\mathrm{d}\nu\\) is Radon-Nikodym derivative of \\(\mu\\) with respect to \\(\nu\\). The integral is taken over the support of \\(\mu\\), and the logarithm is typically the natural logarithm (base \\(e\\)).

For two special case, we can see that for discrete space and continuous space:

* Discrete Distributions: if \\(\mu\\) and \\(\nu\\) are discrete distributions with probability mass functions \\(p(x)\\) and \\(q(x)\\) defined on a countable set \\(\mathcal{X}\\), then:

$$
D_{\text{KL}}(\mu\Vert \nu) = \sum_{x\in\mathcal{X}} p(x)\log\frac{p(x)}{q(x)}.
$$

* Continuous Distributions: if \\(\mu\\) and \\(\nu\\) are continuous distributions with probability density functions $p(x)$ and $q(x)$ with respect to a common measure (e.g., Lebesgue measure), then:

$$
D_{\text{KL}}(\mu\Vert \nu) = \int_{\mathcal{X}} p(x)\log\frac{p(x)}{q(x)}\mathrm{d}x.
$$

KL divergence are long be viewed as a kind of *distance* for distribution. However, the KL divergence is not a true metric (it is asymmetric and does not satisfy the triangle inequality), while it serves as a principled measure of discrepancy between probability distributions. In reinforcement learning (RL), policies are essentially distributions over actions given states, and KL divergence provides an information-theoretic tool to quantify how "far" a new policy deviates from an old one during updates. Let’s dissect its role from the prespective of information theory.

KL divergence in error probobility
------
The KL divergence quantifies the asymptotic decay rate of error probabilities in statistical hypothesis testing, particularly in distinguishing between two distributions \\(\mu\\) (null hypothesis \\(H_0\\)) and \\(\nu\\) (alternative hypothesis \\(H_1\\)) as the sample size \\(n\to\infty\\). The core of this is the Stein's Lemma:

In Neyman-Pearson hypothesis testing with fixed Type I error probability (\\(\alpha\\), False rejection of \\(H_0\\)), he Type II error probability ($\beta_n$, False acceptance of \\(H_0\\)) decays exponentially with $n$. The KL divergence governs this decay rate:

$$
\lim_{n\to\infty} \frac{1}{n}\log \beta_n = - D_{\text{KL}}(\mu\Vert\nu).
$$

Equivalently, $\beta_n\approx \exp(-n\cdot D_{\text{KL}}(\mu\Vert\nu))$ for large $n$. This is because the optimal test uses the log-likelihood ratio is adopted to decide whether to accept the hypothesis \\(H_0\\) or not: denote

$$
\text{LLR} = \frac{1}{n}\sum_{i=1}^n \log\frac{\mu(x_i)}{\nu(x_i)},
$$

and if $\text{LLR}>0$ then accept \\(H_0\\) and accept \\(H_1\\) otherwise. Hence, by the law of large numbers:

$$
\lim_{n\to\infty} \text{LLR} = \mathbb{E}_{\mu} \log\frac{\mu(x)}{\nu(x)} = D_{\text{KL}}(\mu\Vert\nu).
$$ 

That is where the KL divergence comes in the error probability.

Why KL divergence is useful in policy constraint?
======

Motivation of policy constraint
------
When collecting data with new policy \\(\pi\\) yet applying value function of old policy \\(\pi_{\text{old}}\\), there are two risks:

* Overestimation error: the new policy \\(\pi\\) tends to out-of-distribution action of \\(\pi_{\text{old}}\\), which maybe a fault action. This case is common in offline RL tasks.

* Distribution shift: the state grenerated from new policy \\(\pi\\) shifts from that of \\(\pi_{\text{old}}\\), disabling the value function to judge the current state. This case is common in off-policy RL tasks.

The avoid or relief these risks, methods hope to constrain the new policy from too far away from old policy \\(\pi_{\text{old}}\\). This gives birth to the trust region concept.

KL divergence as error constraint
------
In one another way, we can view the constraint as a hyponthesis testing problem. Let 

* \\(H_0\\) be "the policy for collecting data is old policy \\(\pi_{\text{old}}\\)";

* \\(H_1\\) be "the policy for collecting data is new policy \\(\pi\\)".

Our goal is that under trajecoty data \\(\{\tau_i\}_{i=1}^n\\) collected by new policy, with each trajectory \\(\tau_i=(s_i^1,a_i^1,\dots, s_i^{T_i}, a_i^{T_i})\\), find the risk of accepting \\(H_0\\) (as the risk of rejecting \\(H_0\\) is always 0). We adopt the log-likelihood ratio:

$$
\text{LLR} = \frac{1}{n}\sum_{i=1}^n\log\frac{p_{\pi}(\tau_i)}{p_{\pi_{\text{old}}}(\tau_i)},
$$

where 

$$
p_{\pi}(\tau_i) = p(s_i^1)\prod_{t=1}^T \pi(a_i^t\vert s_i^t) T(s_i^{t+1}\vert s_i^t,a_i^t).
$$

is the probability of \\(\tau_i\\) under policy \\(\pi\\) with transition probility \\(T(\cdot\vert s,a)\\). Hence, we have the limit of \\(\text{LLR}\\) as

$$
\lim_{n\to\infty}\text{LLR} = \lim_{n\to\infty}\frac{1}{n}\sum_{t=1}^{T_i}\sum_{i=1}^n\log\frac{\pi(a_i^{t}\vert s_i^t)}{\pi_{\text{old}}(a_i^{t}\vert s_i^t)} = D_{KL}(\pi\Vert \pi_{\text{old}}),
$$

where

$$
D_{KL}(\pi\Vert \pi_{\text{old}}) = \mathbb{E}_{(s,a)\sim\rho_\pi}\left[\frac{\pi(a\vert s)}{\pi_{\text{old}}(a\vert s)}\right],
$$

with \\(\rho_\pi=d_{\pi}(s)\pi(a\vert s)\\) and \\(d_{\pi}(s)\\) is the stable state marginal distribution of policy \\(\pi\\). Therefore, we can view the KL divergence of a natural constraint of policy in the view of information theory.

<!-- Therefore, we can find that the risk of acceptance is actually equalt to the KL divergence of marginal distribution of state-action joint pair $(s,a)$ under different policy. -->

Futher thoughts on KL divergence
======

Maybe some better constraints?
------
For Bayesian testing (minimizing total error $P_e = \pi_0\alpha+\pi_1\beta$, where $\pi_0$ and $\pi_1$ are priors, the error decays as:

$$
\lim_{n\to\infty}\frac{1}{n}P_e = -C(\mu,\nu).
$$

Here, $C(\mu,\nu)$ is named as Chernoff information, which is related to KL divergence:

$$
C(\mu,\nu) = \sup_{t\in[0,1]}\left(-log\int_{\mathcal{X}}\mu^{t}(x)\nu^{1-t}(x)\mathrm{d}x\right) \leq \min\left(D_{\text{KL}}(\mu,\nu),D_{\text{KL}}(\nu,\mu)\right).
$$

So, can we use Chernoff information as one constraint?

What is weighted constraint?
------
The KL divergence can be viewed as constraint on each case equally. However, there can be a case that when the state $s$
explored by the new policy \\(\pi\\) is good (e.g. value function \\(V(s)\\) or reward \\(r(s, \pi(s))\\) is high), we may want to loose the constraint to this state. Therefore, can we adopt a weighted constraint on the log-likelihood ratio?