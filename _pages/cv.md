---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* Ph.D in Tsinghua Shenzhen International Graduate School (SIGS), Tsinghua University, 2026 (expected)
* B.Eng in Chien-Shiung Wu College, Southeast University, 2021

Work experience
======
* Spring 2024: Research Internship
  * Meituan Co., Ltd
  * Duties includes: Develop a training pipeline for a large decision model tailored to fuzz testing.
    * Jobs: Designed and implemented 1B / 2B / 7B parameter models from scratch, including optimization of mixed precision training pipelines (dataset, trainer and model, etc.) and deployment of low-latency serving infrastructure.
    * Results: our approach achieves over 8 times more vulnerability discovery efficiency compared with traditional expert-guided random-walk exploitation.
  * Award: [卓越实践奖](https://mp.weixin.qq.com/s/UoUYaNAbJGxQu1kt0oNkYg)
  * Mentor: Li Zeng (Bill)
  
Skills
======
* Programming
  * MATLAB
  * Python
  * Golang 
* Robotics
  * ROS
  * Gazebo
  * ISSCA gym
* LLM
  * Transformer

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Teaching Assistant of TBSI Course *Advanced Signal Processing* (2025/02-2025/06)
* Teaching Assistant of TBSI Course *Advanced Signal Processing* (2024/02-2024/06)
* Teaching Assistant of TBSI Course *Learning from Data* (2022/09-2023/01)
