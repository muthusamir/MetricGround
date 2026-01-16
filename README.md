# MetricGround: Bridging the Dimensionality Gap for Metric-Aware Embodied Vision-Language Models

Official implementation of the paper:  
**MetricGround: Bridging the Dimensionality Gap for Metric-Aware Embodied Vision-Language Models**

This repository contains the PyTorch implementation of MetricGround – a modular architecture that enables metric-aware 3D grounding in pre-trained Vision-Language Models with minimal disturbance.

## Key Features
- Frozen 2D VLM backbone + lightweight adapters
- Semantic-Metric Fusion via cross-attention
- Language-mediated cross-modal distillation
- Support for SQA (Spatial Question Answering) with metric outputs

## Installation

```bash
git clone https://github.com/yourusername/MetricGround.git
cd MetricGround
pip install -r requirements.txt
