# Iterative Beran EOT

**KL-Regularized Entropic Optimal Transport for Adaptive Survival Analysis with Censored Data**

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Content

- Classic **Beran estimator** implementation (`beran_baseline.py`)
- Proposed **Iterative Beran + KL-regularized EOT** algorithm (`beran_iterative_k_ti.py`)
- Survival function model with adaptive weighting (`survival_function_model.py`)
- Demo script with visualization and comparison (`demo.py`)
- Experiment saving and loading utilities (`save_results.py`)
- Fully reproducible `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NTAILab/Iterative_Beran_EOT.git
   cd Iterative_Beran_EOT
