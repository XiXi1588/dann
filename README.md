# DANN Implementation
This repository contains the DANN  domain adaptation method  for milling deformation prediction from [An online milling deformation prediction method for thin-walled features with domain adversarial neural networks under small samples].

## Setup
1. Clone the repo: `git clone [URL]`
2. Install dependencies: `pip install -r requirements.txt`

## Data
- Mock datasets are in the manuscript appendix.
- Format: 4 columns (3 features + 1 label), 120 source, 60 target samples.
- Place files in `D:/data/` (or modify path in `dann.py`).

## Running
- Execute: `python code/dann.py`
- Outputs saved to `D:/data/`.

## Requirements
- Hardware: Intel Core i5-9400 CPU, 8 GB RAM.
