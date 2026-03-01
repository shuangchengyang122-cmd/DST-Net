# DST-Net: Deep Spatiotemporal Synergy Network for Lithology Identification

This repository contains the official reproducible source code for the manuscript: 
**"A Deep Spatiotemporal Synergy Network for Identifying Thin Interbeds and Transitional Lithologies in Coal Measures"** submitted to *Computers & Geosciences*.

## Overview
DST-Net is a deep learning framework designed to solve the bottlenecks of boundary smearing and context loss in conventional lithology identification. By synergistically coupling a 1D-CNN (for extracting local morphological gradients) with an LSTM (for modeling global sedimentary cycles), DST-Net achieves high-precision identification of complex transitional lithologies and sub-meter thin interbeds.

## Requirements & Installation
To ensure reproducibility, please install the required dependencies:
```bash
pip install -r requirements.txt