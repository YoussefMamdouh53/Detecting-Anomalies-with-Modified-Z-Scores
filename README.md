# Real-Time Anomaly Detection with Modified Z-Score

This project implements a real-time anomaly detection system using the **Modified Z-Score** method. It processes an infinite data stream, detects anomalies, and visualizes the results live using `matplotlib` for plotting and `OpenCV` for live display.

## Table of Contents
- [Overview](#overview)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)

## Overview
This project simulates a data stream with seasonal variations and noise. It uses the Modified Z-Score technique to identify outliers (anomalies) and visualizes them in real time. The anomalies are displayed in red on a live plot, while normal data points are in white.

## Algorithm

### Modified Z-Score:
The **Modified Z-Score** method is used to detect anomalies. It is calculated as:
Modified Z-Score = 0.6745 * (x - median) / MAD

Where:
- `x` is the current data point.
- `median` is the median of the sliding window of past data points.
- `MAD` is the **Median Absolute Deviation** of the data in the window.

### Why Modified Z-Score?
This method is robust to noise and outliers because it relies on the median and MAD, which are less sensitive to extreme values compared to the mean and standard deviation.

## Installation

### Prerequisites:
- Python 3.x
- Install required libraries:

```bash
pip install numpy matplotlib opencv-python
```

## Usage

```bash
py main.py
```
