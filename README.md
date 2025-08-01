# Real-Time-6-DOF-Ship-Motion-Prediction

**Generalization and Comparative Evaluation of RNN-Based Deep Learning Models for Real-Time 6-DOF Ship Motion Prediction**

---

## 📌 Overview

This repository contains the official PyTorch implementation of the paper:  
**"Generalization and Comparative Evaluation of RNN-Based Deep Learning Models for Real-Time 6-DOF Ship Motion Prediction."**

The project evaluates the generalization performance of various RNN-based deep learning models for multivariate time-series prediction of ship motion with six degrees of freedom (6-DOF).

---

## 🔧 Research Pipeline

A schematic overview of the research process is shown below:

<img width="1794" height="863" alt="Research Pipeline" src="https://github.com/user-attachments/assets/045fd7a5-63b8-4664-b143-1f0d2fbf45ae" />

---

## 🧠 Models

The following RNN-based models are implemented and compared:

- Vanilla RNN  
- LSTM (Long Short-Term Memory)  
- GRU (Gated Recurrent Unit)  
- Bi-LSTM (Bidirectional LSTM)  

All model definitions are located in [`models/models.py`](models/models.py).

---

## 📊 Dataset Description

The dataset used for training and evaluation was synthetically generated under various sea states using simulation. Below are the key properties:

| Parameter                         | Values                                                                      |
|----------------------------------|-----------------------------------------------------------------------------|
| Mean period, *T*<sub>mean</sub> [s]        | 7.5 &nbsp;&nbsp; 9.5 &nbsp;&nbsp; 11.5 &nbsp;&nbsp; 7.5 &nbsp;&nbsp; 9.5 &nbsp;&nbsp; 11.5 &nbsp;&nbsp; 7.5 &nbsp;&nbsp; 9.5 &nbsp;&nbsp; 11.5 |
| Propagation direction, χ [deg]   | 90 &nbsp;&nbsp; 90 &nbsp;&nbsp; 90 &nbsp;&nbsp; 135 &nbsp;&nbsp; 135 &nbsp;&nbsp; 135 &nbsp;&nbsp; 180 &nbsp;&nbsp; 180 &nbsp;&nbsp; 180         |
| Significant height, *H*<sub>S</sub> [m]     | 3.5                                                                         |
| Simulation time [s]              | 10,000 for each condition                                                   |
| Time step, *dt* [s]              | 0.01                                                                        |


This configuration results in a total of 9 different sea conditions.

---

## 🛠 Requirements

To install all required Python packages, simply run:

```bash
pip install -r requirements.txt
