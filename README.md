# OceanGo

**OceanGo** is a research project designed to recreate the core architecture of AlphaGo and AlphaZero on a smaller, consumer-accessible scale. 

## Academic Context
This project was developed as a **Final Year Research Project** for the **Bachelor of Computer Science** degree at the **University of Windsor**. 

It was created during a pivotal time in technology—the rapid rise of Artificial Intelligence and Large Language Models (LLMs). The primary motivation for this research is to investigate and build a profound, hands-on understanding of foundational AI methodologies beyond language models, specifically **Monte Carlo Tree Search (MCTS)** and **Policy-Value Neural Networks**.

## Project Intent
The overarching goal of OceanGo is to implement an AlphaGo-like algorithm capable of playing the ancient board game Go, configured to run on accessible hardware. 

While DeepMind's Alpha systems required massive clusters of custom TPUs to train, OceanGo establishes a framework scaled to train and execute on a **single computer with a single CPU and GPU**. It proves that the mathematical principles and architectures of AlphaZero can be structured and understood within the scope of a bachelor's degree timeline using individual consumer hardware.

## Key Technical Features
The theoretical backbone of this engine replicates modern reinforcement learning techniques:
* **Deep Residual Network (ResNet)**: A deep convolutional neural network utilizing residual blocks to evaluate board states, outputting both move probabilities (Policy Head) and win/loss predictions (Value Head).
* **State History & Input Planes**: Feeds the neural network the last 8 board states along with player perspective channels, granting the AI an understanding of "momentum" and complex situational tactics like the Ko rule.
* **Monte Carlo Tree Search (MCTS)**: Advanced forecasting of possible move outcomes utilizing PUCT (Predictor Upper Confidence Bounds).
* **Batched/Parallel MCTS with Virtual Loss**: Enables concurrent evaluations of unique tree search paths on the GPU, maximizing hardware efficiency without bottlenecking in Python loops.
* **Self-Play Enhancements**: Utilizes **Dirichlet Noise** for divergent baseline exploration, **Temperature Decay** for shifting from exploration to exploitation, and **Symmetrical Data Augmentation** (rotations and reflections) to dynamically multiply the training experience.
* **Rigorous Go Environment**: Fully implemented underlying rules engine dictating liberties, chained captures, positional superko, pass detection, and accurate Chinese area scoring incorporating Komi.

## Extensibility
As a degree research project, OceanGo serves as a highly modular foundation. It is fully intended to be extended in the future. Potential extensions include scaling up the neural structure for a 19x19 board, building distributed data-generation pipelines to offload self-play across multiple machines, or rewriting the MCTS node traversal engine into C++ for rapid localized tree expansion.

---
*Developed for the University of Windsor Computer Science Program.*