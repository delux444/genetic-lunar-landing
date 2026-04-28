# Lunar Lander: Genetic Algorithm Optimization

An autonomous agent trained to land on the Moon using a Genetic Algorithm within the OpenAI Gymnasium environment. This project demonstrates the power of evolutionary strategies in solving control problems without gradient-based learning.

---

## Project Overview

The goal of this project is to evolve a controller capable of performing a safe landing in the LunarLander environment. Instead of traditional Reinforcement Learning (RL), we use an evolutionary approach to optimize the controller's decision-making parameters.

## Tech Stack

Language: Python 3.x
Simulation: Gymnasium (Lunar Lander v3)
GA Framework: PyGAD
Visualization: Matplotlib

## How It Works

Genome: Each individual represents a set of control parameters (gains) for the lander's engines.
Fitness Function: Agents are evaluated based on the total reward accumulated during a simulation run (rewarding soft landings and penalizing crashes).

## Evolutionary Loop:

Selection: Top-performing agents are selected as parents.
Crossover: Parent genes are combined to create offspring.
Mutation: Random variations are introduced to explore new maneuvers.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/lunar-lander-ga.git
```
```bash
# Install dependencies
pip install gymnasium[lunar-lander] pygad matplotlib
```
```bash
# Run the demonstration
python main.py
```
---

## Authors
Michał Figołuszka - https://github.com/Michaleq24

Karol Bieżuński - https://github.com/delux444
