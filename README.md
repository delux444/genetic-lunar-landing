# Lunar Lander: Touch-and-Go Neuroevolution

This project implements an Autonomous Flight Pilot for the LunarLander-v3 environment using Genetic Algorithms (GA) to train a Neural Network.
Unlike standard Reinforcement Learning, this uses Neuroevolution to optimize the weights and biases of the pilot's brain.

The pilot is specifically trained for a "Touch-and-Go" mission:
landing between the flags, taking off again, and maintaining stability.


https://github.com/user-attachments/assets/6a4abc2a-7557-4b4c-923f-c83dc36a3c88

---

## Installation

1. Clone the Repository

``` bash
git clone https://github.com/yourusername/lunar-lander-neuroevolution.git
```
``` bash
cd lunar-lander-neuroevolution
```

2. Create a Virtual Environment (Recommended)

On Linux/macOS:

``` bash
python3 -m venv venv
```

``` bash
source venv/bin/activate
```

3. Install Dependencies

```bash
pip install --upgrade pip
```
```bash
sudo dnf/apt/yay install SDL2-devel SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel portmidi-devel libX11-devel
```

``` bash
pip install gymnasium "gymnasium[box2d]" numpy pygad swig
```

4. Start

``` bash
python main.py
```

---

## The Environment

We use gymnasium's LunarLander-v3. The state space consists of 8 variables, including coordinates, velocity, angle, and leg contact sensors.
The action space consists of 4 discrete actions: Do nothing, Fire Left Engine, Fire Main Engine, Fire Right Engine.

## The Neural Network

The pilot's brain is a simple Artificial Neural Network:

 - Input Layer (8): Receives landing data.
 - Hidden Layer (16): Process data using ReLU activation.
 - Output Layer (4): Determines the best action via Argmax.

   The weights and biases are flattened into a "Chromosome" (vector of genes) for the Genetic Algorithm to manipulate.

---

## Fitness Function ("Touch-and-Go")

The fitness_func determines how "fit" a pilot is. In this version, we modified the standard rewards:

- Base Reward: Standard Gym rewards for stability.
- Touchdown Bonus: A massive +100 bonus if the legs touch the ground for the first time.
- Ascent Reward: Extra points if the pilot successfully gains altitude (obs[1] > 0.2) after the initial touchdown.

---

## Multithreading

To prevent AssertionError in Box2D, the code uses threading.local().
This ensures that every thread has its own isolated instance of the Gymnasium environment, allowing for safe 16-core parallel processing.

---

| Autor | Profil GitHub |
| :--- | :--- |
| Michał Figołuszka | [github.com/Michaleq24](https://github.com/Michaleq24) |
| Karol Bieżuński | [github.com/delux444](https://github.com/delux444) |
