# 🐍 Reinforcement Learning Snake AI (PyTorch)

## 📌 Description

This project implements a Deep Reinforcement Learning agent using PyTorch to autonomously learn how to play the classic Snake game. The agent is trained using a Deep Q-Network (DQN) approach, leveraging experience replay and reward-based learning to progressively improve its gameplay strategy.

A custom game environment was built using Pygame, providing real-time state feedback, collision detection, and performance visualization. Over multiple training episodes, the agent learns optimal movement policies to maximize survival time and food collection.

This project demonstrates core reinforcement learning concepts including Q-learning, neural network function approximation, and exploration vs. exploitation tradeoffs.

---

## 🧠 Learning Pipeline Explanation

The system follows a standard Deep Q-Learning pipeline:

### 1. Environment (Snake Game)

* Built using Pygame for real-time simulation
* Tracks:

  * Snake position and movement direction
  * Food location
  * Collision with walls or itself
* Provides:

  * Reward (+10 for eating food, -10 for collision)
  * Game termination conditions

---

### 2. State Representation

Each game step is converted into an 11-dimensional state vector including:

* Immediate dangers (straight, left, right)
* Current movement direction (one-hot encoded)
* Relative food position (left, right, up, down)

This compact representation allows efficient learning while preserving essential environment information.

---

### 3. Deep Q-Network (DQN)

Neural network structure:

* Input layer: 11 features
* Hidden layer: 256 neurons (ReLU activation)
* Output layer: 3 actions (straight, right turn, left turn)

The network predicts Q-values for each possible action.

---

### 4. Training Strategy

* **Epsilon-Greedy Policy**:
  Balances random exploration and learned behavior

* **Experience Replay Buffer**:
  Stores past transitions for batch training

* **Discount Factor (γ)**:
  Encourages long-term reward optimization

* **Loss Function**:
  Mean Squared Error (MSE) between predicted and target Q-values

---

### 5. Continuous Improvement

* Short-term learning occurs each step
* Long-term learning occurs through batch sampling
* Best-performing models are saved automatically

---

## ▶️ Usage Instructions

### Requirements

Make sure you have Python 3.8+ installed.

Install dependencies:

```bash
pip install torch pygame numpy matplotlib
```

---

### Run Training

Start the reinforcement learning agent:

```bash
python snake_agent.py
```

The game window will open and the AI will begin training in real time.
Performance metrics will be plotted as training progresses.

---

### Model Saving

Trained models are automatically saved to:

```bash
./model/model.pth
```

---

## 📂 Project Structure

```
├── snake_agent.py    # Reinforcement learning agent logic
├── snake_game.py     # Pygame-based Snake environment
├── snake_train.py   # Neural network + training methods
├── utils.py         # Plotting utilities
├── model/
│   └── model.pth    # Saved trained model
└── README.md
```

---

## 🚀 Key Features

✔️ Deep Q-Learning with PyTorch
✔️ Custom real-time simulation environment
✔️ Experience replay for stable training
✔️ Automatic model checkpointing
✔️ Performance visualization

---

## 📊 Future Improvements (Recommended)

* Implement target networks for improved training stability
* Add convolutional input for vision-based learning
* Hyperparameter tuning automation
* Support for curriculum learning or larger environments
* Export trained model for inference-only play mode

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Reinforcement Learning (DQN)
* Pygame
* NumPy

---

## 📈 Learning Outcomes

This project demonstrates:

* Practical reinforcement learning implementation
* Neural network-based policy learning
* Simulation environment design
* Training loop optimization techniques
* AI decision-making in dynamic environments
