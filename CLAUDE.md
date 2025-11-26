# CLAUDE.md - Multi-Agent Decision Game

> **Project Context for AI Assistants**: This document provides comprehensive guidance for understanding and working with the Multi-Agent Decision Game codebase. Read this before making any code changes.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Tech Stack & Dependencies](#tech-stack--dependencies)
5. [Key Concepts & Algorithms](#key-concepts--algorithms)
6. [Code Conventions](#code-conventions)
7. [Development Workflows](#development-workflows)
8. [Common Tasks](#common-tasks)
9. [Important Implementation Details](#important-implementation-details)
10. [Testing & Validation](#testing--validation)

---

## 🎯 Project Overview

### Purpose
A multi-agent reinforcement learning environment that explores decision-making under uncertainty through:
- **Reinforcement Learning** (PPO algorithm for agent movement)
- **Monte Carlo Simulation** (strategic decision-making under uncertainty)
- **Deep Learning Perception** (CNN-based environment understanding)
- **Bayesian Opponent Modeling** (inspired by Werewolf Game theory)
- **Deception Detection** (identifying strategic behavioral patterns)

### Core Capabilities
- Grid-based environment with resources, obstacles, and multiple agents
- Agents learn navigation through RL training
- Strategic decision-making using Monte Carlo simulation
- Multi-agent competition with opponent modeling
- Visual real-time gameplay using Pygame

### Project Maturity
**Status**: MVP Complete with Advanced Features
- ✅ Basic grid world environment
- ✅ RL training pipeline (PPO)
- ✅ Monte Carlo decision system
- ✅ Deep CNN perception
- ✅ Werewolf-inspired opponent modeling
- ✅ Multi-agent integration (v3)

---

## 🏗️ Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Visualization Layer                   │
│                    (Pygame + UI)                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Integration Layer                      │
│            (IntegratedGameV3, Game Logic)                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌───────────────┬──────────────────┬──────────────────────┐
│  Strategic    │   Opponent       │   Perception         │
│  Decision     │   Modeling       │   System             │
│  (Monte       │   (Bayesian +    │   (CNN)              │
│   Carlo)      │    Deception)    │                      │
└───────────────┴──────────────────┴──────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                     RL Control Layer                     │
│                  (PPO Agent + Policy)                    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    Environment Layer                     │
│              (AgentDecisionEnv - Gymnasium)              │
└─────────────────────────────────────────────────────────┘
```

### Component Interaction

1. **Environment** (`rl_environment.py`) provides the base grid world
2. **RL Agent** (`train_agent.py`) learns movement policies via PPO
3. **Perception** (`deep_perception.py`) processes grid state into features
4. **Strategic Layer** (`monte_carlo_decisions.py`) makes high-level decisions
5. **Opponent Model** (`werewolf_opponent_model.py`) tracks and predicts opponent behavior
6. **Integration** (`integrated_game_v*.py`) combines all systems
7. **Visualization** (Pygame) renders state and decisions in real-time

---

## 📁 File Structure

### Directory Organization

```
agent_decision_game/
├── README.md                          # User-facing documentation
├── CLAUDE.md                          # This file - AI assistant guide
├── LICENSE                            # MIT License
└── scripts/                           # All Python modules
    ├── game_world.py                  # Basic Pygame grid world (Phase 1)
    ├── rl_environment.py              # Gymnasium RL environment (Phase 2)
    ├── train_agent.py                 # PPO training script
    ├── train_longer.py                # Extended training (100k steps)
    ├── visualize_agent.py             # Watch trained agents play
    ├── compare_agents.py              # Agent performance comparison
    ├── monte_carlo_decisions.py       # Monte Carlo decision framework
    ├── deep_perception.py             # CNN-based perception network
    ├── werewolf_opponent_model.py     # Bayesian opponent modeling
    ├── integrated_game.py             # Basic integration (RL + MC)
    ├── integrated_game_v1.py          # Version 1 integration
    ├── integrated_game_v2.py          # Enhanced integration
    ├── integrated_game_v3.py          # Full integration (all systems)
    ├── check_setup.py                 # Installation verification
    └── game.py                        # (Empty/minimal - legacy)
```

### File Naming Conventions

- **Base modules**: `{component}_{purpose}.py` (e.g., `rl_environment.py`)
- **Scripts**: `{action}_{target}.py` (e.g., `train_agent.py`, `visualize_agent.py`)
- **Versioned integrations**: `{name}_v{number}.py` (v3 is latest)
- **Utilities**: `{action}_{purpose}.py` (e.g., `check_setup.py`)

### Version History

- `game_world.py` → Basic manual control
- `integrated_game.py` → RL + basic Monte Carlo
- `integrated_game_v1.py` → Improved integration
- `integrated_game_v2.py` → Added strategic decisions
- `integrated_game_v3.py` → **CURRENT** - Full AI integration with opponent modeling

---

## 🛠️ Tech Stack & Dependencies

### Required Libraries

```python
# Core RL & ML
gymnasium==0.29.1          # RL environment API (replaces gym)
stable-baselines3==2.1.0   # PPO algorithm implementation
torch>=2.0.0               # Deep learning (CNN, LSTM)
numpy>=1.24.0              # Numerical operations

# Visualization
pygame>=2.5.0              # Game rendering and UI
matplotlib>=3.7.0          # Plotting and analysis

# Utilities
dataclasses                # Standard library (Python 3.7+)
collections                # Standard library
typing                     # Standard library
```

### Installation Command

```bash
pip install gymnasium stable-baselines3 pygame torch numpy matplotlib
```

### Python Version
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: Python 3.10, 3.11

---

## 🧠 Key Concepts & Algorithms

### 1. Reinforcement Learning (PPO)

**Implementation**: `train_agent.py`, `rl_environment.py`

```python
# Key parameters
algorithm: PPO (Proximal Policy Optimization)
policy: MlpPolicy (Multi-layer Perceptron)
learning_rate: 0.0003
n_steps: 2048
batch_size: 64
total_timesteps: 50000 (initial), 100000+ (extended)
```

**Observation Space**: `Box(6,)` containing:
- Agent position (x, y)
- Nearest resource position (x, y)
- Nearest obstacle position (x, y)

**Action Space**: `Discrete(5)`
- 0: UP
- 1: DOWN
- 2: LEFT
- 3: RIGHT
- 4: STAY

**Reward Structure**:
- `+10`: Collect a resource
- `-5`: Hit an obstacle
- `-0.1`: Each step (encourages efficiency)

### 2. Monte Carlo Decision Making

**Implementation**: `monte_carlo_decisions.py`

**Purpose**: High-level strategic decisions under uncertainty

**Key Concept**: Simulate thousands of possible outcomes to estimate expected value

```python
# Decision structure
Decision(
    name: str,                      # e.g., "Aggressive Exploration"
    cost: float,                    # Action cost
    uncertain_parameters: dict,     # e.g., {'success_rate': (0.4, 0.8)}
    category: str                   # e.g., "exploration", "resource"
)
```

**Simulation Process**:
1. Run 100-10,000 simulations per decision
2. Sample uncertain parameters from ranges
3. Calculate outcomes based on scenario
4. Compute statistics: mean, std, percentiles
5. Choose decision with best expected value

**Strategies**:
- **Aggressive**: High risk, high reward (low success rate)
- **Balanced**: Medium risk, medium reward
- **Conservative**: Low risk, guaranteed outcomes (high success rate)
- **Counter-Deceptive**: Response to detected deception

### 3. Deep CNN Perception

**Implementation**: `deep_perception.py`, `integrated_game_v3.py`

**Architecture**:
```
Input: (batch, 5, 10, 10) grid channels
    ↓
Conv2d(5 → 32, k=3) + ReLU
    ↓
Conv2d(32 → 64, k=3) + ReLU
    ↓
Conv2d(64 → 64, k=3) + ReLU
    ↓
AdaptiveAvgPool2d(5, 5)
    ↓
Flatten → Linear(1600 → 256) + Dropout
    ↓
Linear(256 → 128) + Dropout
    ↓
Linear(128 → 64) = Feature Vector
```

**Input Channels** (5 channels):
- **Channel 0**: Self position (binary mask)
- **Channel 1**: Resources (binary mask)
- **Channel 2**: Obstacles (binary mask)
- **Channel 3**: Other agents (binary mask)
- **Channel 4**: Distance/influence map (continuous values)

**Output**: 64-dimensional feature vector representing spatial understanding

### 4. Werewolf-Inspired Opponent Modeling

**Implementation**: `werewolf_opponent_model.py`, `integrated_game_v3.py`

**Inspiration**: Based on Werewolf Game (Mafia) theory where players must identify hidden roles through behavioral analysis.

**Core Components**:

#### A. Bayesian Belief Updates
```python
# Prior beliefs about opponent strategy
beliefs = {
    'aggressive': 0.25,
    'balanced': 0.25,
    'conservative': 0.25,
    'deceptive': 0.25
}

# Update: P(strategy|observation) = P(observation|strategy) * P(strategy) / P(observation)
posterior = (likelihood * prior) / evidence
```

#### B. Deception Detection
Identifies agents with inconsistent behavior patterns:
- **Pattern shifts**: KL divergence between early/late behavior
- **Avoidance patterns**: Selective targeting
- **Coordination detection**: Synchronized actions

**Indicators**:
- Deception score > 0.5 → likely deceptive
- High confidence (>0.7) → take action

#### C. Strategic Revelation
Based on Werewolf Game's optimal information revelation timing:
- **Early game** (R1-R2): Hold information unless certainty > 0.7
- **Mid game** (R3): Reveal if certainty > 0.6
- **End game** (R4+): Reveal all information

### 5. Grid Environment Mechanics

**Grid Size**: 10×10 (default, configurable)
**Cell Size**: 60 pixels (for rendering)
**Max Steps**: 100 per episode

**Entity Generation**:
- 5 resources (random positions)
- 8 obstacles (random positions)
- 1-3 agents (configurable)

**Movement Rules**:
- Boundary constrained (can't move off grid)
- Obstacles block movement (position reverted)
- Resources consumed on contact (removed from grid)

---

## 📝 Code Conventions

### Style Guidelines

#### Naming Conventions
```python
# Classes: PascalCase
class AgentDecisionEnv:
class MonteCarloSimulator:
class DeepPerceptionNetwork:

# Functions/Methods: snake_case
def train_agent():
def get_observation():
def analyze_decision():

# Constants: UPPER_SNAKE_CASE
GRID_SIZE = 10
FPS = 5
MAX_EPISODES = 100

# Private methods: _leading_underscore
def _compute_likelihood():
def _get_observation():
```

#### Type Hints
Use type hints for function signatures:
```python
from typing import Dict, List, Tuple, Optional

def analyze_decision(self, decision: Decision, scenario: dict) -> dict:
    pass

def get_action(self, env: AgentDecisionEnv) -> Tuple[int, Optional[dict]]:
    pass
```

#### Dataclasses
Use dataclasses for structured data:
```python
from dataclasses import dataclass

@dataclass
class Decision:
    name: str
    cost: float
    uncertain_parameters: dict
    category: str = "general"
```

### Documentation Standards

#### Docstrings
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief one-line description.

    Longer description if needed. Explain purpose, not implementation.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
```

#### Module Headers
Every major module starts with:
```python
"""
MODULE NAME
===========
Brief description of module purpose.

Key components:
- Component 1: Description
- Component 2: Description
"""
```

### Code Organization Patterns

#### Class Structure
```python
class ClassName:
    """Class docstring"""

    # 1. Class variables
    DEFAULT_VALUE = 10

    def __init__(self, params):
        """Initialize"""
        # 2. Instance variables
        self.param = params

    # 3. Public methods
    def public_method(self):
        """Public interface"""
        pass

    # 4. Private methods
    def _private_method(self):
        """Internal implementation"""
        pass
```

#### Import Order
```python
# 1. Standard library
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# 2. Third-party packages
import torch
import pygame
from stable_baselines3 import PPO

# 3. Local modules
from rl_environment import AgentDecisionEnv
from monte_carlo_decisions import MonteCarloSimulator
```

### Color Conventions (Pygame)

```python
# Standard colors
WHITE = (255, 255, 255)    # Background
BLACK = (0, 0, 0)          # Text, borders
GRAY = (200, 200, 200)     # Grid lines

# Entity colors
BLUE = (0, 100, 255)       # Agent 1
ORANGE = (255, 165, 0)     # Agent 2
PURPLE = (150, 0, 200)     # Agent 3

GREEN = (100, 255, 100)    # Resources
RED = (255, 100, 100)      # Obstacles

# Strategy indicators
YELLOW = (255, 255, 0)     # Balanced strategy
PINK = (255, 192, 203)     # Deception warning
```

---

## 🔄 Development Workflows

### Workflow 1: Adding a New RL Agent

**Steps**:
1. **Define environment** (`rl_environment.py`)
   - Modify observation space if needed
   - Adjust reward structure
   - Test with `check_env(env)`

2. **Train agent** (`train_agent.py`)
   ```python
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=50000)
   model.save("model_name")
   ```

3. **Test agent** (`visualize_agent.py`)
   - Load model: `model = PPO.load("model_name")`
   - Run episodes visually
   - Verify behavior

4. **Benchmark** (`compare_agents.py`)
   - Compare with random baseline
   - Collect performance metrics

### Workflow 2: Implementing New Decision Strategy

**Steps**:
1. **Define decision** (`monte_carlo_decisions.py`)
   ```python
   new_decision = Decision(
       name="Strategy Name",
       cost=3.0,
       uncertain_parameters={'success_rate': (0.5, 0.9)},
       category="exploration"
   )
   ```

2. **Create scenario**
   ```python
   scenario = {
       'reward': 15,
       'environment': 'uncertain',
       'competition_level': 'high'
   }
   ```

3. **Analyze**
   ```python
   results = simulator.analyze_decision(new_decision, scenario)
   ```

4. **Integrate** into agent strategy selection (`integrated_game_v3.py`)

### Workflow 3: Creating New Integrated Game Version

**When to create new version**:
- Adding major new AI system
- Significant architecture changes
- Breaking changes to existing integration

**Process**:
1. **Copy latest version**: `cp integrated_game_v3.py integrated_game_v4.py`
2. **Update docstring** with changes
3. **Modify classes/functions** as needed
4. **Test thoroughly** before considering stable
5. **Update README.md** to reference new version

### Workflow 4: Modifying Opponent Model

**Components to update**:

1. **Belief structure** (`WerewolfOpponentModel.__init__`)
   - Add new strategy types
   - Adjust prior probabilities

2. **Likelihood computation** (`_compute_likelihood`)
   - Define P(observation | strategy)
   - Add new behavioral patterns

3. **Deception detection** (`_update_deception_score`)
   - Add new detection heuristics
   - Tune thresholds

4. **Counter-strategies** (`_determine_counter_strategy`)
   - Map beliefs to counter-actions
   - Consider game phase

---

## 🎯 Common Tasks

### Task 1: Train an Agent from Scratch

```bash
# 1. Verify setup
python scripts/check_setup.py

# 2. Train initial agent (50k steps, ~5 minutes)
python scripts/train_agent.py
# Saves: trained_agent.zip

# 3. Train longer (100k more steps, ~10 minutes)
python scripts/train_longer.py
# Saves: trained_agent_v2.zip

# 4. Visualize
python scripts/visualize_agent.py
```

### Task 2: Run Full Integrated Game

```bash
# Latest version with all features
python scripts/integrated_game_v3.py

# Controls:
# - SPACE: Pause/Resume
# - R: Reset episode
# - D: Toggle debug info
# - Q: Quit
```

### Task 3: Analyze Decision Strategies

```bash
# Run Monte Carlo analysis
python scripts/monte_carlo_decisions.py

# Generates:
# - Console output with statistics
# - decision_analysis.png (if matplotlib available)
```

### Task 4: Test Opponent Modeling

```bash
# Run Werewolf-inspired opponent modeling demo
python scripts/werewolf_opponent_model.py

# Shows:
# - Belief updates over 40 rounds
# - Deception detection
# - Strategic recommendations
```

### Task 5: Compare Agent Performance

```bash
# Compare trained agent vs random baseline
python scripts/compare_agents.py

# Metrics:
# - Average reward
# - Resources collected
# - Success rate
# - Episode length
```

### Task 6: Modify Grid Environment

**File**: `rl_environment.py`

```python
# Change grid size
def __init__(self, grid_size=15):  # Was 10

# Adjust resource count
self.resources = [... for _ in range(8)]  # Was 5

# Modify reward structure
reward = 20  # Was 10 for resource collection
reward = -0.2  # Was -0.1 for step penalty
```

**Remember**: After environment changes, retrain agents!

### Task 7: Add New Agent Type

**File**: `integrated_game_v3.py`

```python
# In IntegratedGameV3.__init__
agent_strategies = ["Balanced", "Aggressive", "Conservative", "YourNewType"]

# Add strategy decision
decisions.append(
    Decision("Your New Type", cost=X,
            uncertain_parameters={...},
            category="your_category")
)

# Add color mapping
strategy_colors = {
    'YourNewType': (R, G, B)
}
```

---

## 🔍 Important Implementation Details

### 1. Model Persistence

**Location**: Root directory (where scripts run)

```python
# Saving
model.save("trained_agent")  # Creates trained_agent.zip

# Loading
model = PPO.load("trained_agent")  # Loads trained_agent.zip
```

**Important**: When loading models in scripts, ensure correct path:
```python
try:
    self.rl_model = PPO.load("trained_agent")
except:
    print(f"Could not load model for agent {agent_id}")
```

### 2. Observation Space Compatibility

**Critical**: Observation space must match between training and inference!

```python
# Training (rl_environment.py)
self.observation_space = spaces.Box(low=0, high=grid_size, shape=(6,), dtype=np.float32)

# Inference (integrated_game_v3.py)
# Must also provide 6-dimensional observation!
observation = env._get_observation()  # Returns (6,) array
```

**When using CNN perception**:
- Training: Use `DeepPerceptionEnv` (64-dim features)
- Inference: Use same network, ensure consistent preprocessing

### 3. PyTorch vs NumPy Arrays

**Rule**: Convert between types at system boundaries:

```python
# Creating grid for CNN
grid_channels = np.zeros((5, GRID_SIZE, GRID_SIZE), dtype=np.float32)
grid_tensor = torch.FloatTensor(grid_channels).unsqueeze(0)

# After CNN processing
features = self.perception_net(grid_tensor)
return features.squeeze(0).numpy()  # Back to NumPy for RL
```

### 4. Pygame Event Loop Structure

**Pattern**: All Pygame games follow this structure:

```python
def run(self):
    while self.running:
        # 1. Handle events
        self.handle_events()

        # 2. Update game state
        if not self.paused:
            self.step_game()

        # 3. Render
        self.draw_grid()
        self.draw_info_panel()

        # 4. Update display
        pygame.display.flip()
        self.clock.tick(FPS)

    pygame.quit()
```

### 5. Episode Termination Conditions

```python
# Episode ends when:
done = (
    len(self.resources) == 0        # All resources collected
    or self.steps >= self.max_steps  # Max steps reached
)
```

### 6. Resource Collection Detection

```python
# Check if agent position matches resource position
for i, resource in enumerate(self.resources):
    if np.array_equal(self.agent_pos, resource):
        resources_to_remove.append(i)
        reward = 10
        break

# Remove in reverse to maintain indices
for i in reversed(resources_to_remove):
    del self.resources[i]
```

### 7. Strategy Decision Timing

```python
# Agents reconsider strategy every N steps
if self.decision_timer % self.decision_interval == 0:
    strategy_info = self.choose_strategy(env, other_agents)

self.decision_timer += 1
```

**Typical intervals**:
- `decision_interval = 15`: Reconsider every 15 steps
- Too frequent: Unstable behavior
- Too infrequent: Miss opportunities

### 8. Belief Update Pattern

```python
# Bayesian update pattern
prior = self.beliefs[strategy]
likelihood = self._compute_likelihood(observation, strategy)
evidence = sum(prior[s] * likelihood[s] for s in strategies)

if evidence > 0:
    posterior = (prior * likelihood) / evidence
    self.beliefs[strategy] = posterior
```

### 9. Performance Considerations

**Memory**:
- Store only recent history: `deque(maxlen=50)`
- Clear collections between episodes

**CPU**:
- Use `model.predict(..., deterministic=True)` for faster inference
- Reduce `n_simulations` in Monte Carlo for real-time performance

**GPU** (if available):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 10. Random Seed Management

```python
# For reproducible training
env.reset(seed=42)
np.random.seed(42)
torch.manual_seed(42)
```

---

## 🧪 Testing & Validation

### Unit Testing Approach

**File**: Create `tests/test_*.py` (not currently in repo)

```python
# Example test structure
def test_environment_reset():
    env = AgentDecisionEnv(grid_size=10)
    obs, info = env.reset()
    assert obs.shape == (6,)
    assert len(env.resources) == 5
    assert len(env.obstacles) == 8

def test_monte_carlo_simulation():
    simulator = MonteCarloSimulator(n_simulations=100)
    decision = Decision("Test", cost=1.0,
                       uncertain_parameters={'success_rate': (0.5, 0.8)})
    scenario = {'reward': 10}
    results = simulator.analyze_decision(decision, scenario)
    assert 'mean' in results
    assert results['mean'] > 0
```

### Integration Testing

**Check environment integrity**:
```bash
python scripts/check_setup.py
```

**Verify trained agent**:
```python
# In visualize_agent.py or custom script
model = PPO.load("trained_agent")
env = AgentDecisionEnv()
obs, _ = env.reset()

total_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Total reward: {total_reward}")
assert total_reward > -50, "Agent performing poorly!"
```

### Performance Benchmarks

**Training metrics** (50k timesteps):
- Initial `ep_rew_mean`: -20 to -30
- Target `ep_rew_mean`: -5 to +5
- Good `ep_rew_mean`: +5 to +15

**Agent performance** (20 test episodes):
- Random agent: -30 to -20 average reward
- Trained agent: -5 to +10 average reward
- Well-trained: +10 to +20 average reward

**Resources collected**:
- Random: 0-1 per episode
- Trained: 1-2 per episode
- Well-trained: 2-4 per episode

### Debugging Tips

**RL training not improving**:
1. Check reward structure (are rewards meaningful?)
2. Verify observation space (is state informative?)
3. Increase timesteps (50k may be too few)
4. Adjust hyperparameters (learning rate, batch size)

**Pygame not rendering**:
1. Check `pygame.init()` called
2. Verify display mode set: `pygame.display.set_mode(...)`
3. Ensure `pygame.display.flip()` called each frame
4. Check `clock.tick(FPS)` for frame rate

**Model loading fails**:
1. Verify file exists: `trained_agent.zip`
2. Check path (relative vs absolute)
3. Ensure same Stable-Baselines3 version
4. Use try-except for graceful degradation

**Opponent model not detecting deception**:
1. Ensure sufficient history (>10 observations)
2. Check if behavioral patterns differ enough
3. Tune detection thresholds
4. Verify `update_belief()` called each step

---

## 🎓 Learning Resources

### Understanding the Algorithms

**Reinforcement Learning (PPO)**:
- [OpenAI Spinning Up](https://spinningup.openai.com) - RL fundamentals
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io) - PPO implementation
- Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

**Monte Carlo Methods**:
- MIT OpenCourseWare: "Decision Making Under Uncertainty"
- Book: "Simulation and the Monte Carlo Method" (Rubinstein & Kroese)

**Bayesian Inference**:
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- Paper: Werewolf Game modeling (search for "AI Werewolf" papers)

**Deep Learning (CNN)**:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Book: "Deep Learning" (Goodfellow, Bengio, Courville)

### Code Examples

**Simple RL training loop**:
```python
from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv

env = AgentDecisionEnv(grid_size=10)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("my_agent")
```

**Monte Carlo decision analysis**:
```python
from monte_carlo_decisions import MonteCarloSimulator, Decision

simulator = MonteCarloSimulator(n_simulations=1000)
decision = Decision("Explore", cost=2.0,
                   uncertain_parameters={'success_rate': (0.4, 0.8)})
scenario = {'reward': 15, 'environment': 'uncertain'}
results = simulator.analyze_decision(decision, scenario)
print(f"Expected value: {results['mean']:.2f}")
```

**Opponent modeling**:
```python
from werewolf_opponent_model import WerewolfOpponentModel

model = WerewolfOpponentModel(n_opponents=3)
observation = {'action': 2, 'moved_toward_resource': True}
model.update_belief(opponent_id=0, observation=observation)
analysis = model.get_analysis(opponent_id=0)
print(f"Dominant strategy: {analysis['dominant_strategy']}")
```

---

## 🚀 Future Development Directions

### Planned Features (Not Yet Implemented)

1. **Communication Protocol**
   - Agent-to-agent messaging
   - Coalition formation
   - Information sharing

2. **Advanced AHP (Analytical Hierarchy Process)**
   - Multi-criteria decision framework
   - Weighted objective functions
   - Pareto optimization

3. **Dynamic Environments**
   - Moving obstacles
   - Resource regeneration
   - Environmental disasters

4. **Learning Opponent Models**
   - Train neural opponent model
   - Transfer learning across games
   - Meta-learning strategies

5. **Tournaments & Evaluation**
   - Agent vs agent competitions
   - ELO rating system
   - Strategy diversity metrics

### Experimental Ideas

- **Multi-grid federation**: Multiple connected grids
- **Asymmetric information**: Hidden resources/obstacles
- **Temporal reasoning**: Planning over multiple episodes
- **Evolutionary strategies**: Genetic algorithm for agent design
- **Human-AI cooperation**: Human player + AI teammates

---

## 📌 Quick Reference

### Key File → Purpose Mapping

| File | Primary Purpose | When to Modify |
|------|----------------|----------------|
| `rl_environment.py` | Define RL environment | Change grid, rewards, observations |
| `train_agent.py` | Train RL agents | Adjust training parameters |
| `monte_carlo_decisions.py` | Strategic decisions | Add new strategies |
| `deep_perception.py` | CNN perception | Change perception architecture |
| `werewolf_opponent_model.py` | Opponent modeling | Improve belief updates |
| `integrated_game_v3.py` | Full integration | Add features, test systems |
| `visualize_agent.py` | Watch agents | Debug agent behavior |

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `ModuleNotFoundError: gymnasium` | Missing dependency | `pip install gymnasium` |
| `Model file not found` | Wrong path or not trained | Check file exists, train first |
| `Observation shape mismatch` | Env/model incompatibility | Ensure same obs space |
| `pygame.error: No available video device` | No display | Run on machine with display or use virtual display |
| `CUDA out of memory` | GPU memory full | Use smaller batch or CPU |

### Environment Variables

```bash
# Force CPU (no GPU)
export CUDA_VISIBLE_DEVICES=""

# Set random seed
export PYTHONHASHSEED=42
```

---

## 🤝 Contributing Guidelines

### Before Making Changes

1. **Read this document thoroughly**
2. **Understand the affected components**
3. **Check for breaking changes**
4. **Test existing functionality**

### Code Quality Checklist

- [ ] Follows naming conventions
- [ ] Includes type hints
- [ ] Has docstrings for public functions
- [ ] No hardcoded paths (use relative paths)
- [ ] Handles errors gracefully (try-except)
- [ ] Cleaned up debug print statements
- [ ] Comments explain "why", not "what"
- [ ] Tested manually (if applicable)

### Creating New Versions

**When to version**:
- Major feature addition
- Breaking API changes
- Significant refactoring

**How to version**:
1. Copy latest stable version
2. Increment version number (v3 → v4)
3. Update module docstring
4. Mark previous version as deprecated in comments
5. Update README.md references

---

## 📝 Changelog

### Version History

**v3 (Current - Latest Integration)**
- Full AI system integration
- Deep CNN perception
- Werewolf-inspired opponent modeling
- Bayesian belief updates
- Deception detection
- Multi-agent support (3 agents)

**v2 (Enhanced Integration)**
- Monte Carlo decision making
- Strategic decision intervals
- Improved visualization

**v1 (Basic Integration)**
- Combined RL + Monte Carlo
- Basic multi-agent

**v0 (Foundation)**
- Individual components (RL, MC, CNN, Opponent Model)
- Separate scripts

---

## 🎯 Mission Statement

> This project explores how AI agents make decisions under uncertainty, combining multiple AI paradigms (RL, probabilistic reasoning, deep learning, game theory) in a unified framework. The goal is to create adaptive, strategic agents that can compete, cooperate, and learn in dynamic multi-agent environments.

---

## 📞 Support & Contact

For issues, questions, or contributions:
1. Check this CLAUDE.md first
2. Review README.md for user-level guidance
3. Examine relevant source files
4. Test in isolation before integrating

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Maintained For**: Claude Code and AI Assistant Developers
**Project Status**: Active Development (MVP Complete)

---

*This document is living documentation. Update as the project evolves.*
