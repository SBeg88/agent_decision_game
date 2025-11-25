# Multi-Agent Decision Game - Getting Started Guide

## 📁 Project Files Overview

Your project now has the following files:

### Core Files (Already Created)
- `game_world.py` - Basic grid world with manual control
- `rl_environment.py` - RL environment following Gymnasium API
- `train_agent.py` - Initial training script (50k timesteps)
- `check_setup.py` - Verify your installation

### New Files (Just Created)
- `visualize_agent.py` - Watch your trained AI agent play
- `compare_agents.py` - Test and compare agent performance
- `train_longer.py` - Extended training (100k more timesteps)
- `monte_carlo_decisions.py` - Decision-making under uncertainty
- `integrated_game_v2.py` - **Full integration: RL + Monte Carlo**
- `README.md` - This file!

## 🚀 Quick Start - What to Run Now

### Step 1: Watch Your Trained Agent (Most Fun!)
```bash
python visualize_agent.py
```
**Controls:**
- SPACE: Pause/Resume
- R: Reset episode
- Q: Quit

This shows your AI navigating the grid, collecting resources (green), avoiding obstacles (red).

### Step 2: Analyze Performance
```bash
python compare_agents.py
```
This runs 20 test episodes and shows:
- Average reward
- Resources collected
- Success rate
- Comparison with random baseline

### Step 3: Run the Integrated Game (RL + Monte Carlo)
```bash
python integrated_game_v2.py
```
Watch the agent make strategic decisions every 20 steps:
- **Red ring**: Aggressive strategy
- **Yellow ring**: Balanced strategy
- **Green ring**: Conservative strategy

The agent uses Monte Carlo simulation to choose strategies based on:
- Remaining resources
- Number of obstacles
- Current performance

### Step 4: Test Monte Carlo Decisions Standalone
```bash
python monte_carlo_decisions.py
```
See how Monte Carlo simulation compares different strategies under uncertainty.

### Step 5: Train Longer for Better Performance (Optional)
```bash
python train_longer.py
```
This continues training for 100,000 more timesteps (takes ~5-10 minutes).

Then modify `visualize_agent.py` or `integrated_game.py`:
```python
# Change this line:
model = PPO.load("trained_agent")

# To this:
model = PPO.load("trained_agent_v2")
```

## 📊 Understanding the Output

### Training Metrics
- `ep_rew_mean`: Average reward per episode (higher = better)
- `ep_len_mean`: How many steps per episode
- `value_loss`: Agent's prediction accuracy (should decrease)

### Your Results
- Started at: **-26.7** reward
- Ended at: **-7.18** reward
- Best during training: **+0.29** reward
- **This is good progress!** The agent learned significantly.

### What the Rewards Mean
- `+10`: Collected a resource
- `-0.1`: Each step (encourages efficiency)
- `-5`: Hit an obstacle

## 🎯 Next Steps - Phase 4 Development

### Add Multi-Agent Competition
Create multiple agents competing for resources:
```python
# In integrated_game.py, add:
self.agent2 = StrategicAgent("trained_agent", agent_id=2, color=ORANGE)
```

### Implement Analytical Hierarchy Process (AHP)
For multi-criteria decision making:
- Resource value
- Risk level
- Time efficiency
- Cooperation benefit

### Create Different Personas
Agents with different strategies:
- **Risk-taker**: Prefers aggressive exploration
- **Conservative**: Focuses on safe resource gathering
- **Cooperative**: Values team outcomes

### Add Deep Uncertainty Scenarios
- Resource scarcity
- Dynamic obstacles
- Changing reward structures
- Environmental disasters

## 🔧 Customization Options

### Change Grid Size
In `rl_environment.py`:
```python
env = AgentDecisionEnv(grid_size=15)  # Change from 10 to 15
```

### Adjust Training Time
In `train_agent.py` or `train_longer.py`:
```python
model.learn(total_timesteps=200000)  # Train longer
```

### Modify Decision Interval
In `integrated_game_v2.py`:
```python
self.decision_interval = 10  # Make decisions more frequently
```

### Change Strategy Parameters
In `integrated_game_v2.py` -> `choose_strategy()`:
```python
Decision(
    name="Very Aggressive",
    cost=2.0,  # Higher cost
    uncertain_parameters={'success_rate': (0.2, 0.7)},  # Lower success rate
)
```

## 📈 Performance Benchmarks

### What Good Performance Looks Like
- **Beginner Agent** (50k steps): -7 to +2 average reward
- **Intermediate Agent** (150k steps): +2 to +8 average reward
- **Advanced Agent** (500k+ steps): +10 to +20 average reward

### Resources Collected
- **Random Agent**: ~0.5 resources per episode
- **Your Trained Agent**: ~1-2 resources per episode
- **Well-Trained Agent**: 3-4 resources per episode

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install gymnasium stable-baselines3 pygame numpy torch matplotlib
```

### Agent Seems Stuck
- Train longer with `train_longer.py`
- Adjust reward structure in `rl_environment.py`
- Increase grid size for more exploration space

### Visualisation Too Fast/Slow
In `visualize_agent.py` or `integrated_game.py`:
```python
FPS = 10  # Change from 5 to 10 (faster) or 3 (slower)
```

### Training Takes Too Long
```bash
# Use fewer timesteps for testing
model.learn(total_timesteps=10000)  # Faster but less trained
```

## 📚 Learning Resources

### Reinforcement Learning
- OpenAI Spinning Up: https://spinningup.openai.com
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io

### Monte Carlo Methods
- "Decision Making Under Uncertainty" - MIT OpenCourseWare
- Monte Carlo simulation tutorials on YouTube

### Game Development
- Pygame documentation: https://www.pygame.org/docs
- Real Python Pygame tutorial

## 🎓 Project Evolution Path

### Current Stage: MVP Complete ✓
You have:
- ✅ Grid world environment
- ✅ Trained RL agent
- ✅ Monte Carlo decision system
- ✅ Basic integration

### Next Milestones:
1. **Multi-agent system** (2+ competing agents)
2. **Advanced decision trees** (MCTS implementation)
3. **AHP for multi-criteria analysis**
4. **Persona-based agents** (different strategies/preferences)
5. **Deep uncertainty scenarios** (dynamic environments)

## 💡 Pro Tips

1. **Start Simple**: Run `visualize_agent.py` first to understand agent behavior
2. **Test Often**: Use `compare_agents.py` after any changes
3. **Save Models**: Always save before experimenting with new parameters
4. **Track Progress**: Keep notes on what parameter changes improve performance
5. **Visualise Decisions**: The integrated game shows strategic thinking in real-time

## 🤝 Next Session Planning

### If You Want to Improve the AI:
1. Run `train_longer.py`
2. Experiment with reward structure
3. Add more complex decision scenarios

### If You Want to Add Features:
1. Multiple agents
2. Different game modes
3. Save/load game states
4. Statistics dashboard

### If You Want to Understand Better:
1. Read the code comments
2. Modify parameters and observe changes
3. Add print statements to see decision-making
4. Create simpler test scenarios

## 📞 Getting Help

If you encounter issues:
1. Check this README for common solutions
2. Review error messages carefully
3. Test individual components separately
4. Simplify parameters and rebuild complexity

---

**Congratulations on building your Multi-Agent Decision Game!** 🎉

You've successfully integrated:
- Reinforcement Learning
- Monte Carlo Simulation
- Decision-making under uncertainty
- Real-time visualisation

This is a solid foundation for exploring AI agents, uncertainty, and strategic decision-making.

**Have fun experimenting!** 🚀
