import pygame
import numpy as np
from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv
from monte_carlo_decisions import MonteCarloSimulator, Decision

# [Copy the visualization code from visualize_agent.py]
# Add strategic decision-making every 20 steps

class StrategicAgent:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.decision_simulator = MonteCarloSimulator(n_simulations=500)
        self.strategy = "Aggressive"
        self.decision_timer = 0
        
    def choose_strategy(self, env):
        """Use Monte Carlo to choose strategy every 20 steps"""
        decisions = [
            Decision("Aggressive", cost=1, 
                    uncertain_parameters={'success_rate': (0.4, 0.8)}),
            Decision("Conservative", cost=0.5, 
                    uncertain_parameters={'success_rate': (0.6, 0.9)}),
        ]
        
        scenario = {
            'reward': len(env.resources) * 3,
            'risk_level': len(env.obstacles) / 10
        }
        
        results = {}
        for decision in decisions:
            results[decision.name] = self.decision_simulator.analyze_decision(
                decision, scenario)
        
        # Choose best strategy
        best = max(results.items(), key=lambda x: x[1]['mean'])
        self.strategy = best[0]
        print(f"Strategy selected: {self.strategy} (Expected value: {best[1]['mean']:.2f})")
    
    def get_action(self, obs, env):
        """Get action from model, potentially modified by strategy"""
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Reconsider strategy every 20 steps
        self.decision_timer += 1
        if self.decision_timer >= 20:
            self.choose_strategy(env)
            self.decision_timer = 0
        
        return action

# Use this in your visualization