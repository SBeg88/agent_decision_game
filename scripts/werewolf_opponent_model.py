import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

@dataclass
class InformationNode:
    """Represents a possible opponent state configuration"""
    agent_type: str  # 'aggressive', 'conservative', 'deceptive'
    resources: int
    position_belief: np.ndarray
    checked_actions: List[int]  # Actions we've observed
    probability: float

class BayesianOpponentModel:
    """
    Opponent modelling using Werewolf Game's Bayesian approach
    Adapted from the paper's information set framework
    """
    
    def __init__(self, n_agents: int, grid_size: int):
        self.n_agents = n_agents
        self.grid_size = grid_size
        
        # Information sets for each opponent (similar to paper's It)
        self.information_sets = {
            i: [] for i in range(n_agents)
        }
        
        # Strategy mapping inspired by paper's g: It → A
        self.strategy_beliefs = {
            i: {'aggressive': 0.33, 'conservative': 0.33, 'deceptive': 0.34}
            for i in range(n_agents)
        }
        
        # Revelation timing (from paper's prophet strategy)
        self.revelation_round = {i: None for i in range(n_agents)}
        self.hidden_information = {i: [] for i in range(n_agents)}
        
    def update_belief(self, agent_id: int, observation: Dict):
        """
        Update belief using Bayesian inference (similar to paper's recursive formula)
        Following the paper's R(Hiding, It) calculation approach
        """
        current_info_set = self.information_sets[agent_id]
        
        # Prior beliefs
        prior_aggressive = self.strategy_beliefs[agent_id]['aggressive']
        prior_conservative = self.strategy_beliefs[agent_id]['conservative']
        prior_deceptive = self.strategy_beliefs[agent_id]['deceptive']
        
        # Likelihood of observation given each strategy type
        likelihood_aggressive = self._compute_likelihood(
            observation, 'aggressive'
        )
        likelihood_conservative = self._compute_likelihood(
            observation, 'conservative'
        )
        likelihood_deceptive = self._compute_likelihood(
            observation, 'deceptive'
        )
        
        # Posterior calculation (Bayes' theorem)
        evidence = (prior_aggressive * likelihood_aggressive + 
                   prior_conservative * likelihood_conservative +
                   prior_deceptive * likelihood_deceptive)
        
        if evidence > 0:
            post_aggressive = (prior_aggressive * likelihood_aggressive) / evidence
            post_conservative = (prior_conservative * likelihood_conservative) / evidence
            post_deceptive = (prior_deceptive * likelihood_deceptive) / evidence
            
            # Update beliefs
            self.strategy_beliefs[agent_id] = {
                'aggressive': post_aggressive,
                'conservative': post_conservative,
                'deceptive': post_deceptive
            }
    
    def _compute_likelihood(self, observation: Dict, strategy: str) -> float:
        """
        Compute P(observation | strategy)
        Based on paper's action patterns for different roles
        """
        action = observation.get('action', 4)  # Default to STAY
        moved_toward_resource = observation.get('moved_toward_resource', False)
        avoided_conflict = observation.get('avoided_conflict', False)
        
        if strategy == 'aggressive':
            # Aggressive agents move frequently toward resources
            if moved_toward_resource and action != 4:
                return 0.8
            elif action == 4:  # Staying is unlikely
                return 0.1
            return 0.3
            
        elif strategy == 'conservative':
            # Conservative agents avoid conflicts, move cautiously
            if avoided_conflict:
                return 0.7
            elif action == 4:  # Often stay
                return 0.5
            return 0.2
            
        else:  # deceptive
            # Deceptive agents have mixed patterns (like werewolves)
            # They might appear conservative but strike strategically
            if observation.get('pattern_break', False):
                return 0.6
            return 0.4
    
    def should_reveal_information(self, agent_id: int, round_num: int) -> bool:
        """
        Determine if we should reveal what we know about this opponent
        Based on paper's optimal revelation timing f(h, m)
        """
        # From Table 3 in paper: optimal revelation rounds
        belief_strength = max(self.strategy_beliefs[agent_id].values())
        opponents_remaining = sum(1 for i in range(self.n_agents) if i != agent_id)
        
        # Adapt paper's revelation strategy
        if opponents_remaining >= 4 and belief_strength > 0.7:
            optimal_round = 3  # R3 from paper
        elif opponents_remaining >= 2 and belief_strength > 0.6:
            optimal_round = 2  # R2 from paper
        else:
            optimal_round = 4  # R4 or later
        
        return round_num >= optimal_round


class WerewolfInspiredOpponentNet(nn.Module):
    """
    Neural network that implements Werewolf Game strategies for opponent modelling
    Incorporates the paper's strategic decision-making
    """
    
    def __init__(self, state_dim=6, hidden_dim=128, n_strategies=3):
        super().__init__()
        
        # Input processing (like paper's information set)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Bayesian belief representation
        self.belief_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Strategy classifier (aggressive/conservative/deceptive)
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_strategies)
        )
        
        # Action prediction given strategy
        self.action_heads = nn.ModuleDict({
            'aggressive': nn.Linear(hidden_dim, 5),
            'conservative': nn.Linear(hidden_dim, 5),
            'deceptive': nn.Linear(hidden_dim, 5)
        })
        
        # Value estimation (like paper's winning probability)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state_sequence, hidden=None):
        """
        Process sequence and predict opponent behaviour
        Returns strategy distribution and action probabilities
        """
        # Encode states
        batch_size, seq_len, _ = state_sequence.shape
        encoded = self.state_encoder(state_sequence.view(-1, state_sequence.size(-1)))
        encoded = encoded.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        lstm_out, hidden = self.belief_lstm(encoded, hidden)
        
        # Get final hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Predict strategy distribution
        strategy_logits = self.strategy_head(final_hidden)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        
        # Predict actions for each strategy
        action_probs = {}
        for strategy in ['aggressive', 'conservative', 'deceptive']:
            logits = self.action_heads[strategy](final_hidden)
            action_probs[strategy] = F.softmax(logits, dim=-1)
        
        # Weighted action prediction based on strategy beliefs
        final_action_probs = torch.zeros(batch_size, 5)
        strategies = ['aggressive', 'conservative', 'deceptive']
        for i, strategy in enumerate(strategies):
            weight = strategy_probs[:, i:i+1]
            final_action_probs += weight * action_probs[strategy]
        
        # Value estimation
        value = torch.sigmoid(self.value_head(final_hidden))
        
        return {
            'strategy_probs': strategy_probs,
            'action_probs': final_action_probs,
            'value': value,
            'hidden': hidden
        }


class StrategicDeceptionDetector:
    """
    Detect deceptive behaviour patterns inspired by Werewolf Game
    Uses the paper's approach to identifying werewolves (deceptive agents)
    """
    
    def __init__(self, memory_length=20):
        self.memory_length = memory_length
        self.action_history = defaultdict(list)
        self.voting_patterns = defaultdict(list)
        
    def observe_action(self, agent_id: int, action: int, context: Dict):
        """Record action with context"""
        self.action_history[agent_id].append({
            'action': action,
            'context': context,
            'timestamp': len(self.action_history[agent_id])
        })
        
        # Keep only recent history
        if len(self.action_history[agent_id]) > self.memory_length:
            self.action_history[agent_id].pop(0)
    
    def detect_deception(self, agent_id: int) -> Dict:
        """
        Detect deceptive patterns using Werewolf Game insights
        Returns deception indicators and confidence
        """
        history = self.action_history[agent_id]
        
        if len(history) < 5:
            return {'is_deceptive': False, 'confidence': 0.0}
        
        # Pattern 1: Avoiding elimination of allies (like werewolves)
        avoidance_pattern = self._detect_avoidance_pattern(history)
        
        # Pattern 2: Sudden strategy shifts (like werewolves using "all-in")
        strategy_shifts = self._detect_strategy_shifts(history)
        
        # Pattern 3: Coordinated actions with hidden allies
        coordination = self._detect_coordination(agent_id, history)
        
        # Combine indicators (weighted like paper's probability calculations)
        deception_score = (
            0.4 * avoidance_pattern +
            0.3 * strategy_shifts +
            0.3 * coordination
        )
        
        return {
            'is_deceptive': deception_score > 0.5,
            'confidence': deception_score,
            'patterns': {
                'avoidance': avoidance_pattern,
                'shifts': strategy_shifts,
                'coordination': coordination
            }
        }
    
    def _detect_avoidance_pattern(self, history: List) -> float:
        """Detect if agent avoids targeting specific others"""
        if len(history) < 10:
            return 0.0
        
        # Analyse targeting patterns
        targets = [h['context'].get('target') for h in history 
                  if h['context'].get('target') is not None]
        
        if not targets:
            return 0.0
        
        # Check for suspicious consistency (never targeting certain agents)
        unique_targets = set(targets)
        expected_targets = len(set(range(self.memory_length))) * 0.7
        
        if len(unique_targets) < expected_targets:
            return 0.8
        return 0.2
    
    def _detect_strategy_shifts(self, history: List) -> float:
        """Detect sudden strategy changes (like paper's "all-in" strategy)"""
        if len(history) < 10:
            return 0.0
        
        # Split history into halves
        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]
        
        # Calculate action distributions
        first_actions = [h['action'] for h in first_half]
        second_actions = [h['action'] for h in second_half]
        
        # Compute KL divergence between distributions
        kl_div = self._kl_divergence(first_actions, second_actions)
        
        # High divergence indicates strategy shift
        return min(1.0, kl_div / 2.0)
    
    def _detect_coordination(self, agent_id: int, history: List) -> float:
        """Detect coordinated behaviour with other agents"""
        # This would need access to other agents' histories
        # Simplified version here
        coordination_indicators = 0
        
        for h in history:
            if h['context'].get('simultaneous_action'):
                coordination_indicators += 1
        
        return min(1.0, coordination_indicators / max(1, len(history)))
    
    def _kl_divergence(self, dist1: List, dist2: List) -> float:
        """Calculate KL divergence between two action distributions"""
        # Create probability distributions
        p = np.bincount(dist1, minlength=5) + 1e-10
        p = p / p.sum()
        
        q = np.bincount(dist2, minlength=5) + 1e-10
        q = q / q.sum()
        
        # KL divergence
        kl = np.sum(p * np.log(p / q))
        return float(kl)


class IntegratedWerewolfOpponentSystem:
    """
    Complete opponent modelling system using Werewolf Game strategies
    Combines Bayesian updates, neural predictions, and deception detection
    """
    
    def __init__(self, n_agents: int, grid_size: int):
        self.n_agents = n_agents
        self.grid_size = grid_size
        
        # Components
        self.bayesian_model = BayesianOpponentModel(n_agents, grid_size)
        self.neural_model = WerewolfInspiredOpponentNet()
        self.deception_detector = StrategicDeceptionDetector()
        
        # Track game phase (like paper's night/day phases)
        self.phase = 'exploration'  # 'exploration', 'competition', 'endgame'
        self.round_num = 0
        
    def update(self, observations: Dict):
        """Update all models with new observations"""
        self.round_num += 1
        
        for agent_id, obs in observations.items():
            # Bayesian belief update
            self.bayesian_model.update_belief(agent_id, obs)
            
            # Deception detection
            self.deception_detector.observe_action(
                agent_id, obs['action'], obs
            )
            
        # Update game phase
        self._update_phase()
    
    def get_opponent_analysis(self, agent_id: int) -> Dict:
        """
        Complete analysis of an opponent using all systems
        Returns strategic recommendations
        """
        # Get Bayesian beliefs
        beliefs = self.bayesian_model.strategy_beliefs[agent_id]
        
        # Check for deception
        deception = self.deception_detector.detect_deception(agent_id)
        
        # Determine if we should reveal information
        should_reveal = self.bayesian_model.should_reveal_information(
            agent_id, self.round_num
        )
        
        # Strategic recommendation
        strategy = self._determine_counter_strategy(beliefs, deception)
        
        return {
            'beliefs': beliefs,
            'deception': deception,
            'should_reveal': should_reveal,
            'recommended_strategy': strategy,
            'phase': self.phase
        }
    
    def _update_phase(self):
        """Update game phase based on round and agent states"""
        if self.round_num < 10:
            self.phase = 'exploration'
        elif self.round_num < 30:
            self.phase = 'competition'
        else:
            self.phase = 'endgame'
    
    def _determine_counter_strategy(self, beliefs: Dict, deception: Dict) -> str:
        """
        Determine optimal counter-strategy using paper's approach
        Implements ideas from "random strategy+" 
        """
        if deception['is_deceptive'] and deception['confidence'] > 0.7:
            # Against deceptive opponents, use paper's revealing strategy
            return 'reveal_and_coordinate'
        
        # Find dominant belief
        dominant = max(beliefs.items(), key=lambda x: x[1])
        strategy_type, confidence = dominant
        
        if self.phase == 'endgame' and confidence > 0.6:
            # Use paper's "all-in" strategy concept in endgame
            if strategy_type == 'aggressive':
                return 'defensive_all_in'
            elif strategy_type == 'conservative':
                return 'aggressive_all_in'
            else:
                return 'adaptive'
        
        # Standard counter-strategies
        counter_map = {
            'aggressive': 'defensive_counter',
            'conservative': 'exploitative',
            'deceptive': 'cautious_probe'
        }
        
        return counter_map.get(strategy_type, 'balanced')


def demonstrate_werewolf_opponent_modelling():
    """Demonstrate the Werewolf-inspired opponent modelling system"""
    print("\n" + "="*70)
    print("  WEREWOLF-INSPIRED OPPONENT MODELLING")
    print("="*70)
    
    system = IntegratedWerewolfOpponentSystem(n_agents=4, grid_size=10)
    
    print("\nSimulating 40 rounds with 4 agents...")
    print("Agent 0: You (citizen/villager)")
    print("Agent 1: Aggressive (citizen)")  
    print("Agent 2: Conservative (citizen)")
    print("Agent 3: Deceptive (werewolf-like)\n")
    
    # Simulate rounds
    for round_num in range(40):
        # Generate observations based on agent types
        observations = {
            1: {  # Aggressive
                'action': np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15]),
                'moved_toward_resource': np.random.random() > 0.3,
                'avoided_conflict': False,
                'target': np.random.choice([0, 2, 3])
            },
            2: {  # Conservative
                'action': np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.1, 0.1, 0.6]),
                'moved_toward_resource': np.random.random() > 0.7,
                'avoided_conflict': True,
                'target': None
            },
            3: {  # Deceptive (werewolf-like)
                'action': np.random.choice([0, 1, 2, 3, 4]),
                'moved_toward_resource': round_num < 20,  # Changes behaviour
                'avoided_conflict': round_num < 15,
                'pattern_break': round_num == 20,
                'target': np.random.choice([0, 1, 2]) if round_num > 20 else None,
                'simultaneous_action': round_num > 25
            }
        }
        
        # Update system
        system.update(observations)
        
        # Analyse at key rounds (like paper's revelation rounds)
        if round_num in [9, 19, 29, 39]:
            print(f"\n{'='*50}")
            print(f"ROUND {round_num + 1} ANALYSIS (Phase: {system.phase})")
            print('='*50)
            
            for agent_id in [1, 2, 3]:
                analysis = system.get_opponent_analysis(agent_id)
                
                print(f"\nAgent {agent_id}:")
                print(f"  Beliefs: ", end="")
                for strategy, prob in analysis['beliefs'].items():
                    print(f"{strategy}: {prob:.1%} ", end="")
                print()
                
                if analysis['deception']['is_deceptive']:
                    print(f"  ⚠️ DECEPTION DETECTED! "
                          f"(confidence: {analysis['deception']['confidence']:.1%})")
                    print(f"     Patterns: {analysis['deception']['patterns']}")
                
                if analysis['should_reveal']:
                    print(f"  📢 Should reveal information about this agent!")
                
                print(f"  Recommended strategy: {analysis['recommended_strategy']}")
    
    print("\n" + "="*70)
    print("  FINAL ASSESSMENT")  
    print("="*70)
    
    for agent_id in [1, 2, 3]:
        analysis = system.get_opponent_analysis(agent_id)
        dominant = max(analysis['beliefs'].items(), key=lambda x: x[1])
        
        true_type = ['Aggressive', 'Conservative', 'Deceptive'][agent_id - 1]
        predicted_type = dominant[0].capitalize()
        
        print(f"Agent {agent_id}: True={true_type}, "
              f"Predicted={predicted_type} ({dominant[1]:.1%})")
        
        if analysis['deception']['is_deceptive']:
            print(f"         Correctly identified as deceptive!")


if __name__ == "__main__":
    demonstrate_werewolf_opponent_modelling()
