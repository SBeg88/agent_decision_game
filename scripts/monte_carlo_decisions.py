import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

@dataclass
class Decision:
    """Represents a decision with uncertain outcomes"""
    name: str
    cost: float
    uncertain_parameters: dict  # e.g., {'success_rate': (0.6, 0.9)}
    category: str = "general"  # e.g., "exploration", "resource", "cooperation"

class MonteCarloSimulator:
    """Simulate decisions under deep uncertainty"""
    
    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations
        self.history = []  # Track simulation history
    
    def simulate_decision(self, decision: Decision, scenario: dict) -> List[float]:
        """Run Monte Carlo simulation for a decision"""
        outcomes = []
        
        for _ in range(self.n_simulations):
            # Sample uncertain parameters
            sampled_params = {}
            for param, (low, high) in decision.uncertain_parameters.items():
                sampled_params[param] = np.random.uniform(low, high)
            
            # Calculate outcome based on scenario and sampled parameters
            success = np.random.random() < sampled_params.get('success_rate', 0.5)
            
            # Base reward from scenario
            base_reward = scenario.get('reward', 10)
            
            # Apply uncertainty factors
            uncertainty_multiplier = 1.0
            if scenario.get('environment') == 'uncertain':
                uncertainty_multiplier = np.random.uniform(0.7, 1.3)
            
            if scenario.get('competition_level') == 'high':
                uncertainty_multiplier *= np.random.uniform(0.8, 1.0)
            
            # Calculate final outcome
            if success:
                outcome = (base_reward * uncertainty_multiplier) - decision.cost
            else:
                # Partial outcome on failure
                outcome = (base_reward * uncertainty_multiplier * 0.3) - (decision.cost * 0.7)
            
            outcomes.append(outcome)
        
        return outcomes
    
    def analyze_decision(self, decision: Decision, scenario: dict) -> dict:
        """Analyze decision and return comprehensive statistics"""
        outcomes = self.simulate_decision(decision, scenario)
        
        analysis = {
            'mean': np.mean(outcomes),
            'std': np.std(outcomes),
            'percentile_10': np.percentile(outcomes, 10),
            'percentile_25': np.percentile(outcomes, 25),
            'percentile_50': np.percentile(outcomes, 50),
            'percentile_75': np.percentile(outcomes, 75),
            'percentile_90': np.percentile(outcomes, 90),
            'min': np.min(outcomes),
            'max': np.max(outcomes),
            'positive_rate': np.sum(np.array(outcomes) > 0) / len(outcomes),
            'outcomes': outcomes  # Store for plotting
        }
        
        return analysis
    
    def compare_decisions(self, decisions: List[Decision], scenario: dict, verbose=True):
        """Compare multiple decisions under uncertainty"""
        
        if verbose:
            print("\n" + "="*70)
            print("  MONTE CARLO DECISION ANALYSIS")
            print("="*70)
            print(f"\nScenario Configuration:")
            for key, value in scenario.items():
                print(f"  {key}: {value}")
            print(f"\nRunning {self.n_simulations} simulations per decision...\n")
        
        results = {}
        for decision in decisions:
            results[decision.name] = self.analyze_decision(decision, scenario)
        
        # Store in history
        self.history.append({
            'scenario': scenario,
            'decisions': decisions,
            'results': results
        })
        
        # Print results
        if verbose:
            print("="*70)
            print("  RESULTS")
            print("="*70)
            
            for name, stats in results.items():
                print(f"\n{name}:")
                print(f"  Expected Value:       {stats['mean']:>8.2f}")
                print(f"  Risk (Std Dev):       {stats['std']:>8.2f}")
                print(f"  Worst Case (10th %):  {stats['percentile_10']:>8.2f}")
                print(f"  Median (50th %):      {stats['percentile_50']:>8.2f}")
                print(f"  Best Case (90th %):   {stats['percentile_90']:>8.2f}")
                print(f"  Success Rate:         {stats['positive_rate']*100:>7.1f}%")
            
            # Recommend best decision
            print("\n" + "="*70)
            print("  RECOMMENDATION")
            print("="*70)
            
            best_expected = max(results.items(), key=lambda x: x[1]['mean'])
            best_median = max(results.items(), key=lambda x: x[1]['percentile_50'])
            safest = max(results.items(), key=lambda x: x[1]['percentile_10'])
            
            print(f"\nHighest Expected Value: {best_expected[0]} ({best_expected[1]['mean']:.2f})")
            print(f"Highest Median Outcome: {best_median[0]} ({best_median[1]['percentile_50']:.2f})")
            print(f"Safest (Best Worst-Case): {safest[0]} ({safest[1]['percentile_10']:.2f})")
            
            print("\n" + "="*70)
        
        return results
    
    def visualize_comparison(self, decisions: List[Decision], scenario: dict, save_path=None):
        """Create visualization comparing decision outcomes"""
        results = self.compare_decisions(decisions, scenario, verbose=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot comparison
        data_for_box = [results[d.name]['outcomes'] for d in decisions]
        decision_names = [d.name for d in decisions]
        
        ax1.boxplot(data_for_box, labels=decision_names)
        ax1.set_title('Outcome Distribution Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Outcome Value')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax1.legend()
        
        # Bar chart of key metrics
        metrics = ['mean', 'percentile_10', 'percentile_90']
        metric_labels = ['Expected Value', 'Worst Case (10%)', 'Best Case (90%)']
        
        x = np.arange(len(decisions))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[d.name][metric] for d in decisions]
            ax2.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax2.set_xlabel('Decisions')
        ax2.set_ylabel('Value')
        ax2.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(decision_names, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        
        plt.show()
        
        return fig

# Example usage and scenarios
def run_example_scenarios():
    """Run example decision scenarios"""
    
    simulator = MonteCarloSimulator(n_simulations=10000)
    
    # Scenario 1: High-risk exploration
    print("\n\n" + "╔"+"═"*68+"╗")
    print("║" + " "*18 + "SCENARIO 1: High-Risk Exploration" + " "*17 + "║")
    print("╚"+"═"*68+"╝")
    
    decisions_1 = [
        Decision(
            name="Aggressive Exploration",
            cost=5,
            uncertain_parameters={'success_rate': (0.3, 0.7)},
            category="exploration"
        ),
        Decision(
            name="Conservative Gathering",
            cost=2,
            uncertain_parameters={'success_rate': (0.6, 0.9)},
            category="resource"
        ),
        Decision(
            name="Cooperative Strategy",
            cost=3,
            uncertain_parameters={'success_rate': (0.5, 0.8)},
            category="cooperation"
        )
    ]
    
    scenario_1 = {
        'reward': 15,
        'environment': 'uncertain',
        'competition_level': 'high'
    }
    
    results_1 = simulator.compare_decisions(decisions_1, scenario_1)
    
    # Scenario 2: Stable environment
    print("\n\n" + "╔"+"═"*68+"╗")
    print("║" + " "*16 + "SCENARIO 2: Stable Environment" + " "*21 + "║")
    print("╚"+"═"*68+"╝")
    
    decisions_2 = [
        Decision(
            name="Rapid Expansion",
            cost=8,
            uncertain_parameters={'success_rate': (0.5, 0.85)},
            category="exploration"
        ),
        Decision(
            name="Steady Growth",
            cost=4,
            uncertain_parameters={'success_rate': (0.7, 0.95)},
            category="resource"
        ),
        Decision(
            name="Consolidation",
            cost=2,
            uncertain_parameters={'success_rate': (0.8, 0.98)},
            category="defense"
        )
    ]
    
    scenario_2 = {
        'reward': 20,
        'environment': 'stable',
        'competition_level': 'low'
    }
    
    results_2 = simulator.compare_decisions(decisions_2, scenario_2)
    
    # Optional: Visualize if matplotlib is available
    try:
        print("\nGenerating visualization...")
        simulator.visualize_comparison(decisions_1, scenario_1, save_path='decision_analysis.png')
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    run_example_scenarios()
