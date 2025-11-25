import os
import sys

print("=== Project Setup Check ===\n")
print(f"Current directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

required_files = ['rl_environment.py', 'train_agent.py', 'game_world.py']
print(f"\nChecking for required files:")
for file in required_files:
    exists = os.path.exists(file)
    print(f"  {file}: {'✓ Found' if exists else '✗ Missing'}")

print("\nChecking installed packages:")
packages = ['pygame', 'numpy', 'gymnasium', 'stable_baselines3', 'torch']
for package in packages:
    try:
        __import__(package)
        print(f"  {package}: ✓ Installed")
    except ImportError:
        print(f"  {package}: ✗ Not installed")

print("\nChecking if rl_environment module can be imported:")
try:
    from rl_environment import AgentDecisionEnv
    print("  ✓ AgentDecisionEnv successfully imported!")
except Exception as e:
    print(f"  ✗ Error: {e}")