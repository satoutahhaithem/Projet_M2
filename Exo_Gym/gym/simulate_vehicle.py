import exo_gym
from exo_gym.simulators import NetworkSimulator

# Define the simulation environment
env = exo_gym.ExoEnvironment()

# Add a local agent (PC in the vehicle)
local_llm = env.add_local_agent(name="PC_Agent", model="llama-3-8b")

# Define network simulator (switch between local and cloud LLM)
network = NetworkSimulator()
network.add_connection("PC_Agent", "Cloud_LLM", latency=50, bandwidth=100)  # Initial 5G connection

# Simulate disconnection (moving vehicle losing connection)
network.simulate_disconnection("PC_Agent", "Cloud_LLM", after_time=10)  # After 10 sec, lose 5G

# Add cloud LLM model
cloud_llm = env.add_cloud_agent(name="Cloud_LLM", model="llama-3-8b")

# Run the simulation
env.run()
