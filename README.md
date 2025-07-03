# Autonomous and Adaptive Systems - Overcooked

This project was developed for the Autonomous and Adaptive Systems course at the University of Bologna. It explores training cooperative agents for the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) environment using Multi-Agent Proximal Policy Optimization (MAPPO). The primary focus is on evaluating the agents' ability to generalize across different game layouts and adapt to partners with varying behaviors.


# Setup

after cloning the repository, install the requirements:

```bash
pip install -r requirements.txt
```

## Training Agents
To train an agent on one or more layouts, run the following command:

```bash
python src/train.py
``` 

### Command-line Flags
- `--layouts`: A list of layouts to train on (default: `["cramped_room"]`).
- `--episodes`: The number of episodes to train the agent (default: `1000`).
- `--random_prob`: The probability of using a random agent for each episode (default: `0.0`).
- `--model_name`: The name of the model to save (if not specified a default name will be used based on the layouts).

## Testing Agents
To test the trained agents, you can run the following command:

```bash
python src/test.py
```

### Command-line Flags
- `--agents`: A list of agents to test.
- `--layouts`: A list of layouts to test on.
- `render`: If set, an episode will be rendered in a GUI window after testing is completed.
- `verbose`: If set, the results of each episode will be printed to the console.

## Demo
The `train_agents.ipynb` notebook contains the code used to train and test all agents featured in the experiments. For a demonstration of some of these agents in action, refer to `demo.ipynb`.