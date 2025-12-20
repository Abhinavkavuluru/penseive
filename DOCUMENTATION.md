# Pensieve Project Documentation

## Overview

Pensieve is a reinforcement learning-based adaptive bitrate (ABR) video streaming system. It uses Asynchronous Advantage Actor-Critic (A3C) neural networks to learn optimal bitrate selection policies for video streaming, aiming to maximize Quality of Experience (QoE) by balancing video quality, rebuffering, and smoothness.

The project is based on the research paper: **"Neural Adaptive Video Streaming with Pensieve"** by Hongzi Mao et al., SIGCOMM 2017.

---

## Project Structure

```
pensieve/
├── cooked_traces/           # Network bandwidth traces for training
├── sim/                     # Basic simulation training (single video)
├── multi_video_sim/         # Advanced multi-video simulation training
├── test/                    # Testing and baseline algorithms
├── run_exp/                 # Real-world experiment runners
├── rl_server/               # RL server for live streaming
├── video_server/            # Video content server
├── dash.all.min.js          # DASH.js player modified for ABR
├── setup.py                 # Package installation
└── README.md                # Original readme
```

---

## Three Main Models/Algorithms

### 1. Basic Simulation Model (`sim/`)

**Purpose:** Training an A3C-based ABR agent on simulated network traces with a fixed video configuration.

**Key Files:**
- `sim/multi_agent.py` - Main training script using parallel A3C agents
- `sim/a3c.py` - Actor-Critic neural network implementation
- `sim/env.py` - Simulation environment for video streaming
- `sim/load_trace.py` - Network trace loading utility

**Configuration:**
- 6 bitrate levels: [300, 750, 1200, 1850, 2850, 4300] Kbps
- 48 video chunks
- 8 parallel agents

**How to Run:**
```bash
cd sim
python multi_agent.py
```

**Output:**
- Model checkpoints in `sim/results/`
- Training logs in `sim/results/log_*`
- TensorBoard events for visualization

---

### 2. Multi-Video Simulation Model (`multi_video_sim/`)

**Purpose:** Advanced training supporting multiple videos with variable bitrate options. Uses dynamic bitrate masking to handle different video configurations.

**Key Files:**
- `multi_video_sim/multi_agent.py` - Multi-video training script
- `multi_video_sim/a3c.py` - Enhanced A3C with bitrate masking
- `multi_video_sim/env.py` - Multi-video environment
- `multi_video_sim/rl_test.py` - Testing script

**Configuration:**
- 10 maximum bitrate levels: [200, 300, 450, 750, 1200, 1850, 2850, 4300, 6000, 8000] Kbps
- Variable video chunks per video
- Bitrate masking for per-video available quality levels
- 16 parallel agents

**Video File Format:**
```
Line 1: num_bitrates num_chunks
Line 2: mask (10 values, 0 or 1 indicating available bitrates)
Line 3+: chunk_sizes for each available bitrate (one line per chunk)
```

**How to Run:**
```bash
cd multi_video_sim
python multi_agent.py
```

**Output:**
- Model checkpoints in `multi_video_sim/models/`
- Training logs in `multi_video_sim/results/`
- Test results in `multi_video_sim/test_results/`

---

### 3. Baseline Algorithms (`test/`)

**Purpose:** Testing and comparing against baseline ABR algorithms.

#### a) Buffer-Based (BB) Algorithm (`test/bb.py`)
- Uses buffer occupancy to select bitrate
- If buffer < 5s: lowest quality
- If buffer > 15s: highest quality
- Linear interpolation in between

#### b) Model Predictive Control (MPC) (`test/mpc.py`)
- Uses future bandwidth prediction
- Optimizes bitrate selection over a planning horizon
- Considers video chunk sizes and rebuffering penalties

#### c) Trained RL Model Testing (`test/rl_no_training.py`)
- Evaluates trained neural network model
- No gradient updates during testing

**How to Run:**
```bash
cd test
python bb.py           # Buffer-based baseline
python mpc.py          # MPC baseline
python rl_no_training.py ./models/nn_model.ckpt  # Trained model
```

---

## Neural Network Architecture

### Actor Network
The actor network predicts action probabilities (bitrate selection).

**Input State (S_INFO x S_LEN):**
1. Last selected bitrate (normalized)
2. Current buffer size (normalized)
3. Throughput measurements (bandwidth history)
4. Download time measurements
5. Remaining video chunks ratio
6. Next chunk sizes for each bitrate
7. Bitrate mask (for multi-video)

**Architecture:**
```
Input -> Split Features
├── Dense(64, relu) for last bitrate
├── Dense(64, relu) for buffer size
├── Dense(64, relu) for remaining chunks
├── Conv2D(128, 3x3, relu) for throughput/time history
├── Conv1D(128, 4, relu) for next chunk sizes
└── Conv1D(128, 4, relu) for bitrate mask
Merge -> Dense(128, relu) -> Dense(num_actions) -> Softmax
```

### Critic Network
The critic network estimates state value V(s).

**Architecture:** Similar to actor but outputs single value:
```
Merge -> Dense(100, relu) -> Dense(1)
```

---

## Reward Function

The reward function balances three QoE factors:

```
Reward = bitrate/1000 - 4.3 * rebuffer_time - 1.0 * |bitrate_change|/1000
```

- **Bitrate Term:** Higher bitrate = higher reward
- **Rebuffer Penalty:** 4.3x penalty per second of rebuffering
- **Smoothness Penalty:** 1.0x penalty for bitrate changes

---

## Data Files

### Network Traces (`cooked_traces/`)
Format: `time bandwidth` (space-separated)
```
0.0 5.0
1.0 5.0
2.0 5.0
...
```
- Time in seconds
- Bandwidth in Mbps

### Video Size Files (`video_size_*`)
One line per video chunk containing chunk size in bytes.

---

## Dependencies

**Required:**
- Python 3.11+
- TensorFlow 2.x (uses `tensorflow.compat.v1` mode)
- NumPy 1.26+

**Installation:**
```bash
pip install tensorflow numpy
```

---

## Key Updates Made (Python 2 to Python 3 / TF1 to TF2)

### multi_video_sim/ Updates:

1. **env.py:**
   - Changed `print` statements to `print()` functions
   - Changed `raw_input()` to `input()`
   - Changed `'rb'` file mode to `'r'` for text files

2. **a3c.py:**
   - Added `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`
   - Removed `tflearn` dependency, replaced with native `tf.layers`
   - Changed `xrange` to `range`
   - Changed `tf.mul` to `tf.multiply`
   - Changed `tf.sub` to `tf.subtract`
   - Changed `reduction_indices` to `axis`
   - Changed `keep_dims` to `keepdims`
   - Changed `tf.scalar_summary` to `tf.summary.scalar`
   - Changed `tf.merge_all_summaries` to `tf.summary.merge_all`

3. **multi_agent.py:**
   - Added `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`
   - Changed `xrange` to `range`
   - Changed `'wb'` file mode to `'w'` for log files
   - Added directory creation with `os.makedirs()`

4. **rl_test.py:**
   - Same TensorFlow compatibility updates
   - Changed `xrange` to `range`
   - Changed file modes

---

## Training Process

### A3C (Asynchronous Advantage Actor-Critic)

1. **Central Agent:**
   - Holds the global actor and critic networks
   - Aggregates gradients from worker agents
   - Updates global parameters
   - Saves model checkpoints periodically

2. **Worker Agents (16 parallel):**
   - Each runs independent video streaming simulation
   - Computes local gradients
   - Sends gradients to central agent
   - Receives updated network parameters

3. **Training Loop:**
   ```
   For each epoch:
     1. Central agent broadcasts network parameters to workers
     2. Workers collect experience (state, action, reward)
     3. Workers compute gradients and send to central
     4. Central aggregates and applies gradients
     5. Log training metrics (reward, TD loss, entropy)
     6. Save checkpoint every 100 epochs
   ```

---

## Output and Monitoring

### Log Files
- `log_central` - Epoch-level training metrics
- `log_agent_*` - Per-agent streaming logs
- `log_test` - Test performance metrics

### Log Format (Agent)
```
timestamp    bitrate    buffer_size    rebuffer    chunk_size    delay    reward
```

### TensorBoard
```bash
tensorboard --logdir=results/
```
View training curves for TD loss, reward, and entropy.

---

## Testing Trained Models

```bash
cd test
python rl_no_training.py ../sim/results/nn_model_ep_100.ckpt
```

Results saved to `test/results/` with per-trace performance logs.

---

## Troubleshooting

### Common Issues:

1. **AssertionError in video file parsing:**
   - Ensure video files have correct number of chunk size lines
   - Format: Line 1 (header), Line 2 (mask), Lines 3+ (chunk sizes)

2. **TensorFlow deprecation warnings:**
   - These are normal when using TF2 with `compat.v1` mode
   - The code still functions correctly

3. **Memory issues with 16 agents:**
   - Reduce `NUM_AGENTS` in multi_agent.py
   - Or set `CUDA_VISIBLE_DEVICES=''` to force CPU

4. **Empty test results:**
   - Ensure test traces exist in `cooked_test_traces/`
   - Ensure test video exists in `test_video/`

---

## References

- Original Pensieve Paper: https://web.mit.edu/pensieve/
- A3C Algorithm: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016
- DASH Streaming: ISO/IEC 23009-1:2014

---

## File Summary

| File | Purpose |
|------|---------|
| `sim/multi_agent.py` | Basic A3C training |
| `sim/a3c.py` | Neural network definition |
| `sim/env.py` | Streaming simulation |
| `multi_video_sim/multi_agent.py` | Multi-video A3C training |
| `multi_video_sim/a3c.py` | Enhanced NN with masking |
| `multi_video_sim/env.py` | Multi-video simulation |
| `test/bb.py` | Buffer-based baseline |
| `test/mpc.py` | MPC baseline |
| `test/rl_no_training.py` | Model evaluation |
