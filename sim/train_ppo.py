"""
Training Script for PPO on Video Streaming (ABR)
Uses the same environment and dataset as Pensieve A3C

PPO (Proximal Policy Optimization) for adaptive bitrate selection.
"""

import os
import sys
import numpy as np
import time
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env
import load_trace
from ppo import PPOAgent, S_INFO, S_LEN, A_DIM

# Training parameters
NUM_EPOCHS = 1000  # Number of training epochs
ROLLOUT_LENGTH = 100  # Steps per rollout
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42

# Video parameters (same as Pensieve)
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1

# Paths
SUMMARY_DIR = './results_ppo'
LOG_FILE = './results_ppo/log_ppo'
TRAIN_TRACES = './cooked_traces/'


def compute_qoe(bitrate, rebuf, last_bitrate):
    """Compute QoE reward (same as Pensieve)"""
    reward = VIDEO_BIT_RATE[bitrate] / M_IN_K \
             - REBUF_PENALTY * rebuf \
             - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bitrate] - VIDEO_BIT_RATE[last_bitrate]) / M_IN_K
    return reward


def train():
    """Main training function for PPO"""

    np.random.seed(RANDOM_SEED)

    # Create results directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Load network traces
    print("Loading network traces...")
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    print(f"Loaded {len(all_file_names)} trace files")

    # Initialize environment
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                               all_cooked_bw=all_cooked_bw,
                               random_seed=RANDOM_SEED)

    # Initialize PPO agent
    print("Initializing PPO agent...")
    agent = PPOAgent(
        state_dim=(S_INFO, S_LEN),
        action_dim=A_DIM,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )

    # Training metrics
    epoch_rewards = []
    epoch_qoe = []
    epoch_losses = []
    training_log = []

    # Open log file
    with open(LOG_FILE + '.txt', 'w') as log_file:
        log_file.write("epoch\tavg_reward\tavg_qoe\tpolicy_loss\tvalue_loss\tentropy\n")

        print("\n" + "="*60)
        print("Starting PPO Training for Video Streaming")
        print("="*60)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        # Initialize state
        state = np.zeros((S_INFO, S_LEN))

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            epoch_reward = 0
            epoch_steps = 0

            # Collect rollout
            for step in range(ROLLOUT_LENGTH):
                # Select action
                action, log_prob, value = agent.select_action(state, training=True)
                bit_rate = action

                # Get video chunk from environment
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

                # Compute reward (QoE)
                reward = compute_qoe(bit_rate, rebuf, last_bit_rate)
                epoch_reward += reward

                # Update state
                next_state = np.roll(state, -1, axis=1)
                next_state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
                next_state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
                next_state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
                next_state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
                next_state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                next_state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # Store experience
                agent.store_experience(state, action, reward, log_prob, value, end_of_video)

                # Update for next step
                last_bit_rate = bit_rate
                state = next_state
                epoch_steps += 1
                agent.steps += 1

                # Handle end of video
                if end_of_video:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY
                    state = np.zeros((S_INFO, S_LEN))

            # Get last value for GAE computation
            with __import__('torch').no_grad():
                state_tensor = __import__('torch').FloatTensor(state).unsqueeze(0)
                _, last_value = agent.network(state_tensor)
                last_value = last_value.item()

            # PPO update
            update_info = agent.update(last_value)

            # Calculate epoch metrics
            avg_reward = epoch_reward / epoch_steps
            avg_qoe = avg_reward

            epoch_rewards.append(avg_reward)
            epoch_qoe.append(avg_qoe)

            if update_info:
                epoch_losses.append(update_info['loss'])

                # Log
                log_file.write(f"{epoch}\t{avg_reward:.4f}\t{avg_qoe:.4f}\t"
                              f"{update_info['policy_loss']:.6f}\t"
                              f"{update_info['value_loss']:.6f}\t"
                              f"{update_info['entropy']:.6f}\n")
                log_file.flush()

                training_log.append({
                    'epoch': epoch,
                    'avg_reward': avg_reward,
                    'avg_qoe': avg_qoe,
                    'policy_loss': update_info['policy_loss'],
                    'value_loss': update_info['value_loss'],
                    'entropy': update_info['entropy'],
                    'steps': agent.steps
                })

                # Print progress
                if epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | "
                          f"QoE: {avg_qoe:7.3f} | "
                          f"P_Loss: {update_info['policy_loss']:.4f} | "
                          f"V_Loss: {update_info['value_loss']:.4f} | "
                          f"Entropy: {update_info['entropy']:.4f} | "
                          f"Steps: {agent.steps:6d} | "
                          f"Time: {elapsed:.1f}s")

            # Save model periodically
            if epoch % MODEL_SAVE_INTERVAL == 0 and epoch > 0:
                model_path = os.path.join(SUMMARY_DIR, f"ppo_model_ep_{epoch}.pth")
                agent.save_model(model_path)

        # Save final model
        final_model_path = os.path.join(SUMMARY_DIR, "ppo_model_final.pth")
        agent.save_model(final_model_path)

        # Save training history
        history_path = os.path.join(SUMMARY_DIR, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total epochs: {NUM_EPOCHS}")
        print(f"Total steps: {agent.steps}")
        print(f"Final QoE: {avg_qoe:.3f}")
        print(f"Model saved to: {final_model_path}")
        print("="*60)

    return epoch_rewards, epoch_qoe, epoch_losses


if __name__ == '__main__':
    train()
