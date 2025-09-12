"""Main script for running OpenPI policy on RoboCasa tasks."""

import dataclasses
import json
import logging
import pathlib
from datetime import datetime
from typing import Optional

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    """Arguments for running RoboCasa with OpenPI."""
    
    # Output directory
    out_dir: pathlib.Path = pathlib.Path("data/robocasa/videos")
    
    # Results file path
    results_file: pathlib.Path = pathlib.Path("data/robocasa/results.json")
    
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000
    
    # RoboCasa environment settings
    task_catagory: str = "kitchen_drawer"
    task_name: str = "CloseDrawer"  # Default task from RoboCasa
    robots: str = "PandaMobile"  # Default robot from RoboCasa examples
    
    # Action chunking settings - use similar values to aloha_sim
    action_horizon: int = 32
    
    # Episode settings
    num_episodes: int = 3
    max_episode_steps: int = 400
    
    # Display settings
    display: bool = False
    
    save_video: bool = True
    
    # Seed
    seed: Optional[int] = 0


def main(args: Args) -> None:
    """Main function to run RoboCasa evaluation with OpenPI."""
    
    # Create output directory  
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results file directory
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to policy server
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Connected to policy server. Metadata: {ws_client_policy.get_server_metadata()}")
    
    # Create RoboCasa environment
    robocasa_env = _env.RoboCasaEnvironment(
        task_name=args.task_name,
        task_catagory = args.task_catagory,
        robots=args.robots,
        seed=args.seed,
        
    )
    
    logging.info(f"Created RoboCasa environment: {robocasa_env.get_env_info()}")
    
    # Create runtime with action chunking (same pattern as aloha_sim)
    runtime = _runtime.Runtime(
        environment=robocasa_env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            ),
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir,save_video=args.save_video),
        ],
        max_hz=40,  # Control frequency like aloha_sim
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    
    logging.info(f"Starting evaluation for {args.num_episodes} episodes...")
    # Run evaluation
    runtime.run()
    success_times = runtime._environment._success_times
    test_times = runtime._environment._test_times
    
    # Save results to file
    save_results(args.task_name, success_times, test_times, args.results_file)
    
    logging.info(f"Finished evaluation for {args.num_episodes} episodes..., with result {success_times}/{test_times}")


def save_results(task_name: str, success_times: int, test_times: int, results_file: pathlib.Path) -> None:
    """Save task results to a JSON file.
    
    This function safely appends new results to the existing JSON file,
    supporting multiple task runs without overwriting previous results.
    """
    
    # Prepare result data
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_name": task_name,
        "success_times": success_times,
        "test_times": test_times,
        "success_rate": success_times / test_times if test_times > 0 else 0.0,
        "result_string": f"{success_times}/{test_times}"
    }
    
    # Load existing results if file exists
    results = []
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                # Ensure results is a list
                if not isinstance(results, list):
                    logging.warning(f"Results file format unexpected, creating new list")
                    results = []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Could not read existing results file: {e}. Creating new file.")
            results = []
    
    # Append new result
    results.append(result_entry)
    
    # Save updated results with backup strategy
    try:
        # Create a temporary file first to avoid corruption
        temp_file = results_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Atomically replace the original file
        temp_file.replace(results_file)
        
        logging.info(f"Results saved to {results_file}: {task_name} - {success_times}/{test_times}")
        logging.info(f"Total entries in results file: {len(results)}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        # Try to clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
