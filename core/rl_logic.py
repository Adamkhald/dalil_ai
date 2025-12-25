import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    def __init__(self, update_func, frame_func=None, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.update_func = update_func
        self.frame_func = frame_func
        self.last_render_time = 0

    def _on_step(self) -> bool:
        # Update progress every 100 steps
        if self.n_calls % 100 == 0:
            percentage = min(100, int((self.num_timesteps / self.total_timesteps) * 100))
            self.update_func(percentage, f"Step: {self.num_timesteps}/{self.total_timesteps}")
            
        # Live Preview (Throttle to ~30 FPS if possible, or just every N steps)
        if self.frame_func:
            # Render every few steps to avoid killing performance
            if self.n_calls % 5 == 0: 
                try:
                    # SB3 wraps env in VectorEnv, getting the underlying frame can be tricky
                    # We use the first env in the vector
                    frame = self.training_env.envs[0].render()
                    if frame is not None:
                        self.frame_func(frame)
                except Exception:
                    pass # Ignore render errors during live train
        
        return True

    def _on_training_start(self) -> None:
        self.total_timesteps = self.locals['total_timesteps']

from stable_baselines3.common.evaluation import evaluate_policy

class RLPipeline:
    def __init__(self):
        self.model = None
        self.env_id = None
        self.algo = None

    def train_agent(self, env_id, algo_name, timesteps, progress_callback_func, frame_callback_func=None):
        """
        Trains an agent on the specified environment.
        """
        self.env_id = env_id
        
        # Create Environment
        # If preview is requested, we MUST use rgb_array
        render_mode = "rgb_array" if frame_callback_func else None
        
        try:
            # Handle user requested version updates dynamically if needed
            if env_id == "LunarLander-v2": env_id = "LunarLander-v3"
            
            env = gym.make(env_id, render_mode=render_mode)
        except Exception as e:
            return f"Error creating env: {e}"

        # 1. Select Algorithm Class
        if algo_name.startswith("PPO"):
            model_cls = PPO
        elif algo_name.startswith("DQN"):
            model_cls = DQN
        elif algo_name.startswith("A2C"):
            model_cls = A2C
        elif algo_name.startswith("SAC"):
            model_cls = SAC
        else:
            env.close()
            return f"Unknown algorithm: {algo_name}"
            
        # 2. Check Compatibility (Discrete vs Continuous)
        # DQN only works with Discrete
        if algo_name == "DQN" and not isinstance(env.action_space, gym.spaces.Discrete):
            env.close()
            return f"Error: DQN only supports Discrete action spaces. {env_id} is Continuous."
        # SAC only works with Continuous (Box)
        if algo_name == "SAC" and not isinstance(env.action_space, gym.spaces.Box):
            env.close()
            return f"Error: SAC only supports Continuous action spaces. {env_id} is Discrete."

        # Create Model
        try:
            self.model = model_cls("MlpPolicy", env, verbose=0)
        except Exception as e:
            env.close()
            return f"Model init failed: {str(e)}"

        # Callback for UI
        callback = TrainingCallback(progress_callback_func, frame_callback_func)
        
        # Train
        try:
            self.model.learn(total_timesteps=timesteps, callback=callback)
            self.model.save(f"{algo_name}_{env_id}")
            msg = "Training Finished!"
        except Exception as e:
            msg = f"Training failed: {str(e)}"
        
        env.close()
        return msg

    def evaluate_agent(self, env_id, episodes=10):
        if not self.model:
            return "No model trained."
            
        try:
            if env_id == "LunarLander-v2": env_id = "LunarLander-v3"
            eval_env = gym.make(env_id, render_mode="rgb_array")
            
            mean_reward, std_reward = evaluate_policy(self.model, eval_env, n_eval_episodes=episodes)
            eval_env.close()
            
            return f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}"
            
        except Exception as e:
            return f"Evaluation failed: {str(e)}"

    def run_preview(self, env_id, episodes=3):
        """
        Generator that yields frames (numpy arrays) for the UI to display.
        """
        # If model is already trained in memory, use it. Otherwise try to load or use random.
        model = self.model
        
        # Determine strict render mode for frame capture
        try:
            if env_id == "LunarLander-v2": env_id = "LunarLander-v3"
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception as e:
            yield None
            return

        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                if model:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample() # Random agent if no model
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                frame = env.render()
                yield frame
                
        env.close()
