import os
import json
import pickle
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from SatellitePersonality import SatellitePersonality
from cubesat_detumbling_rl import CubeSatDetumblingEnv
from Metodos_Tabulares import train_sarsa_td


MODEL_DIR = "./models_sarsa_detumbling"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_action_map_from_config(action_map_info):
    max_torque = action_map_info.get("max_torque_constant", SatellitePersonality.MAX_TORQUE_REACTION_WHEEL)
    min_factor = action_map_info.get("min_torque_factor", 0.14672190585150027)
    mid_factor = action_map_info.get("mid_torque_factor", 0.377146607797)
    #micro_factor = action_map_info.get("micro_torque_factor", 0.01)  
    print(f"Building action map with max_torque={max_torque:.6g}, min_factor={min_factor:.6g}, mid_factor={mid_factor:.6g}")
    best_min = min_factor * max_torque
    best_mid = mid_factor * max_torque
    #best_micro = micro_factor * max_torque
    
    print(f"best_min={best_min:.6g}, best_mid={best_mid:.6g}, max_torque={max_torque:.6g}")

    action_map = {
        0: np.array([SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        1: np.array([-SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        2: np.array([0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        3: np.array([0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        4: np.array([0, 0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        5: np.array([0, 0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        6: np.array([best_min, 0, 0]),
        7: np.array([-best_min, 0, 0]),
        8: np.array([0, best_min, 0]),
        9: np.array([0, -best_min, 0]),
        10: np.array([0, 0, best_min]),
        11: np.array([0, 0, -best_min]),
        12: np.array([best_mid, 0, 0]),
        13: np.array([-best_mid, 0, 0]),
        14: np.array([0, best_mid, 0]),
        15: np.array([0, -best_mid, 0]),
        16: np.array([0, 0, best_mid]),
        17: np.array([0, 0, -best_mid]),
        18: np.array([0, 0, 0]),  
    }
    return action_map


def main():
    parser = argparse.ArgumentParser(description="Train final SARSA model from JSON config")
    parser.add_argument("--config", type=str, default=os.path.join(".", "best_config_sarsa.json"), help="Path to JSON config produced by the optimizer")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes for final model")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--numero_bins", type=int, default=15, help="Number of bins for discretization")
    parser.add_argument("--granularity", type=int, default=5, help="Granularity argument for train_sarsa_td")
    parser.add_argument("--plot", action="store_true", help="Show plots after training")
    args = parser.parse_args()

    
    def prompt_int(prompt_text, default_val):
        try:
            s = input(f"{prompt_text} [{default_val}]: ").strip()
            return int(s) if s else default_val
        except Exception:
            print("Invalid input, using default.")
            return default_val

    episodes = prompt_int("Number of episodes", args.episodes)
    max_steps = prompt_int("Max steps per episode", args.max_steps)

    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    best_params = cfg.get("best_params", {})
    action_map_info = cfg.get("action_map_info")

    if cfg.get("action_map"):
        action_map = {int(k): np.array(v) for k, v in cfg.get("action_map").items()}
    elif action_map_info:
        action_map = build_action_map_from_config(action_map_info)
    else:
        
        action_map = build_action_map_from_config({"max_torque_constant": SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, "min_torque_factor": 0.05, "mid_torque_factor": 0.5})

    # En caso de probar otros torques borrar o añadir otros
    params_for_train = best_params.copy()
    params_for_train.pop("min_torque", None)
    params_for_train.pop("mid_torque", None)
    #params_for_train.pop("micro_torque", None)

   
    env = CubeSatDetumblingEnv(max_steps=max_steps, action_map=action_map, fast_mode=True)

    #Reentrenamiento con el JSON en caso de querer probar otros datos en otra cosas
    print("Starting final SARSA training with the following parameters:")
    
    print(f"  episodes={episodes}, max_steps={max_steps}, numero_bins={args.numero_bins}")
    print(f"  training params: {params_for_train}")

    q_table_final, ang_vel_bins_final, rewards_final = train_sarsa_td(
        episodes=episodes,
        max_steps=max_steps,
        fast_mode=True,
        granularity=args.granularity,
        numero_bins=args.numero_bins,
        plot_results=True,
        **params_for_train,
    )

    model_data = {
        "q_table": q_table_final,
        "ang_vel_bins": ang_vel_bins_final,
        "action_map": action_map,
        "best_params": best_params,
    }

    out_path = os.path.join(MODEL_DIR, "sarsa_final_from_json.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Final model saved to: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
