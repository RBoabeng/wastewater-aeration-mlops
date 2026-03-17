import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the BSM1 Open Loop model (aeration tanks & clarifier only)
from bsm2_python import BSM1OL 

def run_simulation_and_extract_data():
    print("Initializing Wastewater Plant Simulation...")
    
    # 1. Initialize the BSM1 Open Loop model
    plant = BSM1OL()
    
    # Arrays to store our extracted data over time
    time_steps = []
    do_levels = []       # Dissolved Oxygen (S_O) in Reactor 5
    ammonia_levels = []  # Ammonia (S_NH) in Reactor 5
    bod_levels = []      # Biological Oxygen Demand proxy (S_S)

    print("Running dynamic simulation (this may take a few moments)...")
    
    # 2. Run the simulation step-by-step
    for idx, current_time in enumerate(tqdm(plant.simtime)):
        
        # Default aeration (kla) settings for the 5 BSM1 reactors
        # Reactors 1 & 2 are anoxic (0 oxygen), Reactors 3, 4, & 5 are aerated
        default_klas = np.array([0, 0, 120, 120, 120]) 
        
        # Step the physics and biology engine forward by one time increment
        plant.step(idx, default_klas)
        
        # 3. Extract the states from the 5th Reactor directly using y_out5
        # Index 7 = Dissolved Oxygen
        # Index 9 = Ammonia
        # Index 1 = Readily biodegradable substrate (BOD proxy)
        reactor_5_states = plant.y_out5
        
        time_steps.append(current_time)
        do_levels.append(reactor_5_states[7])
        ammonia_levels.append(reactor_5_states[9])
        bod_levels.append(reactor_5_states[1])

    # 4. Format the data into a Machine Learning-ready Pandas DataFrame
    print("\nFormatting data...")
    df = pd.DataFrame({
        'time_days': time_steps,
        'dissolved_oxygen_mgL': do_levels,
        'ammonia_mgL': ammonia_levels,
        'bod_proxy_mgL': bod_levels
    })
    
    # 5. Save the dataset to the raw data folder securely
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    output_path = os.path.join(project_root, "data", "raw", "bsm_training_data.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Success! Raw dataset generated with {len(df)} rows.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    run_simulation_and_extract_data()