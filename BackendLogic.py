# Backend logic (majority of it already completed in other python files)
# Tells what types of values we should expect to input, API logic, and step/reset activations

from fastapi import FastAPI 
from pydantic import BaseModel 
from RealTrackerEngine import RealtrackerEngine
import traceback

# Creates web server
app = FastAPI()

# Tells us the shape and the correct types of inputs to expect from the frontend (optional but good for practice and structure)
class Config(BaseModel):
    num_objects: int 
    association_method: str
    sigma_base: float 
    range_ref: float  
    lambda_clutter: float 
    gate_threshold: float 
    process_noise: float
    measurement_noise: float
    max_range: float 

# Store the latest configuration 
current_config: Config | None = None

# Our global variable holding our RealtrackerEngine object  --> Global so we dont constantly restart from frame = 0   
engine: RealtrackerEngine | None = None

@app.post("/configure")
def configure_simulation(config: Config):
    global current_config
    current_config = config
    return {"status": "ok", "message": "Configuration stored"}

# Allows the simulation to restart (reset) with new input parameters
@app.get("/reset")
def reset_simulation():
    global engine, current_config

    print("RESET CALLED. current_config =", current_config)

    if current_config is None:
        return {"error": "No configuration provided yet"}

    try:
        cfg = current_config.model_dump()
        print("CONFIG DICT:", cfg)

        engine = RealtrackerEngine(cfg)
        print("ENGINE CREATED:", engine)

        return {"status": "ok", "message": "Simulation reset"}

    except Exception as e:
        print("ENGINE FAILED:", e)
        return {"error": f"Engine failed to initialize: {e}"}
    # A bunch of error checks for when sim was breaking down or I was receiving weird error messages


# Goes through the timesteps of the simulation, giving an output for each "step" (effectively animiating the simulation)
@app.get("/step")
def step_simulation():
    global engine

    if engine is None:
        return {"error": "Simulation not initialized"}

    try:
        frame = engine.step()
        return frame
    except Exception as e:
        print("\n--- BACKEND STEP CRASH ---")
        print("Error:", e)
        traceback.print_exc()
        print("---------------------------\n")
        return {"error": f"Backend crashed: {e}"}

# Similar to the first, a bunch of error checks for debugging

