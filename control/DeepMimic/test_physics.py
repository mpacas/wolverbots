import mujoco
import mujoco.viewer
import time
import os

def test_physics():
    # Load the scene
    xml_path = "scene.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    print(f"Loading {xml_path}...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded successfully.")
    print(f"Gravity: {model.opt.gravity}")
    print(f"Timestep: {model.opt.timestep}")

    # Create viewer
    print("Starting viewer. The robot should fall.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Rudimentary time keeping to match simulation time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if time.time() - start > 10:
                print("10 seconds elapsed.")
                break

if __name__ == "__main__":
    test_physics()
