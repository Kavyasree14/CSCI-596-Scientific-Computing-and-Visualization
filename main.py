import torch
from md_simulation import setup_simulation, run_simulation
from ml_model import ProteinStructurePredictor, train_model, predict_structure
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import openmm.unit as unit


def create_sphere(position, radius=0.1, color="white"):
    """Create a sphere at a given position."""
    sphere = pv.Sphere(radius=radius, center=position)
    return sphere


def interactive_simulation(simulation, model, steps=1000):
    # Initialize PyVista plotter
    plotter = pv.Plotter()

    # Initial positions of atoms
    initial_positions = np.array(
        simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ).reshape(-1, 3)

    # Create spheres for all atoms and add them to the plotter
    spheres = []
    for position in initial_positions:
        sphere = create_sphere(position, radius=0.2, color="blue")  # Radius scaled for visibility
        plotter.add_mesh(sphere, color="blue")
        spheres.append(sphere)

    def update_plot(step):
        # Run a few steps of the simulation
        state = run_simulation(simulation, 10)
        positions = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(-1, 3)

        # Update positions of spheres
        for sphere, new_position in zip(spheres, positions):
            sphere.center = new_position

        # ML prediction (if applicable)
        input_data = torch.tensor(positions.flatten(), dtype=torch.float32).unsqueeze(0)
        prediction = predict_structure(model, input_data)
        print(f"Step {step}: Predicted structure: {prediction.argmax().item()}")

        # Update the plotter
        plotter.render()

    # Add a slider widget for interactive simulation steps
    plotter.add_slider_widget(update_plot, [0, steps], title="Simulation Step")

    # Show the visualization
    plotter.show()


# Set up the molecular dynamics simulation
simulation = setup_simulation('/Users/kavyasreepolavarapu/ProteinFoldingSimulation/protein_folding_env/protein.pdb')

# Set up the machine learning model
input_size = simulation.topology.getNumAtoms() * 3  # 3 coordinates per atom
model = ProteinStructurePredictor(input_size=input_size, hidden_size=100, output_size=3)

# Generate realistic training data
train_data = []
train_labels = []

for i in range(100):
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten()

    # Add noise and normalize positions
    noise = np.random.normal(0, 0.01, positions.shape)  # Add small random noise
    augmented_positions = positions + noise
    normalized_positions = (augmented_positions - np.mean(augmented_positions)) / np.std(augmented_positions)

    train_data.append(normalized_positions)
    train_labels.append(i % 3)  # Cyclic labels: 0, 1, 2

# Convert list to tensor
train_data = torch.tensor(np.array(train_data), dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Train the model
losses = train_model(model, train_data, train_labels, epochs=50)

# Plot training loss curve
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Run the interactive 3D simulation
interactive_simulation(simulation, model)
