# CSCI 596 Final Project
 

# **Protein Folding Simulation with Machine Learning**

This project demonstrates a **protein folding simulation** integrated with **machine learning** to predict protein structures dynamically. Using **OpenMM**, we simulate the molecular dynamics of a protein, visualize it in 3D using **PyVista**, and leverage a machine learning model for real-time structure predictions. GPU acceleration is integrated to ensure high performance.

Why This Project?
Protein folding is a critical biological process where a protein assumes its functional 3D shape. Misfolding can cause diseases like Alzheimer’s and Parkinson’s, while accurate folding is essential for drug design and biotechnology.
I chose this project because it combines molecular simulations and machine learning, offering a powerful, scalable solution for real-time protein analysis.

**Project Workflow**

Input:
  -Protein structures in PDB format are cleaned and preprocessed using PDBFixer.
 
Simulation:
 -OpenMM models the folding process using accurate force fields.
 
Visualization:
 -PyVista renders the 3D protein structure in real time, updating atom positions dynamically.
 
Machine Learning:
 -A neural network trained on augmented protein data predicts the folded structure.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Model Training](#model-training)
- [GPU Acceleration](#gpu-acceleration)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## **Overview**

Protein folding is a fundamental biological process where a protein assumes its functional three-dimensional shape, critical for its biological function. This project integrates **molecular dynamics simulations**, **machine learning**, and **3D visualization** to:
1. First, I used **OpenMM** to simulate protein folding dynamics, starting with a PDB file to model the molecular forces accurately
2. Second, I leveraged **PyVista** for real-time 3D visualization of the protein structure. Atoms are represented as spheres, and the visualization dynamically updates during the folding process, with an interactive slider to control simulation steps. 
3. Lastly, I built a neural network using PyTorch to predict protein structures in real time based on atom coordinates, enabling real-time insights during the simulation.

---

## **Features**

- **Realistic Protein Simulation**: 
   - Uses **OpenMM** to simulate protein folding dynamics with accurate molecular forces.
- **3D Visualization**: 
   - Leverages **PyVista** to render proteins in real-time, showcasing atom-level dynamics.
- **Machine Learning Predictions**:
   - A neural network trained on simulation data predicts protein structures during the folding process.
- **GPU Acceleration**:
   - Utilizes CUDA for faster simulations and training.
- **Interactive Controls**:
   - Adjust simulation steps with a slider in the visualization interface.

---

## **Tech Stack**

### **Core Technologies**
- **Python**: Programming language for implementation.
- **OpenMM**: Molecular dynamics engine for protein simulations.
- **PyTorch**: For building and training the machine learning model.
- **PyVista**: For interactive 3D visualization.

### **Libraries**
- **NumPy**: Numerical computations and data handling.
- **Matplotlib**: For visualizing training loss curves.
- **PDBFixer**: For fixing and cleaning PDB files.

---

## **Project Structure**

```
ProteinFoldingSimulation/
├── protein_folding_env/         # Virtual environment (not included in repo)
├── main.py                      # Main script for simulation, training, and visualization
├── md_simulation.py             # Handles molecular dynamics simulation with OpenMM
├── ml_model.py                  # Defines the machine learning model and training pipeline
├── visualization.py             # 3D visualization utilities using PyVista
├── protein.pdb                  # Input protein structure in PDB format
├── fixed.pdb                    # Fixed and cleaned PDB file for simulation
├── README.md                    # Documentation (this file)
```

---

## **Installation**

### Prerequisites
- Python 3.8 or above
- **GPU Requirements**:
   - NVIDIA GPU with CUDA support (if available).
   - Install CUDA drivers and `torch` with GPU support.
- Required libraries:
   - `openmm`
   - `pytorch`
   - `pyvista`
   - `pdbfixer`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/ProteinFoldingSimulation.git
   cd ProteinFoldingSimulation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv protein_folding_env
   source protein_folding_env/bin/activate  # Linux/Mac
   protein_folding_env\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add the `amber14-all.xml` and `tip3p.xml` force field files to your environment (required by OpenMM).

---

## **Usage**

### **Step 1: Run the Simulation**
Run the main script to simulate, visualize, and predict protein structures:
```bash
python main.py
```

- The program will:
  - Train the ML model.
  - Simulate protein folding using OpenMM.
  - Visualize the protein dynamics in 3D.
  - Predict protein structures in real-time.

### **Step 2: Adjust Simulation Dynamically**
- Use the interactive slider in the PyVista visualization window to control the number of simulation steps dynamically.

---

## **Visualization**

The 3D visualization displays the folding process in real time:
- **Atoms**: Represented as spheres.
- **Dynamic Simulation**: Watch atom positions update step-by-step as the simulation progresses.

### **Example Snapshot**
Here’s how the visualization might look:


---<img width="499" alt="Screenshot 2024-12-06 at 3 22 17 AM" src="https://github.com/user-attachments/assets/30a4d643-c32a-4ca8-a2bc-3b5a3de6f190">


## **Model Training**

### **Machine Learning Model**
- The model is a **neural network** (`ProteinStructurePredictor`) that predicts protein structures:
  - **Input**: Atom positions flattened into a 1D vector.
  - **Output**: Predicted structure class (3 classes in this example).
  - **Loss Function**: Focal Loss for improved training stability.
  - **Optimization**: Adam optimizer with a learning rate scheduler.

### **Training Loss Curve**
After 50 epochs of training, the loss curve is:
<img width="634" alt="Screenshot 2024-12-06 at 3 14 28 AM" src="https://github.com/user-attachments/assets/d00ee0f1-5c88-4a16-8bc8-5d027b82dcca">


## **GPU Acceleration**

This project leverages GPU acceleration for both **machine learning** and **molecular dynamics simulation**:
1. **PyTorch GPU Support**:
   - Automatically detects CUDA if available.
   - Tensors and the model are moved to the GPU for faster training.
2. **OpenMM CUDA Platform**:
   - Ensures molecular dynamics simulations are accelerated using NVIDIA GPUs.

To ensure GPU usage:
- PyTorch: Confirm with `torch.cuda.is_available()`.
- OpenMM: Explicitly set the platform to CUDA:
  ```python
  from openmm import Platform
  platform = Platform.getPlatformByName('CUDA')
  ```

---

## **Future Improvements**

1. **Expand Training Data**:
   - Use more PDB files to train the ML model on diverse protein structures.
   - <img width="923" alt="Screenshot 2024-12-06 at 3 23 30 AM" src="https://github.com/user-attachments/assets/c10de5e1-c7cc-407b-a337-dd1f016dd34d">
2. **Advanced Visualization**:
   - Include bonds between atoms to visualize the full protein structure.
     

https://github.com/user-attachments/assets/d1519ab5-9c41-4f49-af92-d81e5ae8e929


3. **Improved Model Architecture**:
   - Experiment with more advanced deep learning models (e.g., Transformers).
4. **Protein Folding Analysis**:
   - Calculate metrics like RMSD (Root Mean Square Deviation) for validation.
<img width="603" alt="Screenshot 2024-12-06 at 3 30 02 AM" src="https://github.com/user-attachments/assets/37773b0e-13d6-4884-9e7d-c6e6cfb6f1e1">


## **Acknowledgments**

- **OpenMM**: For its powerful molecular dynamics capabilities.
- **PyTorch**: For flexible and scalable machine learning.
- **PyVista**: For interactive and customizable 3D visualizations.
- **PDBFixer**: For preparing PDB files for simulations.

---


## **Contact**

For questions or contributions, feel free to reach out:
- **Name**: Kavya Sree Polavarapu
- **Email**: kpolavar@usc.edu
