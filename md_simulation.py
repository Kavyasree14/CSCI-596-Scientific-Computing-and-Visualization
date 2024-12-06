import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app import PDBFile, ForceField
from pdbfixer import PDBFixer


def setup_simulation(pdb_file, force_field='amber14-all.xml', water_model='tip3p'):
    # Use PDBFixer to add missing hydrogens
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # Add hydrogens at pH 7.0
    
    # Save the fixed structure
    PDBFile.writeFile(fixer.topology, fixer.positions, open('fixed.pdb', 'w'))
    
    # Now use the fixed PDB file for simulation setup
    pdb = app.PDBFile('fixed.pdb')
    forcefield = app.ForceField(force_field, f'{water_model}.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, 
                                     nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return simulation


def run_simulation(simulation, steps):
    simulation.minimizeEnergy()
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getEnergy=True)

    # Debugging positions and energy
    positions = state.getPositions(asNumpy=True)
    energy = state.getPotentialEnergy()
    print(f"Step completed. Energy: {energy}")
    print(f"Sample positions: {positions[:5]}")

    return state
