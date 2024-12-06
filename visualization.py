import pyvista as pv
import numpy as np

def create_protein_mesh(positions):
    points = np.array(positions)
    mesh = pv.PolyData(points)
    mesh['radius'] = np.ones(len(points)) * 0.5
    return mesh

def visualize_protein(mesh):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, render_points_as_spheres=True)
    plotter.show()