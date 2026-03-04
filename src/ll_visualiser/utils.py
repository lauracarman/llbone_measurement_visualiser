
import os
import numpy as np
import pyvista as pv


medial_markers = ['MEC', 'condyle_med', 'malleolus_med']


def get_files_by_extension(directory, extensions):
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if any(file.endswith(extension) for extension in extensions)]


def load_landmarks(landmark_file):
    """
    Load landmarks from a file into a dictionary.

    Args:
        landmark_file (str): Path to landmark file.

    Returns:
        dict: {landmark_name: [x, y, z], ...}
    """
    data = np.loadtxt(landmark_file, dtype=str)
    return {row[0]: row[1:].astype(float).tolist() for row in data}


def visualise_meshes(p, mesh_files):
    """
    Args:
        p (pv.Plotter): PyVista Plotter object.
        mesh_files (list): List of mesh files.
    """
    meshes = []
    for file in mesh_files:
        meshes.append(pv.read(file))

    # TODO: This array is used for things like associated check-boxes.
    bones_mesh_actor_arr = []
    for mesh in meshes:
        bones_mesh_actor_arr.append(p.add_mesh(mesh, color='white', show_edges=False, opacity=0.99))


def add_landmark_spheres(p, sphere_meshes, colour='red'):
    for mesh in sphere_meshes:
        p.add_mesh(mesh, color=colour, show_edges=False, opacity=0.99)


def visualise_landmarks(p, landmarks, differences, side, colour='red'):
    """
    Args:
        p (pv.Plotter): PyVista Plotter object.
        landmarks (dictionary): Dictionary of landmarks.
        differences (dictionary): Differences in positions between original and predicted landmarks.
        side (str): Side (left/right).
        colour (str): Landmark sphere colour.
    """
    label_text_color = 'white'

    plot_landmarks_labels, plot_landmarks_points, line_meshes, sphere_meshes = process_landmarks(landmarks, side)
    add_differences(plot_landmarks_labels, differences)

    # Landmark label lines.
    for mesh in line_meshes:
        p.add_mesh(mesh, color='green', show_edges=False, opacity=0.30, line_width=2)

    # Landmark point spheres.
    add_landmark_spheres(p, sphere_meshes, colour)

    # Plots landmark labels.
    justification = 'left' if side == 'left' else 'right'
    landmark_actor = p.add_point_labels(plot_landmarks_points,
                                        plot_landmarks_labels,
                                        text_color=label_text_color,
                                        font_size=12,
                                        always_visible=True,
                                        point_size=0,
                                        justification_horizontal=justification,
                                        shape=None)


def visualise_landmarks_min(p, landmarks, side, colour):
    """
    Visualize only the landmark spheres without labels or lines.
    """
    plot_landmarks_labels, plot_landmarks_points, _, sphere_meshes = process_landmarks(landmarks, side)
    add_landmark_spheres(p, sphere_meshes, colour)


def add_differences(plot_landmarks_labels, differences):
    for i, label in enumerate(plot_landmarks_labels):
        if label in differences:
            plot_landmarks_labels[i] = f"{label} ({differences[label]}m)"


def process_landmarks(landmarks, side, units='m'):
    # Define landmark size and positioning based on units.
    scale = 1000 if units == 'mm' else 1 if units == 'm' else None
    if scale is None:
        raise ValueError(f"Unsupported units: {units}. Use 'm' or 'mm'.")
    spacing = 0.003 * scale
    offset = 0.02 * scale
    z_offset = 0.05 * scale
    sphere_radius = 0.003 * scale

    plot_landmarks_labels = []
    plot_landmarks_points = []
    line_meshes = []
    sphere_meshes = []

    for i, (label, point) in enumerate(landmarks.items()):
        end_point = point.copy()
        end_point[1] += spacing + offset
        end_point[2] += z_offset * (1 if side == "right" else -1) * (-1 if label in medial_markers else 1)

        line_meshes.append(pv.Line(point, end_point))
        sphere_meshes.append(pv.Sphere(radius=sphere_radius, center=point))
        plot_landmarks_labels.append(label)
        plot_landmarks_points.append(end_point)

    return plot_landmarks_labels, plot_landmarks_points, line_meshes, sphere_meshes

def define_measurements():
    measurements = {
        "MAE": 11.0729,
        "RMSE": 12.705,
        "ASIS Width": 0.246,
        "Left Knee Width (m)": 0.0826,
        "Left Ankle Width (m)": 0.0600,
        "Knee Width (Right)": 0.0963,
        "Ankle Width (Right)": 0.626
    }

    return measurements


def calculate_differences(original_landmarks, predicted_landmarks):
    landmarks_of_interest = ['ASIS', 'LEC', 'MEC', 'malleolus_lat', 'malleolus_med']

    differences = {}
    for landmark in landmarks_of_interest:
        if landmark in original_landmarks and landmark in predicted_landmarks:
            original = np.array(original_landmarks[landmark])
            predicted = np.array(predicted_landmarks[landmark])
            differences[landmark] = np.linalg.norm(original - predicted)

    return differences


def visualise_measurements(plotter, measurements):
    data_table_text = ''

    data_table_text += '---- Measurements ----\n\n'
    for key, value in measurements.items():
        data_table_text += f'{key}: {value}\n'

    actor = plotter.add_text(data_table_text,
                       position='upper_right',
                       color='white',
                       font_size=10)
