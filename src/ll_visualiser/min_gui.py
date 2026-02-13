import os
import pyvista as pv

from ll_visualiser.utils import visualise_meshes, visualise_landmarks, get_files_by_extension, get_files_by_names, load_landmarks

pv.global_theme.allow_empty_mesh = True


original_landmark_files = ["orignial_lms_left.txt", "orignial_lms_right.txt"]
predicted_landmark_files = ["predicted_lms_left.txt", "predicted_lms_right.txt"]


if __name__ == "__main__":
    p = pv.Plotter(lighting='light kit', theme=pv.set_plot_theme('default'), window_size=[2560, 1440])
    p.set_background('dimgrey')

    model_directory = os.path.join('..', '..', 'test', 'test_data')
    mesh_files = get_files_by_extension(model_directory, ['.ply', '.stl'])
    predicted_landmark_files = get_files_by_names(model_directory, predicted_landmark_files)
    predicted_landmarks = load_landmarks(predicted_landmark_files)

    visualise_meshes(p, mesh_files)
    visualise_landmarks(p, predicted_landmarks)

    # Set initial view to frontal view.
    p.view_zy(negative=True)
    p.add_axes(labels_off=False)
    p.show(interactive=True, auto_close=False)
