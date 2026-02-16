
from ll_visualiser.utils import (visualise_meshes, visualise_landmarks, visualise_landmarks_min,
                                 get_files_by_extension, load_landmarks)


def visualise_model(plotter, model_directory, left_original_landmark_file, right_original_landmark_file,
                    left_predicted_landmark_file, right_predicted_landmark_file):

    mesh_files = get_files_by_extension(model_directory, ['.ply', '.stl'])
    left_original_landmarks = load_landmarks(left_original_landmark_file)
    right_original_landmarks = load_landmarks(right_original_landmark_file)
    left_predicted_landmarks = load_landmarks(left_predicted_landmark_file)
    right_predicted_landmarks = load_landmarks(right_predicted_landmark_file)

    visualise_meshes(plotter, mesh_files)
    visualise_landmarks(plotter, left_predicted_landmarks, 'left', 'red')
    visualise_landmarks(plotter, right_predicted_landmarks, 'right', 'red')
    visualise_landmarks_min(plotter, left_original_landmarks, 'left', 'orange')
    visualise_landmarks_min(plotter, right_original_landmarks, 'right', 'orange')

    # Set initial view to frontal view.
    plotter.view_zy(negative=True)
    plotter.add_axes(labels_off=False)
