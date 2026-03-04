
from ll_visualiser.utils import (visualise_meshes, visualise_landmarks, visualise_landmarks_min,
                                 get_files_by_extension, load_landmarks, define_measurements, visualise_measurements,
                                 calculate_differences)


def visualise_model(plotter, model_directory, left_original_landmark_file, right_original_landmark_file,
                    left_predicted_landmark_file, right_predicted_landmark_file):

    mesh_files = get_files_by_extension(model_directory, ['.ply', '.stl'])
    left_original_landmarks = load_landmarks(left_original_landmark_file)
    right_original_landmarks = load_landmarks(right_original_landmark_file)
    left_predicted_landmarks = load_landmarks(left_predicted_landmark_file)
    right_predicted_landmarks = load_landmarks(right_predicted_landmark_file)

    original_landmarks = {**left_original_landmarks, **right_original_landmarks}
    predicted_landmarks = {**left_predicted_landmarks, **right_predicted_landmarks}
    differences = calculate_differences(original_landmarks, predicted_landmarks)

    visualise_meshes(plotter, mesh_files)
    visualise_landmarks(plotter, left_predicted_landmarks, differences, 'left', 'red')
    visualise_landmarks(plotter, right_predicted_landmarks, differences, 'right', 'red')
    visualise_landmarks_min(plotter, left_original_landmarks, 'left', 'orange')
    visualise_landmarks_min(plotter, right_original_landmarks, 'right', 'orange')

    measurements = define_measurements()
    visualise_measurements(plotter, measurements)

    # Set initial view to frontal view.
    plotter.view_zy(negative=True)
    plotter.add_axes(labels_off=False)
