from __future__ import annotations

import json
import math

import numpy as np

####### utils #######


def rotate_axis_using_right_shoulder(points):
    """
    Rotate a 3D axis system with the origin set to new_origin and align the x-axis using the shoulder direction.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        new_origin (numpy.ndarray): Coordinates of new_origin , shape (3,).
        direction_point (numpy.ndarray): Coordinates of direction_point to define the x-axis, shape (3,).

    Returns:
        numpy.ndarray: Transformed points with the new origin and rotated axes.
    """
    new_origin = points[14]  # right shoulder
    direction_point = points[11]  # left shoulder

    # Direction vector for the new x-axis
    x_axis_direction = direction_point - new_origin
    x_axis_direction = x_axis_direction / np.linalg.norm(x_axis_direction)  # Normalize

    # Choose an arbitrary vector for computing the orthogonal z-axis
    arbitrary_vector = np.array([0, 1, 0])

    z_axis = np.cross(x_axis_direction, arbitrary_vector)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis_direction)
    y_axis /= np.linalg.norm(y_axis)

    translated_points = points - new_origin

    # y_axix should be directed on the same direction of the mouth and the nose
    if np.sign(translated_points[9][1]) != np.sign(y_axis[1]):
        y_axis = -y_axis

    # rotation matrix
    rotation_matrix = np.stack([x_axis_direction, y_axis, z_axis], axis=1)

    rotated_points = translated_points @ rotation_matrix

    # rotate 180 degrees around the x-axis
    rotated_points = rotated_points @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return rotated_points


def rotate_axis_using_left_shoulder(points):
    """
    Rotate a 3D axis system with the origin set to new_origin and align the x-axis using the shoulder direction.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        new_origin (numpy.ndarray): Coordinates of new_origin , shape (3,).
        direction_point (numpy.ndarray): Coordinates of direction_point to define the x-axis, shape (3,).

    Returns:
        numpy.ndarray: Transformed points with the new origin and rotated axes.
    """
    new_origin = points[11]  # left shoulder
    direction_point = points[14]  # right shoulder

    # Direction vector for the new x-axis
    x_axis_direction = direction_point - new_origin
    x_axis_direction = -x_axis_direction / np.linalg.norm(
        x_axis_direction
    )  # negative direction

    # Choose an arbitrary vector for computing the orthogonal z-axis
    arbitrary_vector = np.array([0, 1, 0])

    z_axis = np.cross(x_axis_direction, arbitrary_vector)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis_direction)
    y_axis /= np.linalg.norm(y_axis)

    translated_points = points - new_origin

    # y_axix should be directed on the same direction of the mouth and the nose
    if np.sign(translated_points[9][1]) != np.sign(y_axis[1]):
        y_axis = -y_axis

    # rotation matrix
    rotation_matrix = np.stack([x_axis_direction, y_axis, z_axis], axis=1)

    rotated_points = translated_points @ rotation_matrix

    # rotate 180 degrees around the x-axis
    rotated_points = rotated_points @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return rotated_points


def rotate_axis_x(points, new_origin, direction_point):
    """
    Rotate a 3D axis system with the origin set to new_origin and align the x-axis using the shoulder direction.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        new_origin (numpy.ndarray): Coordinates of new_origin , shape (3,).
        direction_point (numpy.ndarray): Coordinates of direction_point to define the x-axis, shape (3,).

    Returns:
        numpy.ndarray: Transformed points with the new origin and rotated axes.
    """

    # Compute the direction vector for the new x-axis
    x_axis_direction = direction_point - new_origin
    x_axis_direction = x_axis_direction / np.linalg.norm(x_axis_direction)  # Normalize

    # Choose an arbitrary vector for computing the orthogonal z-axis
    arbitrary_vector = np.array([0, 1, 0])

    # Compute the new z-axis (orthogonal to x-axis and arbitrary vector)
    z_axis = np.cross(x_axis_direction, arbitrary_vector)
    z_axis /= np.linalg.norm(z_axis)

    # Compute the new y-axis (orthogonal to both x-axis and z-axis)
    y_axis = np.cross(z_axis, x_axis_direction)
    y_axis /= np.linalg.norm(y_axis)

    # Build the rotation matrix (columns are the new basis vectors)
    rotation_matrix = np.stack([x_axis_direction, y_axis, z_axis], axis=1)

    # Translate points to the new origin
    translated_points = points - new_origin

    # Rotate points to align the axes
    rotated_points = translated_points @ rotation_matrix

    rotated_points = rotated_points @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return rotated_points


def calculate_angle(vector_1: np.ndarray, vector_2: np.ndarray) -> int:
    """
    Calculate the angle between two vectors in degrees.

    Parameters:
        vector_1 (numpy.ndarray): First vector.
        vector_2 (numpy.ndarray): Second vector.

    Returns:
        int: The angle between the two vectors in degrees.
    """
    dot_product = np.dot(vector_1, vector_2)
    magnitude_vector_1 = np.linalg.norm(vector_1)
    magnitude_vector_2 = np.linalg.norm(vector_2)

    # Ensure the dot product is within the valid range for arccos
    cos_angle = np.clip(
        dot_product / (magnitude_vector_1 * magnitude_vector_2), -1.0, 1.0
    )

    return int(np.degrees(np.arccos(cos_angle)))


def plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple:
    """
    Calculate the plane equation coefficients from three points.

    Parameters:
        p1 (numpy.ndarray): First point, shape (3,).
        p2 (numpy.ndarray): Second point, shape (3,).
        p3 (numpy.ndarray): Third point, shape (3,).

    Returns:
        tuple: Coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0.
    """
    v1 = p2 - p1
    v2 = p3 - p1

    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector

    a, b, c = normal_vector
    d = -np.dot(normal_vector, p1)

    return a, b, c, d


def project_point_to_plane(point, plane):
    """
    Calculate the projection of a point onto a plane.

    Parameters:
        point: tuple or list
            Coordinates of the point (x, y, z).
        plane: tuple
            Coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0.

    Returns:
        A tuple representing the coordinates of the projected point.
    """
    a, b, c, d = plane

    point = np.array(point)

    normal_vector = np.array([a, b, c])

    distance = (np.dot(normal_vector, point) + d) / np.linalg.norm(normal_vector)

    projection = point - distance * (normal_vector / np.linalg.norm(normal_vector))

    return tuple(projection)


def project_point_to_plane_equation(
    a: float, b: float, c: float, d: float, point: np.ndarray
) -> float:
    """
    Calculate the projection of a point onto a plane using the plane equation.

    Parameters:
        a (float): Coefficient of x in the plane equation.
        b (float): Coefficient of y in the plane equation.
        c (float): Coefficient of z in the plane equation.
        d (float): Constant term in the plane equation.
        point (numpy.ndarray): Coordinates of the point (x, y, z).

    Returns:
        float: The result of the plane equation for the given point.
    """
    return a * point[0] + b * point[1] + c * point[2] + d


######### angles #########


def calculate_elbow_flexion_extension_angle(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
) -> int:
    """
    Calculate the elbow flexion/extension angle.

    Parameters:
        shoulder (numpy.ndarray): Coordinates of the shoulder, shape (3,).
        elbow (numpy.ndarray): Coordinates of the elbow, shape (3,).
        wrist (numpy.ndarray): Coordinates of the wrist, shape (3,).

    Returns:
        int: The elbow flexion/extension angle in degrees.
    """
    shoulder_elbow = shoulder - elbow
    elbow_wrist = wrist - elbow

    return calculate_angle(shoulder_elbow, elbow_wrist)


def calculate_trunk_bending_angle(pelvis: np.ndarray, neck: np.ndarray) -> int:
    """
    Calculate the trunk bending angle.

    Parameters:
        pelvis (numpy.ndarray): Coordinates of the pelvis, shape (3,).
        neck (numpy.ndarray): Coordinates of the neck, shape (3,).

    Returns:
        int: The trunk bending angle in degrees.
    """
    trunk_vector = neck - pelvis

    return calculate_angle(trunk_vector, np.array([0, 0, 1]))


def calculate_upper_arm_posture_angle(points: np.ndarray, side: str) -> int:
    """
    Calculate the angle between the arm and the trunk.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        side (str): 'left' or 'right' indicating which arm to check.

    Returns:
        int: The angle between the arm and the trunk in degrees, or -1 if the arm is in an invalid position.
    """
    if side == "left":
        new_points = rotate_axis_using_left_shoulder(points)
        angle = calculate_angle(new_points[12], new_points[4])
        elbow = new_points[12]
        invalid_position = np.sign(elbow[0]) > 0 and np.sign(elbow[1]) > 0
    else:
        new_points = rotate_axis_using_right_shoulder(points)
        angle = calculate_angle(new_points[15], new_points[1])
        elbow = new_points[15]
        invalid_position = np.sign(elbow[0]) < 0 and np.sign(elbow[1]) < 0

    if invalid_position or np.sign(elbow[1]) < 0:
        return -1
    return angle


def calculate_trunk_twisting_angle(
    hip_left: np.ndarray,
    hip_right: np.ndarray,
    shoulder_left: np.ndarray,
    shoulder_right: np.ndarray,
) -> int:
    """
    Calculate the trunk twisting angle.

    Parameters:
        hip_left (numpy.ndarray): Coordinates of the left hip, shape (3,).
        hip_right (numpy.ndarray): Coordinates of the right hip, shape (3,).
        shoulder_left (numpy.ndarray): Coordinates of the left shoulder, shape (3,).
        shoulder_right (numpy.ndarray): Coordinates of the right shoulder, shape (3,).

    Returns:
        int: The trunk twisting angle in degrees.
    """
    trunk_vector = shoulder_right - shoulder_left
    hip_vector = hip_right - hip_left

    # Remove the y component
    trunk_vector[1] = 0
    hip_vector[1] = 0

    return calculate_angle(trunk_vector, hip_vector)


def calculate_lumbar_spine_posture_angle(
    pelvis: np.ndarray, thorax: np.ndarray, neck: np.ndarray
) -> int:
    """
    Calculate the lumbar spine posture angle.

    Parameters:
        pelvis (numpy.ndarray): Coordinates of the pelvis, shape (3,).
        thorax (numpy.ndarray): Coordinates of the thorax, shape (3,).
        neck (numpy.ndarray): Coordinates of the neck, shape (3,).

    Returns:
        int: The lumbar spine posture angle in degrees.
    """
    pelvis_vector = pelvis - thorax
    thorax_vector = neck - thorax

    return calculate_angle(pelvis_vector, thorax_vector)


def calculate_trunk_bending_sideways(pelvis: np.ndarray, neck: np.ndarray) -> int:
    """
    Calculate the sideways bending angle of the trunk.

    Parameters:
        pelvis (numpy.ndarray): Coordinates of the pelvis, shape (3,).
        neck (numpy.ndarray): Coordinates of the neck, shape (3,).

    Returns:
        int: The sideways bending angle of the trunk in degrees.
    """
    trunk_vector = neck - pelvis

    return calculate_angle(trunk_vector, np.array([0, 1, 0]))


def calculate_knee_flexion_angle(
    hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray
) -> int:
    """
    Calculate the knee flexion angle.

    Parameters:
        hip (numpy.ndarray): Coordinates of the hip, shape (3,).
        knee (numpy.ndarray): Coordinates of the knee, shape (3,).
        ankle (numpy.ndarray): Coordinates of the ankle, shape (3,).

    Returns:
        int: The knee flexion angle in degrees.
    """
    hip_knee = hip - knee
    knee_ankle = ankle - knee

    return calculate_angle(hip_knee, knee_ankle)


def calculate_upper_arm_retroflexion(points: np.ndarray, side: str) -> int:
    """
    Determine if the upper arm is in retroflexion based on the position of the elbow relative to the trunk.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        side (str): 'left' or 'right' indicating which arm to check.

    Returns:
        int: True if the upper arm is in retroflexion, False otherwise.
    """
    if side == "left":
        new_points = rotate_axis_using_left_shoulder(points)
        elbow_y = new_points[12][1]
    else:
        new_points = rotate_axis_using_right_shoulder(points)
        elbow_y = new_points[15][1]

    # If the y-axis value is negative, the elbow is behind the trunk
    return int(elbow_y < 0)


def calculate_arm_abduction(points: np.ndarray, side: str) -> int:
    """
    Determine if the arm is abducted based on the position of the elbow relative to the trunk.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        side (str): 'left' or 'right' indicating which arm to check.

    Returns:
        int: True if the arm is abducted, False otherwise.
    """
    if side == "left":
        new_points = rotate_axis_using_left_shoulder(points)
        elbow = new_points[12][:2]
        return int(np.sign(elbow[0]) > 0 and np.sign(elbow[1]) > 0)
    else:
        new_points = rotate_axis_using_right_shoulder(points)
        elbow = new_points[15][:2]
        return int(np.sign(elbow[0]) < 0 and np.sign(elbow[1]) > 0)


def calculate_external_rotation(points: np.ndarray, side: str) -> int:
    """
    Calculate the external rotation angle of the arm.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        side (str): 'left' or 'right' indicating which arm to check.

    Returns:
        int: The external rotation angle in degrees, or -1 if no external rotation.
    """
    if side == "left":
        new_points = rotate_axis_using_left_shoulder(points)
        elbow = new_points[12][:2]
        if np.sign(elbow[0]) > 0:
            return -1  # no external rotation
    else:
        new_points = rotate_axis_using_right_shoulder(points)
        elbow = new_points[15][:2]
        if np.sign(elbow[0]) < 0:
            return -1  # no external rotation

    return calculate_angle(
        elbow, np.array([0, 1])
    )  # calculate the angle between the elbow and the y-axis


def calculate_lateral_flexion_torso(points: np.ndarray) -> int:
    """
    Determine if there is lateral flexion of the torso based on the angle between the shoulders and the neck.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].

    Returns:
        int: True if there is lateral flexion, False otherwise.
    """

    THRESHOLD = 0.1  # 10%
    shoulder_left_angle = calculate_angle(points[11], points[8])
    shoulder_right_angle = calculate_angle(points[14], points[8])

    magnitude_left = np.linalg.norm(points[11])
    magnitude_right = np.linalg.norm(points[14])

    angle_difference = np.abs(shoulder_left_angle - shoulder_right_angle)
    magnitude_difference = np.abs(magnitude_left - magnitude_right)

    return int(
        angle_difference > max(shoulder_left_angle, shoulder_right_angle) * THRESHOLD
        and magnitude_difference > 0
    )

def calculate_sideway_angle(points: np.ndarray) -> int:
    """
    Calculate the sideways angle of the trunk.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].

    Returns:
        int: The sideways angle of the trunk in degrees.
    """
    angle = calculate_angle(points[8], np.array([0, 0, 1]))
    return int(angle)
def calculate_axial_rotation_torso(points: np.ndarray, direction: np.ndarray) -> int:
    """
    Determine if there is axial rotation of the torso based on the direction.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) with 3D points [x, y, z].
        direction (numpy.ndarray): Direction vector to align the x-axis, shape (3,).

    Returns:
        int: True if there is axial rotation, False otherwise.
    """
    new_points = rotate_axis_x(points, points[0], direction)
    vect = new_points[14][:2]
    # calucalte the angle of the vector
    
    angle_radians = math.atan2(vect[1], vect[0])
    # Convert to degrees if needed
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees = abs(angle_degrees)
    return 180-int(angle_degrees)

    # return int(
    #     not (
    #         np.sign(new_points[11][0]) != np.sign(new_points[14][0])
    #         and np.sign(new_points[11][1]) == np.sign(new_points[14][1])
    #     )
    # )


def calculate_pose(poses: np.ndarray):
    """
    Calculate the pose angles from the poses

    e.g
    {
        "pose_name":[func(x) for x in poses] # array of angles
    }

    poses is a tensor (n_frame, keypoint, 3)
        0        1         2             3               4
    [ pelvis, right_hip, right_knee, right_ankle, left_hip,
        5          6           7      8     9    10         11
     left_knee, left_ankle, thorax, neck, mouth, head, left_shoulder,
        12         13             14           15         16
    left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist]
    """
    pose = {
        "elbow_flexion_extention_left": {
            "value": [
                calculate_elbow_flexion_extension_angle(
                    poses[i][11], poses[i][12], poses[i][13]
                )
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "elbow_flexion_extention_right": {
            "value": [
                calculate_elbow_flexion_extension_angle(
                    poses[i][14], poses[i][15], poses[i][16]
                )
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "trunk_bending": {
            "value": [
                calculate_trunk_bending_angle(poses[i][0], poses[i][8])
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_upper_arm_posture_angle_right": {
            "value": [
                calculate_upper_arm_posture_angle(poses[i], "right")
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_upper_arm_posture_angle_left": {
            "value": [
                calculate_upper_arm_posture_angle(poses[i], "left")
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_axial_rotation_torso": {
            "value": [
                calculate_axial_rotation_torso(poses[i], poses[i][1])
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_sideway_angle": {
            "value": [
                calculate_sideway_angle(poses[i])
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_lumbar_spine_posture_angle": {
            "value": [
                calculate_lumbar_spine_posture_angle(
                    poses[i][0], poses[i][7], poses[i][8]
                )
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_lateral_flexion_torso": {
            "value": [
                calculate_lateral_flexion_torso(poses[i]) for i in range(poses.shape[0])
            ],
            "bool": 1,
        },
        "calculate_knee_angle_left": {
            "value": [
                calculate_knee_flexion_angle(poses[i][4], poses[i][5], poses[i][6])
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_knee_angle_right": {
            "value": [
                calculate_knee_flexion_angle(poses[i][1], poses[i][2], poses[i][3])
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_retroflextion_upper_left": {
            "value": [
                calculate_upper_arm_retroflexion(poses[i], "left")
                for i in range(poses.shape[0])
            ],
            "bool": 1,
        },
        "calculate_retroflextion_upper_right": {
            "value": [
                calculate_upper_arm_retroflexion(poses[i], "right")
                for i in range(poses.shape[0])
            ],
            "bool": 1,
        },
        "calculate_abduction_left": {
            "value": [
                calculate_arm_abduction(poses[i], "left") for i in range(poses.shape[0])
            ],
            "bool": 1,
        },
        "calculate_abduction_right": {
            "value": [
                calculate_arm_abduction(poses[i], "right")
                for i in range(poses.shape[0])
            ],
            "bool": 1,
        },
        "calculate_external_rotation_left": {
            "value": [
                calculate_external_rotation(poses[i], "left")
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
        "calculate_external_rotation_right": {
            "value": [
                calculate_external_rotation(poses[i], "right")
                for i in range(poses.shape[0])
            ],
            "bool": 0,
        },
    }
    print("pose:", pose)
    with open("pose.json", "w") as f:
        json.dump(pose, f)
    return pose


if __name__ == "__main__":
    poses = np.load("output/poses.npy")
    calculate_pose(poses)
