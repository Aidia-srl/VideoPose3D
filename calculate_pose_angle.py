from __future__ import annotations

import json

import numpy as np

####### utils #######

def rotate_axis_using_right_shoulder(points, new_origin, direction_point):
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

    # rotation matrix
    rotation_matrix = np.stack([x_axis_direction, y_axis, z_axis], axis=1)

    translated_points = points - new_origin
    rotated_points = translated_points @ rotation_matrix

    # rotate 180 degrees around the x-axis
    rotated_points = rotated_points @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return rotated_points


def rotate_axis_using_left_shoulder(points, new_origin, direction_point):
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

    # rotation matrix
    rotation_matrix = np.stack([x_axis_direction, y_axis, z_axis], axis=1)

    translated_points = points - new_origin
    rotated_points = translated_points @ rotation_matrix

    # rotate 180 degrees around the x-axis
    rotated_points = rotated_points @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return rotated_points





def calculate_angle(vector_1, vector_2):
    dot_product = np.dot(vector_1, vector_2)
    magnitude_vector_1 = np.linalg.norm(vector_1)
    magnitude_vector_2 = np.linalg.norm(vector_2)

    return int(
        np.degrees(np.arccos(dot_product / (magnitude_vector_1 * magnitude_vector_2)))
    )


def plane_from_points(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    v1 = p2 - p1
    v2 = p3 - p1

    normal_vector = np.cross(v1, v2)

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


def proj_plane(a, b, c, d, point):
    return a * point[0] + b * point[1] + c * point[2] + d


def calculate_elbow_flexion_extention_angle(shoulder, elbow, wrist):
    shoulder_elbow = elbow - shoulder
    elbow_wrist = wrist - elbow

    return calculate_angle(shoulder_elbow, elbow_wrist)


def calculate_trunk_bending_angle(pelvis, neck):
    """
    trunk inclination angle
    Calcualte the angle between the trunk and the vertical axis
    we want to calculate the bending angle of the trunk

    it works only when the person is standing
    """
    trunk_vector = neck - pelvis

    return calculate_angle(trunk_vector, np.array([0, 0, 1]))


def calculate_upper_arm_posture_angle(shoulder, elbow, hip):
    """
    Calculate the angle between the arm and the trunk
    """

    arm_vector = elbow - shoulder
    trunk_vector = hip - shoulder
    return calculate_angle(arm_vector, trunk_vector)


def calculate_upper_arm_posture_angle_2(shoulder, wrist):
    # TODO check if is the same as calculate_upper_arm_posture_angle
    arm_vector = wrist - shoulder
    horizontal_vector = np.array([1, 0, 0])
    return calculate_angle(arm_vector, horizontal_vector)


def calculate_trunk_twsting_angle(hip_left, hip_right, shoulder_left, shoulder_right):
    """
    Calculate the angle between the trunk and the vertical axis
    we want to calculate the twisting angle of the trunk
    """
    trunk_vector = shoulder_right - shoulder_left
    hip_vector = hip_right - hip_left

    # remove the y component
    trunk_vector[2] = 0
    hip_vector[2] = 0

    return calculate_angle(trunk_vector, hip_vector)


def calculate_lumbar_spine_posture_angle(pelvis, thorax, neck):
    # For Sitting: Convex Lumbar Spine Posture

    pelvis_vector = pelvis - thorax
    thorax_vector = neck - thorax

    return calculate_angle(pelvis_vector, thorax_vector)


def calculate_trunk_bending_sideways(pelvis, neck):
    """
    Calculate the angle between the trunk and the vertical axis
    we want to calculate the bending angle of the trunk
    """
    trunk_vector = neck - pelvis

    return calculate_angle(trunk_vector, np.array([0, 1, 0]))


def calculate_knee_flextion_angle(hip, knee, ankle):
    """
    if the flextion is under 40 degrees, it is considered as extreme flextion07
    """

    hip_knee = hip - knee
    knee_ankle = ankle - knee

    return calculate_angle(hip_knee, knee_ankle)


def calculate_upper_arm_retroflexion(
    shoulder_left, shoulder_right, pelvis, elbow, mouth
):
    # calculate the plan that is formed by the shoulders and the pelvis
    plane = plane_from_points(shoulder_left, shoulder_right, pelvis)

    elbow_sign = np.sign(proj_plane(*plane, elbow))
    mouth_sign = np.sign(proj_plane(*plane, mouth))

    # if the elbow and the mouth are on the same side of the plane, the sign is positive
    return elbow_sign * mouth_sign


def calculate_arm_abduction_angle(shoulder, elbow, wrist):
    # creare piano spallae pelvico, calcolare proiezione, la distanza punto tra polso spalla
    pass


def calculate_external_rotation(shoulder, elbow, wrist):
    # angolo del gomito deve essere 90
    # il polso deve avere segno opposto della proiezione rispetto al naso
    pass


def calculate_lateral_flexion_torso(shoulder, hip, ankle):
    # spalle devono essere parallel al bacino
    pass


def calculate_axial_rotation_torso(shoulder, hip, ankle):
    # spalle e bacino devono incidere su piu punti, se incide su un punto solo è flessione laterale
    pass


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
    pose = [
        {
            "elbow_flexion_extention_left": [
                calculate_elbow_flexion_extention_angle(
                    poses[i][11], poses[i][12], poses[i][13]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "elbow_flexion_extention_right": [
                calculate_elbow_flexion_extention_angle(
                    poses[i][14], poses[i][15], poses[i][16]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "trunk_bending": [
                calculate_trunk_bending_angle(poses[i][0], poses[i][8])
                for i in range(poses.shape[0])
            ]
        },
        {
            "calculate_upper_arm_posture_angle_right": [
                calculate_upper_arm_posture_angle(
                    shoulder=poses[i][14], elbow=poses[i][15], hip=poses[i][1]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "calculate_upper_arm_posture_angle_left": [
                calculate_upper_arm_posture_angle(
                    poses[i][11], poses[i][12], poses[i][4]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "calculate_trunk_twsting_angle": [
                calculate_trunk_twsting_angle(
                    poses[i][4], poses[i][1], poses[i][11], poses[i][14]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "calculate_lumbar_spine_posture_angle": [
                calculate_lumbar_spine_posture_angle(
                    poses[i][0], poses[i][7], poses[i][8]
                )
                for i in range(poses.shape[0])
            ]
        },
        {
            "calculate_trunk_bending_sideways": [
                calculate_trunk_bending_sideways(poses[i][0], poses[i][8])
                for i in range(poses.shape[0])
            ]
        },
    ]
    print("pose:", pose)
    with open("pose.json", "w") as f:
        json.dump(pose, f)
    return pose


if __name__ == "__main__":
    poses = np.load("output/poses.npy")
    calculate_pose(poses)
