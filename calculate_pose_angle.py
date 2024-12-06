from __future__ import annotations

import json

import numpy as np


def calculate_dot_product(vector_1, vector_2):
    return np.dot(vector_1, vector_2)


def calculate_magnitude(vector):
    return np.linalg.norm(vector)


def calculate_angle(vector_1, vector_2):
    dot_product = calculate_dot_product(vector_1, vector_2)
    magnitude_vector_1 = calculate_magnitude(vector_1)
    magnitude_vector_2 = calculate_magnitude(vector_2)

    return int(
        np.degrees(np.arccos(dot_product / (magnitude_vector_1 * magnitude_vector_2)))
    )


def calculate_elbow_flexion_extention_angle(shoulder, elbow, wrist):
    shoulder_elbow = elbow - shoulder
    elbow_wrist = wrist - elbow

    return calculate_angle(shoulder_elbow, elbow_wrist)


def calculate_trunk_bending_angle(pelvis, neck):
    """
    Calcualte the angle between the trunk and the vertical axis
    we want to calculate the bending angle of the trunk
    """
    trunk_vector = neck - pelvis

    return calculate_angle(trunk_vector, np.array([0, 1, 0]))


def calculate_upper_arm_posture_angle(shoulder, wrist, hip):
    """
    Calculate the angle between the arm and the trunk
    """

    arm_vector = wrist - shoulder
    trunk_vector = shoulder - hip
    return calculate_angle(arm_vector, trunk_vector)


def calculate_upper_arm_posture_angle_2(shoulder, wrist):
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
    trunk_vector[1] = 0
    hip_vector[1] = 0


def calculate_lumbar_spine_posture_angle(pelvis, thorax, neck):
    
    pelvis_vector = pelvis - thorax
    thorax_vector = neck - thorax

    return calculate_angle(pelvis_vector, thorax_vector)


def calculate_trunk_bending_sideways(hip, neck):
    """
    Calculate the angle between the trunk and the vertical axis
    we want to calculate the bending angle of the trunk
    """
    trunk_vector = neck - hip

    return calculate_angle(trunk_vector, np.array([0, 1, 0]))


def calculate_pose(poses: np.ndarray):
    """
    # TODO describe the function

    poses is a tensor (n_frame, keypoint, 3)

    [ pelvis, right_hip, right_knee, right_ankle, left_hip,
     left_knee, left_ankle, thorax, neck, mouth, head, left_shoulder,
    left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist,]
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
    ]
    print("pose:", pose)
    with open("pose.json", "w") as f:
        json.dump(pose, f)
    return pose


if __name__ == "__main__":
    poses = np.load("output/poses.npy")
    calculate_pose(poses)
