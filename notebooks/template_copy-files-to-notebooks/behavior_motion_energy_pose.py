import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_pose_data(csv_path):
    # Skip the first 3 rows so that row 4 (index 0 in the resulting DataFrame) is the first frame.
    df = pd.read_csv(csv_path, skiprows=3)
    print("CSV Columns:", df.columns.tolist())  # Debug print to check column names

    # First column is a frame index or label that we ignore. The remaining columns contain the coordinate data.
    data = df.iloc[:, 1:]

    n_columns = data.shape[1]
    # Three columns per body part: x, y, likelihood.
    if n_columns % 3 != 0:
        raise ValueError("Expected the number of coordinate columns to be a multiple of 3 (x, y, likelihood per body part).")

    num_bodyparts = n_columns // 3
    pose_data = {}

    for frame_idx, row in data.iterrows():
        coords = []
        for bp in range(num_bodyparts):
            # Calculate column positions for the current body part:
            x = row.iloc[bp * 3]       # x-coordinate
            y = row.iloc[bp * 3 + 1]   # y-coordinate
            # Ignore row.iloc[bp * 3 + 2] (likelihood)
            coords.append((x, y))
        pose_data[frame_idx] = coords

    return pose_data


def compute_motion_energy_with_pose(video_path, pose_data, radius=5, show_plot=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame of the video.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_energy = np.zeros_like(prev_gray, dtype=np.float32)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert the current frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute the absolute difference between the current frame and the previous frame.
        diff = cv2.absdiff(gray, prev_gray)

        # Create a blank mask.
        mask = np.zeros_like(gray, dtype=np.uint8)
        # If pose data exists for this frame, draw a filled circle for each body part.
        if frame_index in pose_data:
            for (x, y) in pose_data[frame_index]:
                cv2.circle(mask, (int(x), int(y)), radius, 255, thickness=-1)

        weighted_diff = cv2.bitwise_and(diff, diff, mask=mask)
        motion_energy += weighted_diff.astype(np.float32)

        prev_gray = gray
        frame_index += 1

    cap.release()

    norm_motion_energy = motion_energy / (motion_energy.max() if motion_energy.max() != 0 else 1)

    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(norm_motion_energy, cmap='inferno')
        ax.set_title("Motion Energy with Pose Data")
        plt.colorbar(im, label="Normalized Motion Energy")
        ax.axis('off')

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_png = os.path.join(os.path.dirname(video_path), f"{base_name}_with_pose.png")
        out_svg = os.path.join(os.path.dirname(video_path), f"{base_name}_with_pose.svg")

        plt.savefig(out_png, format='png', dpi=300)
        plt.savefig(out_svg, format='svg')

        plt.show()

    return motion_energy


if __name__ == "__main__":
    ### USER INPUT: Update these paths with your actual file locations. ###
    video_path = "/Users/justinjames/LUPE_Corder-Lab/behavior_snippets/Still/Still_Seg9.mp4"
    pose_csv_path = "/Users/justinjames/LUPE_Corder-Lab/behavior_snippets/Still/Still_Seg9.csv"
    ### USER INPUT: Update these paths with your actual file locations. ###

    pose_data = load_pose_data(pose_csv_path)

    me_with_pose = compute_motion_energy_with_pose(video_path, pose_data, radius=5, show_plot=True)

    # OPTION: Save the raw motion energy array for further analysis:
    np.save("motion_energy_with_pose.npy", me_with_pose)