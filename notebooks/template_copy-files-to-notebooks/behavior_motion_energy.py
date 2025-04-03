import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_motion_energy(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame of the video.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_energy = np.zeros_like(prev_gray, dtype=np.float32)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, prev_gray)

        motion_energy += diff.astype(np.float32)

        prev_gray = gray
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise ValueError("Video contains no frames beyond the first read.")

    return motion_energy


def process_videos_in_folder(folder_path, file_extension="mp4", show_plots=True):
    search_pattern = os.path.join(folder_path, f"*.{file_extension}")
    video_files = glob.glob(search_pattern)

    if not video_files:
        raise ValueError(f"No video files with extension '{file_extension}' found in {folder_path}")

    for video_path in video_files:
        print(f"Processing: {video_path}")
        motion_energy = compute_motion_energy(video_path)

        norm_motion_energy = motion_energy / (motion_energy.max() if motion_energy.max() != 0 else 1)

        plt.figure(figsize=(6, 5))
        plt.imshow(norm_motion_energy, cmap="inferno")
        plt.title(f"Motion Energy: {os.path.basename(video_path)}")
        plt.colorbar(label="Normalized Motion Energy")
        plt.axis("off")

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_png = os.path.join(folder_path, f"{base_name}_motion_energy.png")
        out_svg = os.path.join(folder_path, f"{base_name}_motion_energy.svg")

        # Save the figure in PNG and SVG
        plt.savefig(out_png, format='png', dpi=300)
        plt.savefig(out_svg, format='svg')

        # Show the plot if requested
        if show_plots:
            plt.show()
        else:
            # If not showing, close the figure to free memory
            plt.close()


if __name__ == "__main__":
    ### USER INPUT: Update these paths with your video file locations. ###
    folder_path = "/Users/justinjames/LUPE_Corder-Lab/behavior_snippets/Still"

    process_videos_in_folder(folder_path, file_extension="mp4", show_plots=True)