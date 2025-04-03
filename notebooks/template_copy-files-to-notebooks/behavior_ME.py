import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_motion_energy_with_bg_subtraction(
        video_path,
        history=500,
        var_threshold=16,
        detect_shadows=False,
        morph_kernel_size=(3, 3),
        threshold_value=127
):
    """
    Computes a motion energy map from a video using:
      1) Background subtraction to isolate the mouse.
      2) Frame differencing within the foreground mask.

    Adjustable parameters for fine-tuning:
      - history: number of frames for background model.
      - var_threshold: threshold on background/foreground separation.
      - detect_shadows: whether to detect shadows in MOG2.
      - morph_kernel_size: size of structuring element for morphological operations.
      - threshold_value: binarization threshold for the foreground mask.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Create a background subtractor (tune parameters as needed)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame of the video.")

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the motion energy accumulator
    motion_energy = np.zeros_like(prev_gray, dtype=np.float32)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction to get a foreground mask
        fg_mask = bg_subtractor.apply(gray)

        # Optional: apply a small morphological operation to reduce noise
        if morph_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Threshold the mask to ensure it's binary (0 or 255)
        if threshold_value is not None:
            _, fg_mask = cv2.threshold(fg_mask, threshold_value, 255, cv2.THRESH_BINARY)

        # Compute absolute difference between current and previous grayscale frames
        diff = cv2.absdiff(gray, prev_gray)

        # Only accumulate differences where the foreground mask is 255
        masked_diff = cv2.bitwise_and(diff, diff, mask=fg_mask)

        # Accumulate in motion_energy
        motion_energy += masked_diff.astype(np.float32)

        # Update prev_gray
        prev_gray = gray
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise ValueError("Video contains no frames beyond the first read.")

    return motion_energy


def process_videos_in_folder(
        folder_path,
        file_extension="mp4",
        show_plots=True,
        history=500,
        var_threshold=16,
        detect_shadows=False,
        morph_kernel_size=(3, 3),
        threshold_value=127
):
    """
    Processes all video files in a folder using background subtraction +
    frame differencing to generate a motion energy plot focusing on the mouse.

    You can pass custom parameters for the background subtractor and morphological
    operations to see how they affect the output.
    """
    search_pattern = os.path.join(folder_path, f"*.{file_extension}")
    video_files = glob.glob(search_pattern)

    if not video_files:
        raise ValueError(f"No video files with extension '{file_extension}' found in {folder_path}")

    for video_path in video_files:
        print(f"Processing: {video_path}")

        # Compute motion energy with adjustable parameters
        motion_energy = compute_motion_energy_with_bg_subtraction(
            video_path,
            history=history,
            var_threshold=var_threshold,
            detect_shadows=detect_shadows,
            morph_kernel_size=morph_kernel_size,
            threshold_value=threshold_value
        )

        # Normalize the motion energy
        max_val = motion_energy.max() if motion_energy.max() != 0 else 1
        norm_motion_energy = motion_energy / max_val

        # Plot
        plt.figure(figsize=(6, 5))
        plt.imshow(norm_motion_energy, cmap="inferno")
        title_str = (
            f"BG Sub: hist={history}, varThr={var_threshold}, "
            f"shadows={detect_shadows}, kernel={morph_kernel_size}, "
            f"thresh={threshold_value}\n"
            f"{os.path.basename(video_path)}"
        )
        plt.title(title_str)
        plt.colorbar(label="Normalized Motion Energy")
        plt.axis("off")

        # Save
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_png = os.path.join(folder_path, f"{base_name}_motion_energy_bgsub.png")
        out_svg = os.path.join(folder_path, f"{base_name}_motion_energy_bgsub.svg")

        plt.savefig(out_png, format='png', dpi=300)
        plt.savefig(out_svg, format='svg')

        if show_plots:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # Example usage with some default parameters
    folder_path = "/Users/justinjames/LUPE_Corder-Lab/behavior_snippets/Still"

    process_videos_in_folder(
        folder_path,
        file_extension="mp4",
        show_plots=True,
        history=500,  # Try smaller (e.g. 50) or larger (e.g. 1000)
        var_threshold=16,  # Try smaller (5) or larger (30)
        detect_shadows=False,  # Try True if your scene has shadows
        morph_kernel_size=(3, 3),  # Try (5,5) or None
        threshold_value=127  # Try 50 or 200 or None
    )