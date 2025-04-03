import deeplabcut

# Path to your project's config file
config_path = '/Users/justinjames/LUPE_Corder-Lab/LUPE_MALE-CORDERLAB-2022-12-05/config.yaml'

# List of new videos you want to analyze
videos = ['/Users/justinjames/Desktop/2404.T0_L1_3in_vonfrey_cylinders_20240604_120554293.mp4']

# Analyze the videos
deeplabcut.analyze_videos(config_path, videos)

# Create labeled videos to visualize the tracking
deeplabcut.create_labeled_video(config_path, videos)

# Extract the data to a CSV file
deeplabcut.analyze_videos(config_path, videos, save_as_csv=True)

# Plot the trajectories
deeplabcut.plot_trajectories(config_path, videos)