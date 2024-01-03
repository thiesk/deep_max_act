import cv2
import os
import parameters
def bake_vid(name):
    # Folder containing images
    image_folder = './output/'

    # Output video file
    output_video_path = f'{name}.mp4'

    # Get the list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sort the images based on their numerical part (assuming filenames are numbered sequentially)
    images.sort(key=lambda x: int(x.split('.')[0]))

    # Video properties
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer object
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), parameters.fps, (width, height))

    # Loop through the images and write each frame to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Ensure the frame is not None (image loading success)
        if frame is not None:
            video.write(frame)

    # Release the video writer
    video.release()

    print(f"Video created successfully at: {output_video_path}")
