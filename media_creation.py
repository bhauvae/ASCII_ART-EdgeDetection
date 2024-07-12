from PIL import Image, ImageSequence
import numpy as np
import sys
from scipy.ndimage import gaussian_filter
import cv2
import os
import subprocess
import re
import shutil
import tempfile
import glob
import multiprocessing
from ASCII import ascii_filter as process


def process_gif_file(input_gif_path, output_dir="output", affix="-crt", **kwargs):
    """
    Process a GIF file by extracting its frames, applying effects to each frame with `process_image`,
    and reassembling the frames into a new GIF file.
    """
    gif = Image.open(input_gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    def convert_rgba_to_rgb_with_black_background(frame):
        """
        Convert an RGBA image to an RGB image, filling any transparent areas with black.

        Args:
            frame (PIL.Image): The source image in RGBA mode.

        Returns:
            PIL.Image: The converted image in RGB mode with a black background.
        """
        # Create a new black background image in RGBA mode
        black_background = Image.new("RGBA", frame.size, (0, 0, 0, 255))

        # Composite the RGBA frame onto the black background
        rgb_frame_with_alpha = Image.alpha_composite(black_background, frame)

        return rgb_frame_with_alpha

    processed_frames = []
    for i, frame in enumerate(frames[1:]):
        # Convert RGBA to RGB with a black background if necessary
        if frame.mode == "RGBA":
            frame = convert_rgba_to_rgb_with_black_background(frame)
        else:
            frame = frame.convert("RGB")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", mode="w+b"
        ) as temp_frame_file:
            frame.save(temp_frame_file, format="PNG")
            temp_frame_file.flush()
            os.fsync(temp_frame_file.fileno())
            # Process the frame and save it to the output directory
            processed_frame_path = process_image(
                temp_frame_file.name, output_dir=output_dir, affix=affix, **kwargs
            )
            processed_frames.append(Image.open(processed_frame_path))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine new filename and output path
    base_name = os.path.basename(input_gif_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}.gif"
    output_gif_path = os.path.join(output_dir, new_name)

    # Save processed frames as a new GIF
    processed_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=processed_frames[1:],
        loop=0,
        format="GIF",
        duration=gif.info["duration"],
    )

    print(f"Processed GIF saved to {output_gif_path}")

    # Clean up temporary files
    for frame in processed_frames:
        frame.close()
    for temp_frame_file in glob.glob(os.path.join(output_dir, f"*{affix}.png")):
        os.unlink(temp_frame_file)


def get_video_framerate(video_path):
    """
    Use ffprobe to get the framerate of the video.

    Args:
        video_path: Path to the video file.

    Returns:
        Framerate of the video as a float.
    """
    cmd = [
        "ffprobe",
        "-v",
        "0",
        "-of",
        "csv=p=0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        video_path,
    ]
    output = subprocess.check_output(cmd).decode().strip()
    try:
        num, den = map(int, output.split("/"))
        framerate = num / den
    except ValueError:
        framerate = 30.0  # Fallback framerate
    return framerate


def process_image(image_path, output_dir="output", affix="-ascii", **kwargs):
    """
    Process the image by stretching and applying scanline modulation.
    Args:
        image_path: Path to the input image.
        output_dir: Directory where the processed image will be saved.
        affix: String to be affixed to the output filename.
        **kwargs: Additional keyword arguments for image processing effects.
    Returns:
        Path to the processed image.
    """

    image = process(filename=image_path)

    # Determine new filename and output path
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}{ext}"
    output_path = os.path.join(output_dir, new_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the color space from BGR to RGB

    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")

    return output_path


def process_video_file(input_video_path, output_video_filename, **kwargs):
    """
    Process a video file by extracting its frames, applying effects to each frame,
    and reassembling the frames into a new video file using the original video's framerate.

    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        **kwargs: Additional keyword arguments passed to the image processing function.
    """
    temp_frame_dir = "imgs"  # Use the hardcoded directory from extract.py
    os.makedirs(temp_frame_dir, exist_ok=True)

    current_dir = os.getcwd()  # Get the current working directory
    output_video_path = os.path.join(current_dir, output_video_filename)

    # Extract frames using external script
    subprocess.call(["python", "extract.py", input_video_path])

    # Get the list of extracted frame files
    frame_files = sorted(os.listdir(temp_frame_dir))

    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() // 2))
    print(f"Processing frames with {pool._processes} worker process(es)...")

    processed_frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frame_dir, frame_file)
        if frame_path.lower().endswith((".png", ".jpg", ".jpeg")):
            processed_frame = pool.apply_async(
                process_image, (frame_path,), {"output_dir": "output", **kwargs}
            )
            processed_frames.append(processed_frame)

    # Wait for all frames to be processed
    for processed_frame in processed_frames:
        processed_frame.wait()

    pool.close()
    pool.join()

    # Get the original video's framerate
    framerate = get_video_framerate(input_video_path)

    # Reassemble video from processed frames in the "output" directory
    processed_frame_dir = "output"  # Directory where processed frames are saved
    # Assuming processed frames are saved with filenames like "0000001-crt.png", adjust the pattern accordingly
    ffmpeg_command_assemble = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        os.path.join(processed_frame_dir, "%07d-crt.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "av1_nvenc",
        "-b:v",
        "10M",
        output_video_path.replace(".mkv", "_no_audio.mkv"),
    ]
    subprocess.call(ffmpeg_command_assemble)

    # Add audio back to the processed video
    ffmpeg_command_audio = [
        "ffmpeg",
        "-i",
        output_video_path.replace(".mkv", "_no_audio.mkv"),
        "-i",
        input_video_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        output_video_path,
    ]
    subprocess.call(ffmpeg_command_audio)

    # Cleanup temporary directories and intermediate no-audio video
    dirs_to_delete = [temp_frame_dir, processed_frame_dir]
    file_to_delete = output_video_path.replace(".mkv", "_no_audio.mkv")

    for dir_path in dirs_to_delete:
        shutil.rmtree(dir_path, ignore_errors=True)
        print(f"Deleted directory: {dir_path}")

    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"Deleted file: {file_to_delete}")


def process_input(input_path, **kwargs):
    """
    Determine if input is an image or video and process accordingly.
    """
    if input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        process_image(input_path, **kwargs)
    elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Extract directory, base filename, and extension
        dir_name = os.path.dirname(input_path)
        base_name, ext = os.path.splitext(os.path.basename(input_path))
        # Construct new output path with '_processed' suffix before the extension
        output_video_path = os.path.join(dir_name, f"{base_name}_processed{ext}")
        process_video_file(input_path, output_video_path, **kwargs)
    elif input_path.lower().endswith((".gif")):
        process_gif_file(input_path, **kwargs)
    else:
        print("Unsupported file format.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_input")
    else:
        input_path = sys.argv[1]
        process_input(input_path, external_handler=True)
