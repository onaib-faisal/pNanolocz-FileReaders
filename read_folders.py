import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from read_nhf import open_nhf
from read_jpk import open_jpk
from read_ibw import open_ibw
from read_spm import open_spm
from read_gwy import open_gwy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_FILE_TYPES = {'.nhf', '.jpk', '.ibw', '.spm', '.gwy'}

def check_folder(folder_path):
    file_list = list(Path(folder_path).rglob('*'))
    file_format_count = {}

    for file_path in file_list:
        if file_path.is_file():
            ext = file_path.suffix
            if ext in EXPECTED_FILE_TYPES:
                file_format_count[ext] = file_format_count.get(ext, 0) + 1

    dominant_format = max(file_format_count, key=file_format_count.get)
    dominant_format_files = [str(file_path) for file_path in file_list if file_path.suffix == dominant_format]

    return dominant_format, dominant_format_files

def load_images(file_paths, dominant_format):
    images = []
    metadata = None

    for file_path in file_paths:
        if dominant_format == '.nhf':
            im, meta = open_nhf(file_path, 'Topography')
        elif dominant_format == '.jpk':
            im, meta = open_jpk(file_path, 1)
        elif dominant_format == '.ibw':
            im, meta = open_ibw(file_path, 1)
        elif dominant_format == '.spm':
            im, meta = open_spm(file_path, 1)
        elif dominant_format == '.gwy':
            im, meta = open_gwy(file_path, 1)
        images.append(im)
        metadata = meta  # Assume metadata is same for all images

    return images, metadata

def play_images(images):
    fig, ax = plt.subplots()
    im = ax.imshow(images[0], animated=True)

    def updatefig(i):
        im.set_array(images[i])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=len(images), interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    folder_path = 'data/'  # Replace with folder path
    dominant_format, file_paths = check_folder(folder_path)
    logger.info(f"Dominant format: {dominant_format}")
    logger.info(f"Number of files: {len(file_paths)}")

    images, metadata = load_images(file_paths, dominant_format)
    play_images(images)
