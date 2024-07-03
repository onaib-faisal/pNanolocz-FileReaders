from __future__ import annotations
from pathlib import Path
import numpy as np
from igor2 import binarywave
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def _ibw_pixel_to_nm_scaling(scan: dict) -> float:
    """
    Extract pixel to nm scaling from the IBW image metadata.

    Parameters
    ----------
    scan : dict
        The loaded binary wave object.

    Returns
    -------
    float
        A value corresponding to the real length of a single pixel.
    """
    notes = {}
    for line in str(scan["wave"]["note"]).split("\\r"):
        if ":" in line:
            key, val = line.split(":", 1)
            notes[key.strip()] = val.strip()
    return (
        float(notes["SlowScanSize"]) / scan["wave"]["wData"].shape[0] * 1e9,  # Convert to nm
        float(notes["FastScanSize"]) / scan["wave"]["wData"].shape[1] * 1e9,  # Convert to nm
    )[0]

def extract_metadata(notes: str) -> dict:
    """
    Extract metadata from the IBW notes.

    Parameters
    ----------
    notes : str
        The notes string from the IBW file.

    Returns
    -------
    dict
        A dictionary containing extracted metadata.
    """
    metadata = {}
    for line in notes.split("\\r"):
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip()] = val.strip()
    return metadata

def load_ibw(file_path: Path | str, channel: str) -> tuple[np.ndarray, float, dict]:
    """
    Load image from Asylum Research (Igor) .ibw files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .ibw file.
    channel : str
        The channel to extract from the .ibw file.

    Returns
    -------
    tuple[np.ndarray, float, dict]
        A tuple containing the image, its pixel to nanometre scaling value, and metadata.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the channel is not found in the .ibw file.
    """
    file_path = Path(file_path)
    scan = binarywave.load(file_path)
    labels = []
    for label_list in scan["wave"]["labels"]:
        for label in label_list:
            if label:
                labels.append(label.decode())
    if channel not in labels:
        raise ValueError(f"Channel '{channel}' not found in {file_path}. Available channels: {labels}")
    
    channel_idx = labels.index(channel)
    image = scan["wave"]["wData"][:, :, channel_idx].T * 1e9  # Convert to nm
    image = np.flipud(image)
    scaling = _ibw_pixel_to_nm_scaling(scan)
    metadata = extract_metadata(str(scan["wave"]["note"]))

    return image, scaling, metadata

def update_frame(frame_number, im, frame_image):
    frame_image.set_data(im[:, :, frame_number])
    return frame_image,

if __name__ == "__main__":
    file_path = 'data/tops70s14_190g0000.ibw'
    channel = 'HeightTracee'  # Replace with the appropriate channel name
    try:
        im, scaling_factor, metadata = load_ibw(file_path, channel)
        print(f"Scaling factor: {scaling_factor} nm/pixel")
        print(f"Image shape: {im.shape}")
        print("Metadata:", metadata)

        # Check if the image is 2D or 3D
        if im.ndim == 2:
            # Single frame case
            plt.imshow(im, cmap='gray')
            plt.colorbar(label='Height (nm)')
            plt.show()
        elif im.ndim == 3:
            # Multi-frame case
            fig, ax = plt.subplots()
            frame_image = ax.imshow(im[:, :, 0], cmap=metadata['ColorMap 1'])

            # Create the animation
            ani = animation.FuncAnimation(fig, update_frame, frames=im.shape[2], fargs=(im, frame_image), interval=50, blit=True)

            # Display the animation
            plt.show()
        else:
            print("Unsupported image dimensions.")

    except Exception as e:
        print(f"Error: {e}")
