import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from numpy import roll

AFM = np.load('AFM_cmap.npy')
AFM = colors.ListedColormap(AFM)

def read_int(file, dtype='int32', count=1):
    return np.fromfile(file, dtype=np.dtype(dtype).newbyteorder('<'), count=count)[0]

def read_float(file, count=1):
    return np.fromfile(file, dtype=np.dtype('float32').newbyteorder('<'), count=count)[0]

def read_bool(file, count=1):
    return np.fromfile(file, dtype=np.dtype('bool').newbyteorder('<'), count=count)[0]

def read_char(file, count=1):
    raw_bytes = np.fromfile(file, dtype=np.dtype('S{}'.format(count)).newbyteorder('<'), count=count)[0]
    return raw_bytes.decode('utf-8', errors='replace')

def skip_bytes(file, count):
    file.seek(count, 1)

def update_frame(frame_number, shift, im_rotated):
    shifted_frame = roll(im_rotated[:, :, frame_number], shift, axis=1)
    frame_image.set_data(shifted_frame)
    return frame_image,

def open_asd(file_path: str, channel: int = 1):
    header = {}

    with open(file_path, 'rb') as asd_file:
        header['fileVersion'] = read_int(asd_file, 'int32')
        header['fileHeaderSize'] = read_int(asd_file, 'int32')
        header['frameHeaderSize'] = read_int(asd_file, 'int32')
        header['encNumber'] = read_int(asd_file, 'int32')
        header['operationNameSize'] = read_int(asd_file, 'int32')
        header['commentSize'] = read_int(asd_file, 'int32')
        header['dataTypeCh1'] = read_int(asd_file, 'int32')
        header['dataTypeCh2'] = read_int(asd_file, 'int32')
        header['numberFramesRecorded'] = read_int(asd_file, 'int32')
        header['numberFramesCurrent'] = read_int(asd_file, 'int32')
        header['scanDirection'] = read_int(asd_file, 'int32')
        header['fileName'] = read_int(asd_file, 'int32')
        header['xPixel'] = read_int(asd_file, 'int32')
        header['yPixel'] = read_int(asd_file, 'int32')
        header['xScanRange'] = read_int(asd_file, 'int32')
        header['yScanRange'] = read_int(asd_file, 'int32')
        header['avgFlag'] = read_bool(asd_file)
        header['avgNumber'] = read_int(asd_file, 'int32')
        header['yearRec'] = read_int(asd_file, 'int32')
        header['monthRec'] = read_int(asd_file, 'int32')
        header['dayRec'] = read_int(asd_file, 'int32')
        header['hourRec'] = read_int(asd_file, 'int32')
        header['minuteRec'] = read_int(asd_file, 'int32')
        header['secondRec'] = read_int(asd_file, 'int32')
        header['xRoundDeg'] = read_int(asd_file, 'int32')
        header['yRoundDeg'] = read_int(asd_file, 'int32')
        header['frameAcqTime'] = read_float(asd_file)
        header['sensorSens'] = read_float(asd_file)
        header['phaseSens'] = read_float(asd_file)
        skip_bytes(asd_file, 12)  # Skip 12 bytes
        header['machineNum'] = read_int(asd_file, 'int32')
        header['adRange'] = read_int(asd_file, 'int32')
        header['adRes'] = read_int(asd_file, 'int32')
        header['xMaxScanRange'] = read_float(asd_file)
        header['yMaxScanRange'] = read_float(asd_file)
        header['xExtCoef'] = read_float(asd_file)
        header['yExtCoef'] = read_float(asd_file)
        header['zExtCoef'] = read_float(asd_file)
        header['zDriveGain'] = read_float(asd_file)
        header['operName'] = read_char(asd_file, header['operationNameSize'])
        # header['comment'] = read_char(asd_file, header['commentSize'])

        # Determine which data type to read based on the selected channel
        data_type = header['dataTypeCh1'] if channel == 1 else header['dataTypeCh2']

        # Reading frame data
        valid_frames = 0
        preIm = np.zeros((header['yPixel'], header['xPixel'], header['numberFramesCurrent']), dtype=np.float32)
        for k in range(header['numberFramesCurrent']):
            frame_header = {}
            frame_header['frameNumber'] = read_int(asd_file, 'int32')
            frame_header['frameMaxData'] = read_int(asd_file, 'int16')
            frame_header['frameMinData'] = read_int(asd_file, 'int16')
            frame_header['xOffset'] = read_int(asd_file, 'int16')
            frame_header['dataType'] = read_int(asd_file, 'int16')
            frame_header['xTilt'] = read_float(asd_file)
            frame_header['yTilt'] = read_float(asd_file)
            frame_header['flagLaserIr'] = read_bool(asd_file)
            skip_bytes(asd_file, header['frameHeaderSize'] - 21)

            # Read frame data
            frame_size = header['xPixel'] * header['yPixel']
            sub = np.fromfile(asd_file, dtype=np.dtype('int16').newbyteorder('<'), count=frame_size)

            if sub.size != frame_size:
                print(f"Frame {k}: Skipping incomplete frame. Expected {frame_size}, got {sub.size}")
                continue

            preIm[:, :, valid_frames] = sub.reshape((header['yPixel'], header['xPixel'])).T
            valid_frames += 1

        # Adjust the preIm array to include only valid frames
        preIm = preIm[:, :, :valid_frames]

        im = -preIm / 205 * header['zExtCoef']
        im = np.flip(im, axis=0)
        
        # Rotate the image 90 degrees clockwise
        im_rotated = np.rot90(im, k=-1, axes=(0, 1))
    
    return im_rotated, header

if __name__ == "__main__":
    file_path = 'data/070920180003.asd'
    channel = 2  # Set the channel to 1 or 2
    im_rotated, header = open_asd(file_path, channel)

    # Do something with the rotated image (im_rotated) and header
    print("Header: ", header)
    print("Rotated Image shape: ", im_rotated.shape)

    shift = 207  # Number of pixels to shift the image

    if im_rotated.ndim == 2:
        # Single frame case
        shifted_image = roll(im_rotated, shift, axis=1)
        plt.imshow(shifted_image, cmap=AFM)
        plt.colorbar(label='Height (nm)')
        plt.show()
    elif im_rotated.ndim == 3:
        # Multi-frame case
        fig, ax = plt.subplots()
        frame_image = ax.imshow(im_rotated[:, :, 0], cmap=AFM)

        # Create the animation
        ani = animation.FuncAnimation(fig, update_frame, frames=im_rotated.shape[2], fargs=(shift, im_rotated), interval=50, blit=True)

        # Display the animation
        plt.show()
    else:
        print("Unsupported image dimensions.")
