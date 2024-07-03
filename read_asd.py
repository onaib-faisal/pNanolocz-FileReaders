import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def update_frame(frame_number):
    frame_image.set_data(im[:, :, frame_number])
    return frame_image,

def open_asd(file_path: str):
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

        # Debug print for header values
        # print("Header:", header)

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

            # Debug print for frame size and data
            # print(f"Frame {k}: Expected size = {frame_size}, Read size = {sub.size}")

            if sub.size != frame_size:
                print(f"Frame {k}: Skipping incomplete frame. Expected {frame_size}, got {sub.size}")
                continue

            preIm[:, :, valid_frames] = sub.reshape((header['yPixel'], header['xPixel'])).T
            valid_frames += 1

        # Adjust the preIm array to include only valid frames
        preIm = preIm[:, :, :valid_frames]

        im = -preIm / 205 * header['zExtCoef']
        im = np.flip(im, axis=0)
    
    return im, header

if __name__ == "__main__":
    file_path = 'data/070920180003.asd'
    im, header = open_asd(file_path)

    # Do something with the image (im) and header
    print("Header: ", header)
    print("Image shape: ", im.shape)

    if im.ndim == 2:
        # Single frame case
        plt.imshow(im, cmap='gray')
        plt.colorbar(label='Height (nm)')
        plt.show()
    elif im.ndim == 3:
        # Multi-frame case
        fig, ax = plt.subplots()
        frame_image = ax.imshow(im[:, :, 0], cmap='gray')

        # Create the animation
        ani = animation.FuncAnimation(fig, update_frame, frames=im.shape[2], interval=50, blit=True)

        # Display the animation
        plt.show()
    else:
        print("Unsupported image dimensions.")
