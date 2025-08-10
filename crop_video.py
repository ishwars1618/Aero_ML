import cv2
import sys
import os

# from chatgpt 4o

SQUARE_SIZE = 325

def crop_upper_right(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get original video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'mov' if needed

    if width < SQUARE_SIZE or height < SQUARE_SIZE:
        print("Error: Video too small to crop 200x200 from upper right.")
        return

    # Set up video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (SQUARE_SIZE, SQUARE_SIZE))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop upper right 200x200 block
        crop = frame[0:SQUARE_SIZE, width-SQUARE_SIZE:width]
        out.write(crop)

    cap.release()
    out.release()
    print(f"Saved cropped video to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python crop_video.py input.mov output.mp4")
    else:
        crop_upper_right(sys.argv[1], sys.argv[2])
