import cv2
import csv
import os
import sys
import random

def export_frames_with_annotations(video_path, csv_path, output_dir, key_filename="color_key.txt"):
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    annotations = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # header row
        coord_headers = headers[1:]  # skip frame_number

        # Create point names (one per x,y pair)
        point_names = []
        for i in range(0, len(coord_headers), 2):
            point_label = coord_headers[i].replace("x", "P")  # e.g., x1 -> P1
            point_names.append(point_label)

        # Assign fixed colors for each point pair
        colors = {}
        random.seed(42)  # reproducible colors
        for pname in point_names:
            colors[pname] = tuple(random.randint(50, 255) for _ in range(3))

        # Load annotations into dict
        for row in reader:
            if not row or not row[0].isdigit():
                continue
            frame_num = int(row[0])
            coords = row[1:]
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords) and coords[i] and coords[i+1]:
                    try:
                        x = int(float(coords[i]))
                        y = int(float(coords[i+1]))
                        pname = point_names[i // 2]
                        color = colors[pname]
                        points.append((x, y, color))
                    except ValueError:
                        pass
            annotations[frame_num] = points

    # Save color key file
    with open(os.path.join(output_dir, key_filename), 'w') as kf:
        for pname, color in colors.items():
            kf.write(f"{pname}: RGB{color}\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames. Exporting {len(annotations)} frames...")

    for frame_num, points in annotations.items():
        if frame_num >= total_frames:
            print(f"Skipping frame {frame_num} (out of range)")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num}")
            continue

        # Save original image
        orig_path = os.path.join(output_dir, f"frame_{frame_num}.png")
        cv2.imwrite(orig_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Draw annotations
        annotated = frame.copy()
        for (x, y, color) in points:
            cv2.circle(annotated, (x, y), 5, color, -1)

        # Save annotated image
        annt_path = os.path.join(output_dir, f"frame_{frame_num}_annt.png")
        cv2.imwrite(annt_path, annotated, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Saved: {orig_path} and {annt_path}")

    cap.release()
    print("Done. Color key saved to", os.path.join(output_dir, key_filename))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python export_frames_with_annotations.py input.mov annotations.csv output_dir")
    else:
        export_frames_with_annotations(sys.argv[1], sys.argv[2], sys.argv[3])
