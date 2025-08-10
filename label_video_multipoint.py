import cv2
import csv

# Store all labeled points: list of tuples (frame_num, x, y)
all_points = []
# Temporary points for current frame
current_points = []
current_frame_num = 0

def click_event(event, x, y, flags, param):
    global current_points, current_frame_num
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Frame {current_frame_num}: Clicked at ({x}, {y})")
        current_points.append((current_frame_num, x, y))

def label_video(video_path, output_csv):
    global current_frame_num, current_points, all_points

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", click_event)

    while 0 <= current_frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        while True:
            display = frame.copy()

            # Show existing points on this frame
            for (_, x, y) in [pt for pt in current_points if pt[0] == current_frame_num]:
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            cv2.imshow("Video", display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):
                # Save points for this frame
                all_points.extend(current_points)
                current_points = []
                current_frame_num += 1
                break
            elif key == ord('z'):
                # Undo last point on this frame
                for i in range(len(current_points) - 1, -1, -1):
                    if current_points[i][0] == current_frame_num:
                        print(f"Removed point: {current_points[i]}")
                        current_points.pop(i)
                        break
            elif key == ord('j'):
                try:
                    frame_input = input(f"Jump to frame (0 to {total_frames - 1}): ")
                    target_frame = int(frame_input)
                    if 0 <= target_frame < total_frames:
                        all_points.extend(current_points)
                        current_points = []
                        current_frame_num = target_frame
                        break
                    else:
                        print("Invalid frame number.")
                except ValueError:
                    print("Please enter a valid integer.")
            elif key == ord('q'):
                print("Quitting early.")
                cap.release()
                cv2.destroyAllWindows()
                write_csv(output_csv, all_points + current_points)
                return

    cap.release()
    cv2.destroyAllWindows()
    write_csv(output_csv, all_points)
    print(f"Saved labeled points to {output_csv}")

def write_csv(path, data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'x', 'y'])
        writer.writerows(data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python label_video_multipoint.py input.mov")
    else:
        label_video(sys.argv[1], "labeled_points_080525.csv")
