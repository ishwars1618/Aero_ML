import csv
import sys
from collections import OrderedDict

def group_csv(input_csv, output_csv):
    grouped_data = OrderedDict()  # preserves insertion order

    # Read input CSV
    with open(input_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].isdigit():
                continue  # skip empty lines or headers
            frame = int(row[0])
            x, y = row[1], row[2]
            if frame not in grouped_data:
                grouped_data[frame] = []
            grouped_data[frame].extend([x, y])

    # Write output CSV preserving order
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for frame, coords in grouped_data.items():
            writer.writerow([frame] + coords)

    print(f"Grouped CSV saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python group_points_by_frame_preserve.py input.csv output.csv")
    else:
        group_csv(sys.argv[1], sys.argv[2])
