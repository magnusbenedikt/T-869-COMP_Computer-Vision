import cv2
import numpy as np
import time

# Vectorized line length calculation
def length_of_line(line):
    p1, p2 = np.array(line[0][:2]), np.array(line[0][2:])
    return np.linalg.norm(p2 - p1)

# Convert endpoints of a line segment into a line in vector form.
def points_2_line(line):
    return np.cross([line[0], line[1], 1], [line[2], line[3], 1])

# Find intersection points of given lines within the frame boundaries.
def find_corners(lines, shape_frame):
    # Convert line endpoints to line vectors outside the loop
    line_vectors = [points_2_line(line[0]) for line in lines]
    intersects = []
    # Loop through pairs of line vectors and compute intersection
    for i, l1 in enumerate(line_vectors):
        for l2 in line_vectors[i + 1:]:
            x12 = np.cross(l1, l2)
            # Check if intersection is valid and within frame boundaries
            if x12[2] != 0:
                x, y = x12[0] / x12[2], x12[1] / x12[2]
                if 0 <= x < shape_frame[1] and 0 <= y < shape_frame[0]:
                    intersects.append([int(x), int(y)])
    return intersects

# Vectorized line length calculation for all lines
def find_longest_lines(lines, n=4):
    lengths = np.array([length_of_line(line) for line in lines])
    return [lines[i] for i in np.argsort(lengths)[-n:]]

# Order corner points in a clockwise manner starting from the top-left.
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Video capture setup
cap = cv2.VideoCapture(1)
new_width, new_height = 640, 480

# Setup for FPS calculation
tick_freq = cv2.getTickFrequency()
prev_ticks = cv2.getTickCount()

# Main loop for frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame preprocessing
    start_time = time.time()
    frame_resized = cv2.resize(frame, (new_width, new_height))
    frame_preprocessed = cv2.medianBlur(frame_resized, 9)
    edges = cv2.Canny(frame_preprocessed, 50, 150)

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=50)

    # Process detected lines
    if lines is not None:
        longest_lines = find_longest_lines(lines)
        intersections = find_corners(longest_lines, frame_resized.shape)

        # Perform perspective transformation if enough corners are found
        if len(intersections) >= 4:
            corner_points = np.array(intersections, dtype="float32")
            ordered_points = order_points(corner_points)
            dst_pts = np.array([[0, 0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(ordered_points, dst_pts)
            warped = cv2.warpPerspective(frame_resized, M, (new_width, new_height))
            cv2.imshow('Warped Image', warped)

        # Draw the longest lines and corners on the resized frame
        for line in longest_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for point in intersections:
            cv2.circle(frame_resized, (point[0], point[1]), 5, (255, 0, 0), -1)

    # FPS calculation and display
    current_ticks = cv2.getTickCount()
    time_elapsed = (current_ticks - prev_ticks) / tick_freq
    prev_ticks = current_ticks
    fps = 1.0 / time_elapsed
    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(frame_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_time = (time.time() - start_time) * 1000
    print(f'Processing Time: {processing_time:.2f}ms')

    # Display processed frames
    cv2.imshow('Frame with Lines', frame_resized)
    cv2.imshow('Edges with Lines and Corners', edges)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
