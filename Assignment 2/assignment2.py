import cv2
import numpy as np
import random
import time

# Calculate the distance of a point from a line.
def distance_from_line(point, line, precomputed_denom):
    a, b, c = line
    x0, y0 = point
    return abs(a * x0 + b * y0 + c) / precomputed_denom

# Calculate line coefficients a, b, c for the line passing through points p1 and p2
def line_from_points(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    return a, b, c

# Draw a line on an image given its coefficients a, b, c (ax + by + c = 0)
def draw_line_from_coefficients(img, line):
    a, b, c = line
    cols, rows = img.shape[:2]
    if a != 0 and b != 0:
        lefty = int((-c - a * 0) / b)
        righty = int((-c - a * cols) / b)
        cv2.line(img, (0, lefty), (cols, righty), (0, 0, 255), 2)
    elif b == 0:  # Vertical line
        x = int(-c / a)
        cv2.line(img, (x, 0), (x, rows), (0, 0, 255), 2)
    elif a == 0:  # Horizontal line
        y = int(-c / b)
        cv2.line(img, (0, y), (cols, y), (0, 0, 255), 2)

# Custom RANSAC implementation for line fitting.
def custom_ransac(edges, num_iterations, inlier_threshold, early_termination_threshold):
    max_inliers = []
    best_line = None
    for _ in range(num_iterations):
        p1, p2 = random.sample(list(edges), 2)
        line = line_from_points(p1, p2)
        precomputed_denom = np.sqrt(line[0] ** 2 + line[1] ** 2)
        inliers = [point for point in edges if distance_from_line(point, line, precomputed_denom) < inlier_threshold]
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_line = line
        if len(max_inliers) > early_termination_threshold:
            break
    return best_line

# Main loop for capturing and processing frames
cap = cv2.VideoCapture(0)
# Define the new size
new_width = 640
new_height = 480

# Initialize tick count and tick frequency
tick_freq = cv2.getTickFrequency()
prev_ticks = cv2.getTickCount()

# Define RANSAC sampling parameter
k = 10  # Only consider every k-th point for RANSAC

# Define Canny parameters
lower_threshold = 110  # Lower threshold for Canny edge detection
upper_threshold = 330  # Upper threshold for Canny edge detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start measuring processing time
    start_time = time.time()

    # Resize and optionally preprocess the frame
    frame_resized = cv2.resize(frame, (new_width, new_height))  
    frame_preprocessed = cv2.GaussianBlur(frame_resized, (3, 3), 1)

    # Adjusted Canny Edge Detector
    edges = cv2.Canny(frame_preprocessed, lower_threshold, upper_threshold)

    # Get edge points and sample every k-th point
    y_coords, x_coords = np.where(edges >= 255)
    edge_points = list(zip(x_coords, y_coords))[::k]  # Sample every k-th point

    if len(edge_points) > 10:  # Ensure enough points to apply RANSAC
        best_line = custom_ransac(edge_points, num_iterations=300, inlier_threshold=0.25, early_termination_threshold=100)
        if best_line is not None:
            draw_line_from_coefficients(frame_resized, best_line)


    # Calculate FPS
    current_ticks = cv2.getTickCount()
    time_elapsed = (current_ticks - prev_ticks) / tick_freq
    prev_ticks = current_ticks
    fps = (1.0 / time_elapsed)
    fps_text = f'FPS: {fps:.5f}'

    # Display FPS on the frame
    cv2.putText(frame_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame with Line', frame_resized)
    cv2.imshow('Edges', edges)

    # Calculate processing time in milliseconds
    processing_time = (time.time() - start_time) * 1000
    print(f'Processing Time: {processing_time:.10f}ms')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()