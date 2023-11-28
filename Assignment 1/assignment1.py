import cv2 as cv
import numpy as np
import time

# Find the brightest pixel
def find_brightest_pixel(frame):
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the position of the brightest pixel
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(gray)
    return maxLoc

# Find the reddest pixel
def find_reddest_pixel(frame):
    # Split into color channels
    B, G, R = cv.split(frame)
    # Ensure red is dominant and above a threshold (e.g., 100 out of 255)
    red_dominant = (R > G) & (R > B) & (R > 100)
    # Calculate a more nuanced measure of redness
    redness = R - (G + B) / 2
    redness = redness * red_dominant  # Apply the red dominant mask
    # Find the position of the reddest pixel
    if np.any(redness):
        y, x = np.unravel_index(np.argmax(redness), redness.shape)
    else:
        x, y = -1, -1  # No sufficiently red pixel found
    return x, y

# Finding reddest and brightest pixel with a double for loop, iterating through each pixel
def find_reddest_and_brightest_pixel(frame):
    rows, cols, _ = frame.shape
    max_brightness = -1
    max_redness = -1
    brightest_pixel_coords = (-1, -1)
    reddest_pixel_coords = (-1, -1)
    for i in range(rows):
        for j in range(cols):
            B, G, R = frame[i, j]
            brightness = int(B) + int(G) + int(R)  # Sum of RGB components
            redness = int(R) - max(int(G), int(B))  # Redness calculation
            # Check for brightest pixel
            if brightness > max_brightness:
                max_brightness = brightness
                brightest_pixel_coords = (j, i)
            # Check for reddest pixel
            if redness > max_redness and R > 100:  # Additional check for red intensity
                max_redness = redness
                reddest_pixel_coords = (j, i)
    return reddest_pixel_coords, brightest_pixel_coords

# MacBook Camera
# cap = cv.VideoCapture(0)
# iPhone Camera
cap = cv.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize tick count and tick frequency
tick_freq = cv.getTickFrequency()
prev_ticks = cv.getTickCount()

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    # Start measuring processing time
    start_time = time.time()

    ## Brightest and reddest pixel functions with OpenCV functions
    # Find the brightest pixel
    # x_bright, y_bright = find_brightest_pixel(frame)
    # # Mark the brightest pixel with a purple circle
    # cv.circle(frame, (x_bright, y_bright), 10, (255, 0, 255), 2)
    # Find the reddest pixel
    # x_red, y_red = find_reddest_pixel(frame)
    # # Mark the reddest pixel with a red circle
    # if x_red != -1 and y_red != -1:
    #     cv.circle(frame, (x_red, y_red), 10, (0, 0, 255), 2)

    ## Find the reddest and brightest pixels with double for loop
    (x_red, y_red), (x_bright, y_bright) = find_reddest_and_brightest_pixel(frame)

    # Mark the reddest pixel with a red circle
    if x_red != -1 and y_red != -1:
        cv.circle(frame, (x_red, y_red), 10, (0, 0, 255), 2)

    # Mark the brightest pixel with a purple circle
    if x_bright != -1 and y_bright != -1:
        cv.circle(frame, (x_bright, y_bright), 10, (255, 0, 255), 2)

    # Calculate processing time in milliseconds
    processing_time = (time.time() - start_time) * 1000
    print(f'Processing Time: {processing_time:.10f}ms')

    # Calculate FPS
    current_ticks = cv.getTickCount()
    time_elapsed = (current_ticks - prev_ticks) / tick_freq
    prev_ticks = current_ticks
    fps = (1.0 / time_elapsed)
    fps_text = f'FPS: {fps:.5f}'

    # Display FPS on the frame
    cv.putText(frame, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display the frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()