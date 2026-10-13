"""Generate a synthetic vertical-belt test video (no phone needed).
   python scripts/make_test_video.py   ->  test_belt.avi
   Then paste the full path of test_belt.avi into the dashboard's stream URL box."""
import math
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 20, 30
out = cv2.VideoWriter("test_belt.avi", cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
rng = np.random.default_rng(0)
for i in range(FPS * SECONDS):
    t = i / FPS
    drift = int(25 * math.sin(t / 4))                    # belt wanders left/right
    img = np.full((H, W, 3), (110, 120, 125), np.uint8)  # floor
    left, right = W // 2 - 150 + drift, W // 2 + 150 + drift
    img[:, left:right] = (45, 45, 48)                    # belt
    for y in range(-60, H, 60):                          # cleats scrolling down
        yy = (y + int(t * 120)) % (H + 60) - 30
        cv2.line(img, (left, yy), (right, yy), (70, 70, 74), 3)
    cv2.line(img, (left, 0), (left, H), (160, 160, 165), 3)
    cv2.line(img, (right, 0), (right, H), (160, 160, 165), 3)
    if 8 < t < 22:                                       # a crack-like dark scratch + dust specks
        y0 = (int(t * 120) % H)
        cv2.line(img, (left + 80, y0), (left + 140, y0 + 40), (10, 10, 10), 3)
        for _ in range(60):
            cv2.circle(img, (int(rng.integers(left, right)), int(rng.integers(0, H))), 2, (170, 175, 180), -1)
    img = cv2.add(img, rng.integers(0, 8, img.shape, dtype=np.uint8))
    out.write(img)
out.release()
print("wrote test_belt.avi")
