import cv2
import time
from picamera2 import Picamera2

def main():
    # Initialize camera
    picam2 = Picamera2()

    # A safe default preview config (works well on Pi)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Give camera a moment to warm up
    time.sleep(0.5)

    # OpenCV QR detector
    qr = cv2.QRCodeDetector()

    print("QR Detector running. Press 'q' to quit.")

    while True:
        # Capture RGB frame from PiCamera2
        frame = picam2.capture_array()  # RGB888 (H, W, 3)

        # OpenCV typically works fine with RGB here, but we can convert to BGR for display consistency
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect and decode
        data, points, _ = qr.detectAndDecode(frame_bgr)

        if points is not None and len(points) > 0:
            pts = points.astype(int).reshape(-1, 2)

            # Draw the QR boundary
            for i in range(len(pts)):
                cv2.line(frame_bgr, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

            # If decoded successfully, show data
            if data:
                cv2.putText(frame_bgr, data, (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print("QR:", data)

        cv2.imshow("QR Scanner", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
