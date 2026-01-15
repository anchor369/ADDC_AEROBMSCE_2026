import cv2
import numpy as np
from pymavlink import mavutil
from pyzbar.pyzbar import decode
from picamera2 import Picamera2, Preview
import time

# --- Configuration ---
# Set up the camera resolution. A higher resolution might help with detection at a distance.
CAMERA_RESOLUTION = (640, 480)
# Font settings for displaying text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0) # Green color for text
RECT_COLOR = (0, 0, 255) # Red color for rectangle

PORT = '/dev/ttyAMA0'
BAUD = 115200

def send_statustext(master, message, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
    """Send a statustext message that appears in Mission Planner"""
    # Ensure message is properly formatted (max 50 chars for MAVLink 1.0)
    msg_bytes = message.encode('utf-8')[:50]
    master.mav.statustext_send(severity, msg_bytes)
    print(f"Sent [{severity}]: {message}")

def main():
    print("Connecting to FC...")
    master = mavutil.mavlink_connection(PORT, baud=BAUD)

    # Wait for heartbeat
    master.wait_heartbeat()
    print(f"Heartbeat received! System {master.target_system}, Component {master.target_component}")
    
    # Set source system and component IDs
    # Use the FC's system ID so Mission Planner recognizes it
    master.mav.srcSystem = master.target_system
    master.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
    
    print(f"Source set to System {master.mav.srcSystem}, Component {master.mav.srcComponent}")
    
    print("[INFO] Starting camera initialization...")
    picam2 = Picamera2()
    # Configure the camera with a main output for processing
    config = picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION, "format": "RGB888"})
    picam2.configure(config)
    
    # Start the camera and give it a moment to warm up
    picam2.start()
    time.sleep(1.0)
    print("[INFO] Camera ready. Press 'q' to quit.")
    count = 0
    while True:
        # Grab a frame from the camera as a NumPy array
        # This is the "grab frame" step using picamera2
        frame = picam2.capture_array("main")
        # 'picamera2' provides the frame in RGB format, which is compatible with pyzbar and cv2
        
        # Find and decode QR codes in the frame
        decoded_objects = decode(frame)

        # Process detected QR codes
        for obj in decoded_objects:
            # Extract the data and type
            data = obj.data.decode('utf-8')
            obj_type = obj.type
            print(f"Detected {obj_type}: {data}")

            # Draw a bounding box around the QR code using OpenCV
            (x, y, w, h) = obj.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, FONT_THICKNESS)
            
            # Put the decoded text on the image
            text = f"{data}"
            cv2.putText(frame, text, (x, y - 10), FONT, FONT_SCALE * 0.5, TEXT_COLOR, FONT_THICKNESS)
            if count >= 24:
                send_statustext(master, text, mavutil.mavlink.MAV_SEVERITY_WARNING)
                count = 0
            else:
                count = count + 1
            
        # Display the frame with detections (optional, for visualization)
        cv2.imshow("QR Code Scanner", frame)

        # Check for key press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    print("[INFO] Program terminated and resources released.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
