import cv2
import numpy as np

# Loading  the Haar Cascade for license plate detection
Harcascade = "model/haarcascades/haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)  # Using  0 for default camera
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500  # Minimum area of the number plate to be considered

# Function to get the dominant color (refined)
def get_dominant_color(image):
    # Converting to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for Green,  White, Blue
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    

    # Creating masks for each color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    

    # Calculate the percentage of each color in the image
    green_percentage = np.sum(mask_green) / (image.shape[0] * image.shape[1] * 255)
    white_percentage = np.sum(mask_white) / (image.shape[0] * image.shape[1] * 255)
    blue_percentage = np.sum(mask_blue) / (image.shape[0] * image.shape[1] * 255)
    
    # Classify based on percentages 
    if green_percentage > 0.2:
        return "EV"
    elif white_percentage > 0.3:
        return "Private"
    elif blue_percentage > 0.1:
        return "Embassy"
    else:
        return "None"

while True:
    success, img = cap.read()

    # Check if frame is successfully read
    if not success:
        print("Ignoring empty camera frame.")
        continue

    plate_cascade = cv2.CascadeClassifier(Harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw rectangle around the license plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the license plate region
            img_interest = img[y : y + h, x : x + w]

            # Get dominant color and classify vehicle type
            vehicle_type = get_dominant_color(img_interest)

            # Display vehicle type when 't' is pressed
            if cv2.waitKey(1) & 0xFF == ord('t'):
                print("Vehicle Type:", vehicle_type)
                cv2.putText(img, vehicle_type, (x, y - 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Save image when 's' is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("plates/scanned_plate.jpg", img_interest)
                print("License plate image saved!")

            # Display saved image when 'd' is pressed
            if cv2.waitKey(1) & 0xFF == ord('d'):
                saved_img = cv2.imread("plates/scanned_plate.jpg")
                if saved_img is not None:
                    cv2.imshow("Saved Plate", saved_img)
                else:
                    print("Saved image not found!")

    # Display the main frame
    cv2.imshow("Camera Feed", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the camera and close windows
cap.release()
cv2.destroyAllWindows()