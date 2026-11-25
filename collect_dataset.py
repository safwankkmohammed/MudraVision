import cv2
import os

mudra_name = input("Enter Mudra Name: ")

save_path = f"MudraDataset1/{mudra_name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
print("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Captured: {count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    # SPACE = capture image
    if key == 32:
        img_name = f"{count}.jpg"
        cv2.imwrite(os.path.join(save_path, img_name), frame)
        count += 1
        print("Captured:", img_name)

    # ESC = exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset Collection Completed!")
