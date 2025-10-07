# import cv2, os

# def flip_images():
# 	gest_folder = "gestures"
# 	images_labels = []
# 	images = []
# 	labels = []
# 	for g_id in os.listdir(gest_folder):
# 		for i in range(1200):
# 			path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
# 			new_path = gest_folder+"/"+g_id+"/"+str(i+1+1200)+".jpg"
# 			print(path)
# 			img = cv2.imread(path, 0)
# 			img = cv2.flip(img, 1)
# 			cv2.imwrite(new_path, img)

# flip_images()
import cv2
import numpy as np
import os

def rotate_image(image, angle):
    """Rotate an image around its center without cropping."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

def augment_gesture_images():
    gestures_path = "gestures"

    if not os.path.exists(gestures_path):
        print("❌ 'gestures' folder not found. Please run create_gestures.py first.")
        return

    gesture_dirs = [os.path.join(gestures_path, d) for d in os.listdir(gestures_path)
                    if os.path.isdir(os.path.join(gestures_path, d))]

    print(f"📂 Found {len(gesture_dirs)} gesture folders.")
    print("🔄 Starting augmentation (rotation + flipping)...")

    for folder in gesture_dirs:
        images = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
        count = len(images)
        print(f"🖐 Gesture ID {os.path.basename(folder)}: {count} images")

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Flip horizontally
            flipped = cv2.flip(img, 1)
            new_name = os.path.splitext(img_name)[0] + "_flipped.jpg"
            cv2.imwrite(os.path.join(folder, new_name), flipped)

            # Rotate at different angles
            for angle in [-15, -10, -5, 5, 10, 15]:
                rotated = rotate_image(img, angle)
                new_name = os.path.splitext(img_name)[0] + f"_rot{angle}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), rotated)

        print(f"✅ Augmentation done for gesture {os.path.basename(folder)}")

    print("\n🎉 All gestures augmented successfully!")

if __name__ == "__main__":
    augment_gesture_images()
