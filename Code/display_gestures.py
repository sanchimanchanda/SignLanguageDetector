# import cv2, os, random
# import numpy as np

# def get_image_size():
# 	img = cv2.imread('gestures/0/100.jpg', 0)
# 	return img.shape

# gestures = os.listdir('gestures/')
# gestures.sort(key = int)
# begin_index = 0
# end_index = 5
# image_x, image_y = get_image_size()

# if len(gestures)%5 != 0:
# 	rows = int(len(gestures)/5)+1
# else:
# 	rows = int(len(gestures)/5)

# full_img = None
# for i in range(rows):
# 	col_img = None
# 	for j in range(begin_index, end_index):
# 		img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
# 		img = cv2.imread(img_path, 0)
# 		if np.any(img == None):
# 			img = np.zeros((image_y, image_x), dtype = np.uint8)
# 		if np.any(col_img == None):
# 			col_img = img
# 		else:
# 			col_img = np.hstack((col_img, img))

# 	begin_index += 5
# 	end_index += 5
# 	if np.any(full_img == None):
# 		full_img = col_img
# 	else:
# 		full_img = np.vstack((full_img, col_img))


# cv2.imshow("gestures", full_img)
# cv2.imwrite('full_img.jpg', full_img)
# cv2.waitKey(0)











import cv2
import os
import random
import numpy as np

def get_image_size():
    # Get size of any sample image
    img = cv2.imread('gestures/1/1.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Sample image not found in gestures/0/. Make sure gesture images are created.")
    return img.shape


def main():
    gestures = os.listdir('gestures/')
    gestures = [g for g in gestures if g.isdigit()]  # only numeric gesture folders
    gestures.sort(key=int)

    begin_index = 0
    end_index = 5
    image_y, image_x = get_image_size()

    if len(gestures) % 5 != 0:
        rows = int(len(gestures) / 5) + 1
    else:
        rows = int(len(gestures) / 5)

    full_img = None

    for i in range(rows):
        col_img = None
        for j in range(begin_index, min(end_index, len(gestures))):
            gesture_dir = f"gestures/{gestures[j]}"
            images = os.listdir(gesture_dir)
            if not images:
                print(f"⚠️ No images in {gesture_dir}, skipping.")
                continue

            random_img = random.choice(images)
            img_path = os.path.join(gesture_dir, random_img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                img = np.zeros((image_y, image_x), dtype=np.uint8)

            if col_img is None:
                col_img = img
            else:
                col_img = np.hstack((col_img, img))

        begin_index += 5
        end_index += 5

        if col_img is None:
            continue

        if full_img is None:
            full_img = col_img
        else:
            full_img = np.vstack((full_img, col_img))

    if full_img is None:
        print("❌ No gestures found to display.")
        return

    cv2.imshow("All Gestures", full_img)
    cv2.imwrite('full_img.jpg', full_img)
    print("✅ Saved 'full_img.jpg' with all gestures.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
