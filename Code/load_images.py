# import cv2
# from glob import glob
# import numpy as np
# import random
# from sklearn.utils import shuffle
# import pickle
# import os

# def pickle_images_labels():
# 	images_labels = []
# 	images = glob("gestures/*/*.jpg")
# 	images.sort()
# 	for image in images:
# 		print(image)
# 		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
# 		img = cv2.imread(image, 0)
# 		images_labels.append((np.array(img, dtype=np.uint8), int(label)))
# 	return images_labels

# images_labels = pickle_images_labels()
# images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
# images, labels = zip(*images_labels)
# print("Length of images_labels", len(images_labels))

# train_images = images[:int(5/6*len(images))]
# print("Length of train_images", len(train_images))
# with open("train_images", "wb") as f:
# 	pickle.dump(train_images, f)
# del train_images

# train_labels = labels[:int(5/6*len(labels))]
# print("Length of train_labels", len(train_labels))
# with open("train_labels", "wb") as f:
# 	pickle.dump(train_labels, f)
# del train_labels

# test_images = images[int(5/6*len(images)):int(11/12*len(images))]
# print("Length of test_images", len(test_images))
# with open("test_images", "wb") as f:
# 	pickle.dump(test_images, f)
# del test_images

# test_labels = labels[int(5/6*len(labels)):int(11/12*len(images))]
# print("Length of test_labels", len(test_labels))
# with open("test_labels", "wb") as f:
# 	pickle.dump(test_labels, f)
# del test_labels

# val_images = images[int(11/12*len(images)):]
# print("Length of test_images", len(val_images))
# with open("val_images", "wb") as f:
# 	pickle.dump(val_images, f)
# del val_images

# val_labels = labels[int(11/12*len(labels)):]
# print("Length of val_labels", len(val_labels))
# with open("val_labels", "wb") as f:
# 	pickle.dump(val_labels, f)
# del val_labels
import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
    images_labels = []
    images = glob(os.path.join("gestures", "*", "*.jpg"))
    images.sort()

    if len(images) == 0:
        print("❌ No images found in 'gestures' folder. Please ensure images exist before running this script.")
        return []

    for image in images:
        # Extract label from folder name (e.g., gestures/1/xyz.jpg → label=1)
        label = os.path.basename(os.path.dirname(image))
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Could not read {image}, skipping.")
            continue

        images_labels.append((np.array(img, dtype=np.uint8), int(label)))

    return images_labels


def main():
    images_labels = pickle_images_labels()
    if not images_labels:
        return

    # Shuffle data multiple times for good randomness
    for _ in range(4):
        images_labels = shuffle(images_labels, random_state=random.randint(0, 1000))

    images, labels = zip(*images_labels)

    print(f"📸 Total images loaded: {len(images_labels)}")

    # Split dataset into train, test, validation
    total_len = len(images_labels)
    train_end = int(5 / 6 * total_len)
    test_end = int(11 / 12 * total_len)

    train_images, train_labels = images[:train_end], labels[:train_end]
    test_images, test_labels = images[train_end:test_end], labels[train_end:test_end]
    val_images, val_labels = images[test_end:], labels[test_end:]

    # Save as pickle files
    data_splits = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "val_images": val_images,
        "val_labels": val_labels
    }

    for name, data in data_splits.items():
        with open(name, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Saved {name} ({len(data)})")

    print("\n🎉 Dataset successfully pickled and ready for training!")


if __name__ == "__main__":
    main()
