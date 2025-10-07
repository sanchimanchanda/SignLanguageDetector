# import cv2
# import numpy as np
# import pickle

# def build_squares(img):
# 	x, y, w, h = 420, 140, 10, 10
# 	d = 10
# 	imgCrop = None
# 	crop = None
# 	for i in range(10):
# 		for j in range(5):
# 			if np.any(imgCrop == None):
# 				imgCrop = img[y:y+h, x:x+w]
# 			else:
# 				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
# 			#print(imgCrop.shape)
# 			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
# 			x+=w+d
# 		if np.any(crop == None):
# 			crop = imgCrop
# 		else:
# 			crop = np.vstack((crop, imgCrop)) 
# 		imgCrop = None
# 		x = 420
# 		y+=h+d
# 	return crop

# def get_hand_hist():
# 	cam = cv2.VideoCapture(1)
# 	if cam.read()[0]==False:
# 		cam = cv2.VideoCapture(0)
# 	x, y, w, h = 300, 100, 300, 300
# 	flagPressedC, flagPressedS = False, False
# 	imgCrop = None
# 	while True:
# 		img = cam.read()[1]
# 		img = cv2.flip(img, 1)
# 		img = cv2.resize(img, (640, 480))
# 		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
# 		keypress = cv2.waitKey(1)
# 		if keypress == ord('c'):		
# 			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
# 			flagPressedC = True
# 			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
# 			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
# 		elif keypress == ord('s'):
# 			flagPressedS = True	
# 			break
# 		if flagPressedC:	
# 			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
# 			dst1 = dst.copy()
# 			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
# 			cv2.filter2D(dst,-1,disc,dst)
# 			blur = cv2.GaussianBlur(dst, (11,11), 0)
# 			blur = cv2.medianBlur(blur, 15)
# 			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 			thresh = cv2.merge((thresh,thresh,thresh))
# 			#cv2.imshow("res", res)
# 			cv2.imshow("Thresh", thresh)
# 		if not flagPressedS:
# 			imgCrop = build_squares(img)
# 		#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
# 		cv2.imshow("Set hand histogram", img)
# 	cam.release()
# 	cv2.destroyAllWindows()
# 	with open("hist", "wb") as f:
# 		pickle.dump(hist, f)


# get_hand_hist()
import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    # Try to access external webcam first, fallback to default
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist = None

    print("🖐 Press 'C' to capture the hand region and create histogram.")
    print("💾 Press 'S' to save histogram and exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            print("❌ Camera not detected.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                flagPressedC = True
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                print("✅ Histogram created. Press 'S' to save.")
            else:
                print("⚠️ No hand region captured yet.")

        elif keypress == ord('s'):
            if hist is not None:
                flagPressedS = True
                break
            else:
                print("⚠️ Please create histogram first by pressing 'C'.")

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresholded Hand", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Set Hand Histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        print("💾 Histogram saved successfully as 'hist'.")
    else:
        print("❌ No histogram saved. Try again.")

if __name__ == "__main__":
    get_hand_hist()
