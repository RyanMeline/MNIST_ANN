import cv2
import numpy as np
import struct

def to_mnist_format(digit_img):
    h, w = digit_img.shape
    pad = abs(h - w) // 2
    if h > w:
        digit_img = np.pad(digit_img, ((0,0),(pad,pad)), 'constant')
    else:
        digit_img = np.pad(digit_img, ((pad,pad),(0,0)), 'constant')
    digit_img = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
    digit_img = np.pad(digit_img, ((4,4),(4,4)), 'constant')
    return digit_img

def save_mnist_format(image_file, label_file, images, labels):
    with open(image_file, 'wb') as f:
        f.write(struct.pack('>IIII', 0x00000803, len(images), 28, 28))
        for img in images:
            f.write(img.astype(np.uint8).tobytes())
    with open(label_file, 'wb') as f:
        f.write(struct.pack('>II', 0x00000801, len(labels)))
        for label in labels:
            f.write(struct.pack('B', label))

# --- Load and threshold ---
img = cv2.imread("raw/DMELINE.PNG", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV ) # + cv2.THRESH_OTSU

kernel = np.ones((1,1), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)

cv2.imwrite('debug_thresh.png', thresh)
# --- Find digits ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Optional: save debug image to verify bounding boxes ---
debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 3 and h > 15:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imwrite('debug_boxes.png', debug_img)
print("Saved debug_boxes.png — check this before labeling")

# --- Label digits interactively ---
images = []
labels = []
total = sum(1 for cnt in contours if cv2.boundingRect(cnt)[2] > 10 and cv2.boundingRect(cnt)[3] > 10)
count = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 3 and h > 15:
        count += 1
        crop = thresh[y:y+h, x:x+w]
        mnist_img = to_mnist_format(crop)

        display = cv2.resize(mnist_img, (280, 280), interpolation=cv2.INTER_NEAREST)
        print(f"Digit {count} of {total} — press 0-9 to label, D to discard")
        cv2.imshow('What digit is this? (0-9 to label, D to discard)', display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if chr(key) in '0123456789':
                images.append(mnist_img)
                labels.append(int(chr(key)))
                break
            elif key == ord('d') or key == ord('D'):
                print("Discarded")
                break

cv2.destroyAllWindows()

# --- Save in MNIST IDX format ---
save_mnist_format('dad_images.idx3-ubyte', 'dad_labels.idx1-ubyte', images, labels)
print(f"Done — saved {len(images)} labeled digits")