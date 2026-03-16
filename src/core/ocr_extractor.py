import pytesseract
import cv2


def extract_text_from_image(image_path):

    try:
        image = cv2.imread(image_path)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # enlarge image
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        text = pytesseract.image_to_string(thresh)

        return text.strip()

    except:
        return ""