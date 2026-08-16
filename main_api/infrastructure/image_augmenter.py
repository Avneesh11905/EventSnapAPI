import base64
from typing import List
import numpy as np
import cv2
from application.ports.image_services import IImageAugmenter


class OpenCVImageAugmenter(IImageAugmenter):
    def augment(self, b64_images: List[str]) -> List[str]:
        augmented_b64_images = []
        for b64_str in b64_images:
            image_data = base64.b64decode(b64_str)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            augmented_b64_images.append(b64_str)

            # Sharpness (Unsharp Masking approximation)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp_img = cv2.filter2D(img, -1, kernel)
            _, buf1 = cv2.imencode(
                ".jpg", sharp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            augmented_b64_images.append(base64.b64encode(buf1).decode("utf-8"))

            # Brightness (Scale pixel values by 1.2)
            bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
            _, buf2 = cv2.imencode(
                ".jpg", bright_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            augmented_b64_images.append(base64.b64encode(buf2).decode("utf-8"))

        return augmented_b64_images
