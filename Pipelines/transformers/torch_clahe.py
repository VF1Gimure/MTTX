import cv2
import numpy as np
from PIL import Image

class CLAHEBaseTransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply_clahe(self, img_np):
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(img_gray)  # Apply CLAHE and keep single channel

    def __call__(self, img):
        img_np = np.array(img)
        img_clahe = self.apply_clahe(img_np)
        return Image.fromarray(img_clahe, mode="L")  # Ensure single-channel grayscale

class CLAHETransform(CLAHEBaseTransform):
    pass

class CLAHEUnsharpTransform(CLAHEBaseTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), unsharp_amount=1.5):
        super().__init__(clip_limit, tile_grid_size)
        self.unsharp_amount = unsharp_amount

    def apply_unsharp_masking(self, img_np):
        gaussian_blur = cv2.GaussianBlur(img_np, (0, 0), 2)
        return cv2.addWeighted(img_np, self.unsharp_amount, gaussian_blur, -0.5, 0)

    def __call__(self, img):
        img_np = np.array(img)
        img_clahe = self.apply_clahe(img_np)
        img_sharpened = self.apply_unsharp_masking(img_clahe)
        return Image.fromarray(img_sharpened, mode="L")


class CLAHEUnsharpColorTransform(CLAHEBaseTransform):
    """
    NEW: Applies CLAHE + Unsharp Masking while PRESERVING COLOR (LAB color space).
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), unsharp_amount=1.5):
        super().__init__(clip_limit, tile_grid_size)
        self.unsharp_amount = unsharp_amount

    def apply_clahe(self, img_np):
        """
        Applies CLAHE on the L (lightness) channel in LAB color space.
        """
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)  # Convert to LAB
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)

        return cv2.merge([l_clahe, a, b])  # Merge enhanced L with original A and B

    def apply_unsharp_masking(self, img_np):
        """
        Applies unsharp masking on the enhanced L channel.
        """
        l, a, b = cv2.split(img_np)
        gaussian_blur = cv2.GaussianBlur(l, (0, 0), 2)
        l_sharpened = cv2.addWeighted(l, self.unsharp_amount, gaussian_blur, -0.5, 0)
        return cv2.merge([l_sharpened, a, b])

    def __call__(self, img):
        img_np = np.array(img)
        img_lab = self.apply_clahe(img_np)
        img_lab_sharpened = self.apply_unsharp_masking(img_lab)
        img_rgb = cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)