import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def unsharp_mask(image, kernel_size=(5, 5), sigma=0.6, amount=1.0):
    """Apply Unsharp Masking to deblur an image."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened

def compare_images(image1, image2):
    """Compute the Structural Similarity Index between two images."""
    score, _ = ssim(image1, image2, full=True)
    return score

input_image_path = '9.png'  # Replace with your image path
reference_image_path = '9.png'  # Replace with your image path

input_image = load_image(input_image_path)
reference_image = load_image(reference_image_path)

deblurred_image = unsharp_mask(input_image)
enhanced_image = enhance_contours(deblurred_image)

similarity_index = compare_images(input_image, reference_image)
print(f"Structural Similarity Index: {similarity_index} - no processing")
cv2.imwrite('input.png',input_image)
similarity_index = compare_images(deblurred_image, reference_image)
print(f"Structural Similarity Index: {similarity_index} - deblurred")
cv2.imwrite('deblurred.png',deblurred_image)
similarity_index = compare_images(enhanced_image, reference_image)