import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
import matplotlib.pyplot as plt

input_image_path = './images/27twentyseven.jpg'  
segmented_output_path = './images/removed/segmented_image27.png'  
manual_segmented_path = './images/output/manual-27.jpg'  
final_output_auto_path = './images/output/final_auto_image27.png'  
new_bg_path = './images/back-2.png'  

def remove_background(input_image_path, output_segmented_path):
    """Remove the background using rembg and save the segmented image with transparency."""
    with open(input_image_path, 'rb') as input_file:
        input_bytes = input_file.read()
        output_bytes = remove(input_bytes)  
    with open(output_segmented_path, 'wb') as output_file:
        output_file.write(output_bytes)
    return output_segmented_path

def change_background(input_segmented_path, new_bg_path, final_output_path, target_size=None):
    """Change the background of the segmented image."""
    segmented_img = cv2.imread(input_segmented_path, cv2.IMREAD_UNCHANGED)
    if segmented_img is None:
        print(f"Error: The image at {input_segmented_path} could not be loaded!")
        return False

    if segmented_img.shape[-1] != 4:
        print(f"Adding alpha channel to the image at {input_segmented_path}")
        b, g, r = cv2.split(segmented_img)  
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  
        segmented_img = cv2.merge((b, g, r, alpha))

    if target_size:
        segmented_img = cv2.resize(segmented_img, target_size, interpolation=cv2.INTER_AREA)

    new_bg = Image.open(new_bg_path).convert('RGB')  
    new_bg_resized = new_bg.resize((segmented_img.shape[1], segmented_img.shape[0]))
    new_bg_np = np.array(new_bg_resized)

    new_bg_np = cv2.cvtColor(new_bg_np, cv2.COLOR_RGB2BGR)

    alpha_channel = segmented_img[:, :, 3]
    alpha_normalized = alpha_channel / 255.0

    foreground = segmented_img[:, :, :3]
    blended_foreground = (foreground * alpha_normalized[:, :, None]).astype(np.uint8)
    blended_background = (new_bg_np * (1 - alpha_normalized[:, :, None])).astype(np.uint8)

    final_result = cv2.add(blended_foreground, blended_background)

    cv2.imwrite(final_output_path, final_result)
    print(f"Final image saved to {final_output_path}")
    return True

def preprocess_mask(image_path, target_size=None):
    """Load and preprocess the mask image. Converts it to binary and resizes if needed."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    
    _, binary_mask = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def dice_index(manual_path, auto_path, target_size=None):
    """Calculate the Dice Index between manual and auto-segmented images."""
    manual_mask = preprocess_mask(manual_path, target_size)
    auto_mask = preprocess_mask(auto_path, target_size)
    
    intersection = np.sum(manual_mask * auto_mask)
    total_pixels = np.sum(manual_mask) + np.sum(auto_mask)
    dice = (2.0 * intersection) / total_pixels if total_pixels > 0 else 0.0
    return dice

def show_images_for_comparison(segmented_path, ground_truth_path, target_size):
    """Display segmented and ground truth images for visual comparison using Matplotlib."""
    seg_img = cv2.imread(segmented_path)  # Read in color
    ground_truth_img = cv2.imread(ground_truth_path)  # Read in color
    if seg_img is None or ground_truth_img is None:
        print("Error: One or both images could not be loaded for display.")
        return

    seg_img = cv2.resize(seg_img, target_size, interpolation=cv2.INTER_NEAREST)
    ground_truth_img = cv2.resize(ground_truth_img, target_size, interpolation=cv2.INTER_NEAREST)

    
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    ground_truth_img = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2RGB)

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Segmented Image")
    plt.imshow(seg_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth Image")
    plt.imshow(ground_truth_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


segmented_image_path = remove_background(input_image_path, segmented_output_path)

original_image = cv2.imread(input_image_path)
if original_image is not None:
    original_size = (original_image.shape[1], original_image.shape[0])  
else:
    print(f"Error: Could not load the original image from {input_image_path}.")
    exit()

if os.path.exists(segmented_image_path):
    change_background(segmented_image_path, new_bg_path, final_output_auto_path, original_size)

if os.path.exists(manual_segmented_path):
    dice_score = dice_index(manual_segmented_path, final_output_auto_path, original_size)
    print(f"Dice Index (Auto vs Manual): {dice_score}")


show_images_for_comparison(final_output_auto_path, manual_segmented_path, original_size)
