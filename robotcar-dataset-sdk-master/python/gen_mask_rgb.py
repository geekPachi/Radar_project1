import cv2
import os

# Input directory containing RGB images
input_directory = r'C:\Radar_project\tiny_dataset\radar-oxford-10k-partial\mono_right'

# Output directory for masks and modified RGB images
output_directory = r'C:\Radar_project\tiny_dataset\radar-oxford-10k-partial\Saved_images\mono_right'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    # Construct the full path to the input image
    rgb_image_path = os.path.join(input_directory, image_file)

    # Load the RGB image
    rgb_image = cv2.imread(rgb_image_path)

    # Perform image processing to create a mask (replace this with your mask generation code)
    # For example, here we create a simple binary mask using thresholding
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Combine RGB and mask
    combined_image = cv2.addWeighted(rgb_image, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Save the modified RGB image
    modified_image_path = os.path.join(output_directory, f'rgb_{image_file}')
    cv2.imwrite(modified_image_path, combined_image)

    # Save the mask image
    mask_image_path = os.path.join(output_directory, f'mask_{image_file}')
    cv2.imwrite(mask_image_path, mask)

print("Images and masks saved in the output directory.")
