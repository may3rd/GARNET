from garnet.utils.deeplsd_utils import define_torch_device, load_deeplsd_model, detect_lines, draw_and_save_lines, export_lines_to_json, combine_close_lines
import numpy as np # Added for np.array in draw_and_save_lines for combined_lines

# Example usage
device = define_torch_device()
model, conf = load_deeplsd_model(device)
image_to_process = 'output/stage5_skeleton_inverted.png'
gray_img, detected_lines = detect_lines(model, image_to_process, device)
output_image_path = 'out/output.jpg'
draw_and_save_lines(gray_img, detected_lines, output_image_path)
export_lines_to_json(detected_lines, 'out/detected_lines.json')

# Combine close lines and draw them
combined_lines = combine_close_lines(detected_lines.tolist()) # Convert numpy array to list for the function
combined_output_image_path = 'out/combined_lines.jpg'
draw_and_save_lines(gray_img, np.array(combined_lines), combined_output_image_path)
print(f"Combined {len(detected_lines)} lines into {len(combined_lines)} lines.")

