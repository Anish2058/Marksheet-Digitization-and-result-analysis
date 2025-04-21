import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
from ultralyticsplus import YOLO, render_result
from paddleocr import PPStructure, save_structure_res, PaddleOCR
import re

image_path = 'S:/anishproject/project/marksheet_digitization/temp1/Image.jpg'  # Adjust the path if needed

threshold_angle=0

# Load the image
image = cv2.imread(image_path)

#print("Original Image:")
#cv2_imshow(image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#print("Grayscale Image:")
#cv2_imshow(gray)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#print("canny edge Image:")
#cv2_imshow(edges)

# Use Hough Line Transform to detect lines in the image
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)


# Calculate the average angle of the detected horizontal lines
angles = []
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if -10 < angle < 10:  # Focus on near-horizontal lines
                 angles.append(angle)
    if len(angles) == 0:
        print(f"No significant horizontal lines detected in image {image_path}.")

    median_angle = np.median(angles)

    # Check if the image is tilted
    if abs(median_angle) > threshold_angle:
        # Rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Save the deskewed image
        deskewed_image_path = os.path.splitext(image_path)[0] + '_deskewed.jpg'
        cv2.imwrite(deskewed_image_path, deskewed_image)

        # Convert back to PIL Image for display
        img1 = Image.fromarray(cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2RGB))

        # Display the original and deskewed images
        #print("Deskewed Image:")
        #cv2_imshow(deskewed_image)
        img = Image.open(deskewed_image_path)
        image_path = deskewed_image_path


    else:
        #print(f"Image {image_path} is not tilted.")
        img = Image.open(image_path)


else:
    #print(f"No lines detected in image {image_path}.")
    img = Image.open(image_path)



# Load YOLOv8 model for table extraction
#model = YOLO('keremberke/yolov8m-table-extraction')

model = YOLO('foduucom/table-detection-and-extraction')

# Set model parameters
model.overrides['conf'] = 0.25  # Lower confidence threshold
model.overrides['iou'] = 0.45  # Higher IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-specific
model.overrides['max_det'] = 1000  # Increase maximum detections

# perform inference
img_rgb = img.convert('RGB')
results = model.predict(img_rgb)

# observe results
print('Boxes: ', results[0].boxes)
render = render_result(model=model, image=img_rgb, result=results[0])
render

x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.cpu().numpy()[0]) # Move the tensor to CPU using .cpu() before converting to NumPy
img = np.array(Image.open(image_path)) # Use the correct variable 'image_path' instead of 'image'
#cropping
cropped_image = img[y1:y2, x1:x2]
cropped_image = Image.fromarray(cropped_image)
cropped_image.save('cropped_image.png', dpi=(300, 300))

# Convert the image to a NumPy array
img_array = np.array(img)

# Assuming 'results' contains the detection results from YOLO
# results[0].boxes.data.cpu().numpy() should provide an array of bounding boxes
# Format: [[x1, y1, x2, y2, _, _], [x3, y3, x4, y4, _, _], ...]

# Create a mask with the same dimensions as the image, initialized to True (1)
mask = np.ones(img_array.shape[:2], dtype=bool)

# Get the dimensions of the image
height, width = img_array.shape[:2]

# Process each detected bounding box and update the mask
for box in results[0].boxes.data.cpu().numpy():
    x1, y1, _, _, _, _ = tuple(int(item) for item in box)
    mask[y1:height, x1:width] = False

# Apply the mask to the image
mask_image = np.zeros_like(img_array)
mask_image[mask] = img_array[mask]

# Convert the cropped numpy array back to an image
mask_image = Image.fromarray(mask_image)

# Display the cropped image
#mask_image.show()

# Convert the cropped PIL Image to a NumPy array
mask_image_np = np.array(mask_image)

# Display the cropped image using cv2_imshow
#cv2_imshow(mask_image_np)


ocr = PaddleOCR(use_angle_cls=True, lang='en')
table_engine = PPStructure(show_log=True, lang='en', layout=False)

save_folder='temp2'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cropped_image_path='cropped_image.png'
# Get table structure result
img = cv2.imread(cropped_image_path)
result = table_engine(img)

# Save the structure result
save_structure_res(result, save_folder, os.path.basename(cropped_image_path).split('.')[0])

image_xlsx_file = os.path.join(save_folder, os.path.basename(cropped_image_path).split('.')[0] + '.xlsx')

# Print the results without the image data
for line in result:
    line.pop('img')
    print(line)



#img_np = np.array(mask_image_np)


# Convert to grayscale
gray = cv2.cvtColor(mask_image_np, cv2.COLOR_BGR2GRAY)

# Adjust the contrast of the image
alpha = 1.5  # Simple contrast control (1.0-3.0)
beta = 0    # Simple brightness control (0-100)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Apply histogram equalization
equalized = cv2.equalizeHist(adjusted)

# Apply median blurring to remove salt-and-pepper noise
median = cv2.medianBlur(equalized, 3)

# Apply Otsu's binarization
_, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to reduce noise and separate characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded = cv2.erode(otsu, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# Save the preprocessed image for visualization (optional)
#cv2.imwrite('preprocessed_image.png', dilated)

# Configure tesseract
custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789*:-./& "'

# Perform OCR
d = pytesseract.image_to_data(dilated, config=custom_config, output_type=Output.DICT)
# Convert to DataFrame
df = pd.DataFrame(d)


# Clean up blanks
df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Sort blocks vertically
sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()

# Initialize text storage
extracted_text = ""

# Extract text block by block
for block in sorted_blocks:
    curr = df1[df1['block_num'] == block]
    sel = curr[curr.text.str.len() > 3]
    char_w = (sel.width / sel.text.str.len()).mean()
    prev_par, prev_line, prev_left = 0, 0, 0
    text = ''
    for ix, ln in curr.iterrows():
        # Add new line when necessary
        if prev_par != ln['par_num']:
            text += '\n'
            prev_par = ln['par_num']
            prev_line = ln['line_num']
            prev_left = 0
        elif prev_line != ln['line_num']:
            text += '\n'
            prev_line = ln['line_num']
            prev_left = 0

        added = 0  # Number of spaces that should be added
        if ln['left'] / char_w > prev_left + 1:
            added = int((ln['left']) / char_w) - prev_left
            text += ' ' * added
        text += ln['text'] + ' '
        prev_left += len(ln['text']) + added + 1
    text += '\n'
    extracted_text += text

# Extract the portion of the text starting from "Name" to the next 3 lines
lines = extracted_text.split("\n")
start_keyword = "Name"
relevant_lines = []

# Find the index of the line containing "Name"
start_index = next((i for i, line in enumerate(lines) if start_keyword in line), -1)

# If found, extract the relevant lines
if start_index != -1:
    relevant_lines = lines[start_index:start_index + 7]

# Join the extracted lines
cleaned_text = "\n".join(relevant_lines)
print(cleaned_text)


def convert_text_to_xlsx(text, xlsx_file_path):
    # Define the order and pairs of headers we expect in the input text
    expected_pairs = [
        ("Name", "Exam Roll No"),
        ("Level", "CRN"),
        ("Campus", "T.U. Regd. No"),
        ("Year/Part", "Programme")
    ]

    # Split the text into lines
    lines = text.strip().split('\n')

    # Extract headers and data
    data = {}
    for line in lines:
        for header1, header2 in expected_pairs:
            if header1 == "Year/Part" and header2 == "Programme":
                # Specific handling for Year/Part and Programme
                pattern = fr'{header1}\s*-\s*([^\n]+)\s+{header2}\s*:-\s*([^\n]+)'
            else:
                pattern = fr'{header1}\s*:-\s*([^-\n]+)\s*{header2}\s*:-\s*([^-\n]+)'
            match = re.search(pattern, line)
            if match:
                data[header1] = match.group(1).strip()
                data[header2] = match.group(2).strip()
                break

    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Write data to Excel
    df.to_excel(xlsx_file_path, index=False)

# Define the Excel file path
xlsx_file_path = 'student_data.xlsx'

# Convert text to Excel
convert_text_to_xlsx(cleaned_text, xlsx_file_path)

print(f"Excel file has been created at: {xlsx_file_path}")


# Define the function to merge multiple XLSX files
def merge_xlsx_files(xlsx_files, merged_xlsx_file):
    # List to hold all DataFrames
    dfs = []

    # Read each XLSX file into a DataFrame and append to the list
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file)
        dfs.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged DataFrame to a new XLSX file
    merged_df.to_excel(merged_xlsx_file, index=False)

    print(f"Merged Excel file has been created at: {merged_xlsx_file}")

# Define the folder containing the XLSX files from image processing
cropped_image_folder = 'temp2/cropped_image'


# List all XLSX files in the folder
cropped_image_xlsx_files = [os.path.join(cropped_image_folder, f) for f in os.listdir(cropped_image_folder) if f.endswith('.xlsx')]

# Define the Excel file path created from text conversion
text_xlsx_file = 'student_data.xlsx'

# Add the text XLSX file to the list of XLSX files to merge
all_xlsx_files = cropped_image_xlsx_files + [text_xlsx_file]

# Define the path for the merged XLSX file
merged_xlsx_file = 'merged_student_data.xlsx'

# Merge all XLSX files
merge_xlsx_files(all_xlsx_files, merged_xlsx_file)


# Define the function to delete files in a folder
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"All files in {folder_path} have been deleted.")

# Delete all XLSX files in the cropped_image_folder
delete_files_in_folder(cropped_image_folder)

# Delete the student_data.xlsx file
if os.path.isfile(text_xlsx_file):
    os.remove(text_xlsx_file)
    print(f"{text_xlsx_file} has been deleted.")
else:
    print(f"{text_xlsx_file} does not exist.")