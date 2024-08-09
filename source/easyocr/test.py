import pandas as pd
import numpy as np
from paddleocr import PPStructure
import cv2

# Initialize the table recognition model
table_engine = PPStructure(table=True, show_log=True, use_gpu=False)

# Specify the path to your pre-trained weights
table_engine.table_model_path = r"D:\Downloads\ch_ppstructure_openatom_SLANetv2_infer\inference.pdiparams"

# Load an image
img_path = r'D:\Workspace\PartTime\Heza-OCR\ExtractImages\15.jpg'
img = cv2.imread(img_path)

# Perform table recognition
result = table_engine(img)
print(result)
# # Draw bounding boxes around detected tables and cells
# for line in result:
#     if 'bbox' in line:
#         box = line['bbox']
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
#     if 'res' in line:
#         table_structure = line['res']
#         if 'text' in table_structure:
#             for cell in table_structure['cell_bbox']:
#                 bbox = cell
#                 cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)

# # Display the image with bounding boxes
# cv2.imshow('Table Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Process the results
for line in result:
    if line['type'] == 'table':
        table_structure = line['res']
        
        # Extract the text and cell structure
        html = table_structure['html']
        cells = table_structure['cell_bbox']
        
        # Create a 2D list to represent the table
        rows = html.split('<tr>')
        table_data = []
        for row in rows[1:]:  # Skip the first empty element
            cols = row.split('<td>')
            row_data = [col.split('</td>')[0].strip() for col in cols[1:]]  # Skip the first empty element
            table_data.append(row_data)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(table_data)
        
        # Print the DataFrame
        print(df)
        
        # Optionally, save to CSV
        df.to_csv('table_output.csv', index=False)

        # If you want to work with the cell coordinates
        for cell in cells:
            row_start, row_end = cell['row']
            col_start, col_end = cell['col']
            text = cell['text']
            bbox = cell['bbox']
            print(f"Cell: ({row_start},{col_start}) to ({row_end},{col_end}), Text: {text}, BBox: {bbox}")