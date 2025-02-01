# import streamlit as st
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLO model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect objects using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)

#     # Run inference
#     results = model(image_np)
    
#     # Extract and visualize bounding boxes
#     for result in results:
#         im_array = result.plot()  # Draw bounding boxes
#         im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

#         # Display image in Streamlit
#         st.image(im_array, caption="Detected Objects", use_column_width=True)
        
#         # Option to download the processed image
#         output_path = "detected_image.jpg"
#         cv2.imwrite(output_path, im_bgr)
#         with open(output_path, "rb") as file:
#             btn = st.download_button(label="Download Processed Image",
#                                      data=file,
#                                      file_name="detected_image.jpg",
#                                      mime="image/jpeg")








# import streamlit as st
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLO model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect objects using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)

#     # Run inference
#     results = model(image_np)
    
#     # Extract and visualize bounding boxes
#     for result in results:
#         im_array = result.plot()  # Draw bounding boxes
#         im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

#         # Display image in Streamlit
#         st.image(im_array, caption="Detected Objects", use_container_width=True)
        
#         # Option to download the processed image
#         output_path = "detected_image.jpg"
#         cv2.imwrite(output_path, im_bgr)
#         with open(output_path, "rb") as file:
#             btn = st.download_button(label="Download Processed Image",
#                                      data=file,
#                                      file_name="detected_image.jpg",
#                                      mime="image/jpeg")









# import streamlit as st
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLO model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect objects using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)

#     image_resized = image.resize((300, 200))
    
#     # Display the original image first
#     st.subheader("Original Image")
#     st.image(image, caption="Original Image", use_container_width=False)
    
#     # Run inference
#     results = model(image_np)
    
#     # Extract and visualize bounding boxes
#     for result in results:
#         im_array = result.plot()  # Draw bounding boxes
#         im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

#         image_resized = image.resize((300, 200))

#         # Display the processed image (detected objects)
#         st.subheader("Processed Image (Detected Objects)")
#         st.image(im_array, caption="Processed Image (Detected Objects)", use_container_width=False)
        
#         # Option to download the processed image
#         output_path = "detected_image.jpg"
#         cv2.imwrite(output_path, im_bgr)
#         with open(output_path, "rb") as file:
#             btn = st.download_button(label="Download Processed Image",
#                                      data=file,
#                                      file_name="detected_image.jpg",
#                                      mime="image/jpeg")










# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLO model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect objects using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
    
#     # Resize image for better display (300x200)
#     # image_resized = image.resize((300, 200))  # Resize to 300x200

#     # Display original image with headline
#     st.subheader("Original Image")
#     st.image(image, caption="Original Image", width=500)  # Resize the image to 300px width
    
#     # Run inference
#     results = model(image_np)
    
#     # Extract and visualize bounding boxes
#     for result in results:
#         im_array = result.plot()  # Draw bounding boxes
#         im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

#         # Resize processed image for better display (300x200)
#         # processed_image_resized = Image.fromarray(im_array).resize((300, 200))  # Resize to 300x200

#         # Display processed image with headline
#         st.subheader("Processed Image (Detected Objects)")
#         st.image(im_array, caption="Processed Image (Detected Objects)", width=500)  # Resize the image to 300px width
        
#         # Option to download the processed image
#         output_path = "detected_image.jpg"
#         cv2.imwrite(output_path, im_bgr)
#         with open(output_path, "rb") as file:
#             btn = st.download_button(label="Download Processed Image",
#                                      data=file,
#                                      file_name="detected_image.jpg",
#                                      mime="image/jpeg")









# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd

# # Load trained model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect pillars using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
    
#     # Run inference
#     results = model(image_np)
    
#     # Initialize a list to store pillar coordinates
#     pillar_coordinates = []

#     # Extract bounding box coordinates for each detected pillar
#     for idx, result in enumerate(results):
#         boxes = result.boxes.xyxy  # Get coordinates as (x1, y1, x2, y2)
#         labels = result.names  # Object class labels
        
#         # Loop through each detected object
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box  # Bounding box coordinates
#             label = labels[int(result.boxes.cls[i])]  # Object label
            
#             # If the label is 'pillar', store the coordinates
#             if label == 'pillar':  # Ensure you're detecting 'pillar'
#                 pillar_coordinates.append({
#                     'Pillar': f'Pillar {len(pillar_coordinates) + 1}',
#                     'x1': x1,
#                     'y1': y1,
#                     'x2': x2,
#                     'y2': y2
#                 })
    
#     # Display the coordinates of detected pillars
#     if pillar_coordinates:
#         st.subheader("Detected Pillars and Coordinates")
#         df = pd.DataFrame(pillar_coordinates)
#         st.write(df)
        
#         # Option to download the data as Excel
#         excel_file = "pillar_coordinates.xlsx"
#         df.to_excel(excel_file, index=False)
#         with open(excel_file, "rb") as file:
#             st.download_button(
#                 label="Download Coordinates as Excel",
#                 data=file,
#                 file_name=excel_file,
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
    
#     # Option to display the image with bounding boxes
#     for result in results:
#         im_array = result.plot()  # Draw bounding boxes
#         im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)

#         # Display the image
#         st.subheader("Processed Image (Detected Pillars)")
#         st.image(im_array, caption="Processed Image with Pillars Detected", use_container_width=True)
        
#         # Option to download the processed image
#         output_path = "detected_pillars_image.jpg"
#         cv2.imwrite(output_path, im_bgr)
#         with open(output_path, "rb") as file:
#             st.download_button(
#                 label="Download Processed Image",
#                 data=file,
#                 file_name=output_path,
#                 mime="image/jpeg"
#             )






# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd

# # Load trained model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect pillars using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
    
#     # Run inference
#     results = model(image_np)
    
#     # Initialize a list to store pillar coordinates
#     pillar_coordinates = []

#     # Extract bounding box coordinates for each detected pillar
#     for idx, result in enumerate(results):
#         boxes = result.boxes.xyxy  # Get coordinates as (x1, y1, x2, y2)
#         labels = result.names  # Object class labels
        
#         # Loop through each detected object
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box  # Bounding box coordinates
#             label = labels[int(result.boxes.cls[i])]  # Object label
            
#             # If the label is 'pillar', store the coordinates
#             if label == 'pillar':  # Ensure you're detecting 'pillar'
#                 pillar_coordinates.append({
#                     'Pillar': f'Pillar {len(pillar_coordinates) + 1}',
#                     'x1': x1,
#                     'y1': y1,
#                     'x2': x2,
#                     'y2': y2
#                 })
    
#     # Display the original image without bounding boxes
#     st.subheader("Original Image")
#     st.image(image, caption="Original Image", use_container_width=True)

#     # Option to download the coordinates as Excel
#     if pillar_coordinates:
#         st.subheader("Download Pillar Coordinates")
#         df = pd.DataFrame(pillar_coordinates)
#         excel_file = "pillar_coordinates.xlsx"
#         df.to_excel(excel_file, index=False)
#         with open(excel_file, "rb") as file:
#             st.download_button(
#                 label="Download Coordinates as Excel",
#                 data=file,
#                 file_name=excel_file,
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#     else:
#         st.write("No pillars detected in the image.")













# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd

# # Load trained model
# model = YOLO("runs/detect/train/weights/best.pt")  # Ensure this path is correct

# # Streamlit UI
# st.title("YOLOv8 Object Detection App")
# st.write("Upload an image to detect pillars using YOLOv8")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV format
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
    
#     # Run inference
#     results = model(image_np)
    
#     # Initialize a list to store pillar coordinates
#     pillar_coordinates = []
#     processed_image = image_np.copy()  # Copy original image for processing

#     # Extract bounding box coordinates for each detected pillar
#     for idx, result in enumerate(results):
#         boxes = result.boxes.xyxy  # Get coordinates as (x1, y1, x2, y2)
#         labels = result.names  # Object class labels
        
#         # Loop through each detected object
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box  # Bounding box coordinates
#             label = labels[int(result.boxes.cls[i])]  # Object label
            
#             # If the label is 'pillar', store the coordinates and draw bounding boxes
#             if label == 'pillar':  # Ensure you're detecting 'pillar'
#                 pillar_coordinates.append({
#                     'Pillar': f'Pillar {len(pillar_coordinates) + 1}',
#                     'x1': x1,
#                     'y1': y1,
#                     'x2': x2,
#                     'y2': y2
#                 })
                
#                 # Draw bounding box on the processed image (using OpenCV)
#                 cv2.rectangle(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#     # Display the original image
#     st.subheader("Original Image")
#     st.image(image, caption="Original Image", use_container_width=True)

#     # Display the processed image with bounding boxes
#     st.subheader("Processed Image with Bounding Boxes")
#     st.image(processed_image, caption="Processed Image", use_container_width=True)

#     # Option to download the processed image with bounding boxes
#     output_path = "processed_image.jpg"
#     cv2.imwrite(output_path, processed_image)
#     with open(output_path, "rb") as file:
#         st.download_button(
#             label="Download Processed Image",
#             data=file,
#             file_name=output_path,
#             mime="image/jpeg"
#         )

#     # Option to download the coordinates as Excel
#     if pillar_coordinates:
#         st.subheader("Download Pillar Coordinates")
#         df = pd.DataFrame(pillar_coordinates)
#         excel_file = "pillar_coordinates.xlsx"
#         df.to_excel(excel_file, index=False)
#         with open(excel_file, "rb") as file:
#             st.download_button(
#                 label="Download Coordinates as Excel",
#                 data=file,
#                 file_name=excel_file,
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#     else:
#         st.write("No pillars detected in the image.")














import streamlit as st # type: ignore
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Load trained model
model = YOLO("runs/detect/train6/weights/best.pt")  # Ensure this path is correct

# Streamlit UI
st.title("YOLOv8 Object Detection App")
st.write("Upload an image to detect pillars using YOLOv8")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Run inference
    results = model(image_np)
    
    # Initialize a list to store pillar coordinates
    pillar_coordinates = []
    processed_image = image_np.copy()  # Copy original image for processing

    # Extract bounding box coordinates for each detected pillar
    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy  # Get coordinates as (x1, y1, x2, y2)
        labels = result.names  # Object class labels
        
        # Loop through each detected object
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box  # Bounding box coordinates
            label = labels[int(result.boxes.cls[i])]  # Object label
            
            # If the label is 'pillar', store the coordinates and draw bounding boxes
            if label == 'pillar':  # Ensure you're detecting 'pillar'
                pillar_coordinates.append({
                    'Pillar': f'Pillar {len(pillar_coordinates) + 1}',
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
                
                # Draw bounding box on the processed image (using OpenCV)
                cv2.rectangle(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_container_width=True)

    # Display the processed image with bounding boxes
    st.subheader("Processed Image with Bounding Boxes")
    st.image(processed_image, caption="Processed Image", use_container_width=True)

    # Show pillar coordinates below the processed image
    if pillar_coordinates:
        st.subheader("Pillar Coordinates")
        df = pd.DataFrame(pillar_coordinates)
        st.table(df)  # Display pillar coordinates as a table
        
        # Option to download the processed image with bounding boxes
        output_path = "processed_image.jpg"
        cv2.imwrite(output_path, processed_image)
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Image",
                data=file,
                file_name=output_path,
                mime="image/jpeg"
            )

        # Option to download the coordinates as Excel
        excel_file = "pillar_coordinates.xlsx"
        df.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as file:
            st.download_button(
                label="Download Coordinates as Excel",
                data=file,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.write("No pillars detected in the image.")
