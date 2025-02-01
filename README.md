# 🚀 YOLOv8 Object Detection with Streamlit

This project uses **YOLOv8** for real-time object detection of **pillars** in images, deployed through a **Streamlit** web app. Users can upload images, see detected pillars with bounding boxes, and download processed images and their coordinates.

## 📋 Project Overview

- **YOLOv8 Model**: Trained to detect pillars in images.
- **Streamlit App**: 
  - Users can upload images for detection.
  - View original and processed images with bounding boxes.
  - Download the processed image and detected coordinates in Excel format.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install streamlit opencv-python numpy ultralytics Pillow pandas
