# Day 8: Packaged CV Module & Explainability

## 1. Scenario Context & Objective
- **Problem Statement**: E-commerce visual consistency is paramount. Messy backgrounds hurt conversion, and black-box AI decisions erode trust among catalog managers.
- **Engineering Deliverable**: An end-to-end CV module that segments out backgrounds (OpenCV/U-Net), tags the product (CNN), and renders a Grad-CAM heatmap showing the decisive pixels.

## 2. Techniques Taught
- **Otsu's Thresholding (OpenCV)**: Automatically calculates the optimal bimodal threshold to separate the foreground product from the background canvas.
- **Grad-CAM**: Uses the gradient of the classification score with respect to the final convolutional feature map to highlight the regions that most influenced the network's prediction.

## 3. Production Integration
- \src/vision_service.py\ was refactored into a \PackagedCVModule\.
- Input: Raw image tensor.
- Process: \emove_background()\ -> \classifier.predict()\ -> \get_gradcam_heatmap()\.
- Output: Standardized image array, text classification, and a localized heatmap array.
