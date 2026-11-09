# Day 7: Transfer Learning & YOLO Object Detection

## 1. Scenario Context & Objective
- **Problem Statement**: Scratch CNNs require massive datasets and struggle with complex RGB textures. Additionally, catalog images often contain multiple items, requiring spatial detection before classification.
- **Engineering Deliverable**: A fine-tuned MobileNetV2 transfer classifier for RGB products and a YOLOv8 bounding-box detector for multi-product shots and shelf auditing.

## 2. Architecture & Fine-Tuning Strategy
1. **Feature Extraction**: Base MobileNetV2 pre-trained on ImageNet. Initial pass freezes all backbone weights and trains a custom classification head.
2. **Fine-Tuning**: Unfreezes top 20 backbone layers with a small learning rate (\1e-5\) to adapt representations without catastrophic forgetting.
3. **Detection (YOLOv8)**: Generates localized bounding boxes (\[x1, y1, x2, y2]\) and confidence scores to crop individual products from lifestyle and multi-item catalog shots.

## 3. Business Value
- **Catalog QA**: Rejects images where the target garment is obscured or absent.
- **Multi-Product Tagging**: Automatically detects and tags both "Shirt" and "Trouser" in full-body model photos.
