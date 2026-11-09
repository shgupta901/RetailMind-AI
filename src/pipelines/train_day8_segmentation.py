import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout

MODEL_DIR = "models"
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "catalog_unet_v8.keras")

def build_unet(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    
    # Contracting Path (Encoder)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    d3 = Dropout(0.3)(c3)
    
    # Expansive Path (Decoder)
    u4 = UpSampling2D((2, 2))(d3)
    concat4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    
    u5 = UpSampling2D((2, 2))(c4)
    concat5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
    
    # Output Layer (Binary Mask: 1 for product, 0 for background)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_day8_segmentation():
    print("[INFO] Building U-Net Segmentation Architecture...")
    unet = build_unet()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    unet.save(UNET_MODEL_PATH)
    print(f"[SUCCESS] U-Net Prototype saved to: {UNET_MODEL_PATH}")
    print("[INFO] (In production, this is trained on paired images and alpha-masks).")

if __name__ == "__main__":
    run_day8_segmentation()
