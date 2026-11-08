import os
import numpy as np
import tensorflow as tf

CNN_MODEL_PATH = os.path.join("models", "catalog_cnn_v6.keras")

class CatalogVisionService:
    def __init__(self):
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"Vision model not found at {CNN_MODEL_PATH}. Run Day 6 pipeline.")
        self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def predict_image(self, image_array: np.ndarray) -> dict:
        """
        Accepts a preprocessed image array of shape (1, 28, 28, 1) and returns the top category.
        """
        predictions = self.model.predict(image_array, verbose=0)
        confidence = np.max(predictions[0])
        class_idx = np.argmax(predictions[0])
        
        return {
            "predicted_category": self.class_names[class_idx],
            "confidence_score": round(float(confidence), 4),
            "status": "Auto-Tagged" if confidence > 0.75 else "Requires Manual Review"
        }

if __name__ == "__main__":
    # Test service with a dummy dummy (like an empty image canvas)
    dummy_image = np.zeros((1, 28, 28, 1), dtype="float32")
    service = CatalogVisionService()
    
    print("[ACTIVE SERVICE] Day 6 Vision Auto-Tagger (CNN)")
    result = service.predict_image(dummy_image)
    print(f"Prediction Result: {result}")
