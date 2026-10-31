import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# 1. Load the pre-trained VGG16 model with ImageNet weights
print("Loading VGG16 model...")
model = VGG16(weights='imagenet')

# 2. Define the exact file path to your image
image_path = r'C:\Users\pipow\PycharmProjects\CobblerLearnChemProjects\ImageClassifier\photo-1570913149827-d2ac84ab3f9a_fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.webp'

# 3. Process the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
processed_image = preprocess_input(img_array_expanded)

print("Image successfully loaded, resized, and preprocessed!")

# 4. Run the model prediction (returns a probability distribution across 1000 ImageNet classes)
print("\nRunning model prediction...")
predictions = model.predict(processed_image)

# 5. Decode predictions to show the top 5 highest confidence classes
# You can change 'top=5' to any number you prefer (e.g., top=3 or top=10)
num_top_predictions = 5
decoded_preds = decode_predictions(predictions, top=num_top_predictions)[0]

print(f"\n--- Top {num_top_predictions} Model Predictions ---")
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}. Label: {label.capitalize()} | Confidence: {score * 100:.2f}%")