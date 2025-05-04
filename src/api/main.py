
import os
import torch
from torchvision import transforms
from PIL import Image
import io
import logging
import google.generativeai as genai
import socket
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, ViTForImageClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT model setup for animal prediction
class_names = sorted(os.listdir(r"C:\Users\Meges\Downloads\DermaVet_ML\Dataset_Animal"))
num_classes = len(class_names)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(r"C:\Users\Meges\Downloads\DermaVet_ML\main.pth", map_location=device))
model.to(device)
model.eval()

# Transform for ViT (used for both animal and skin condition prediction)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads")
MODEL_DIR = r"C:\Users\Meges\Downloads\DermaVet_ML\Code"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Animal and skin condition classes
ANIMAL_CLASSES = [
    "rabbits", "freshwater_Fish", "donkeys", "Sheep", "Goat", "Horse", "Poultry",
    "Dogs", "Cat", "Cow"
]

SKIN_CLASSES = {
    "sheep": ["sheep scaby", "Orf", "Dermatophilosis","Fleece Rot","healthy"],
    "rabbits": ["Fur Mites", "Healthy", "Myxomatosis", "Ringworm", "Bumblefoot", "Ear Mites"],
    "poultry": ["Scaly Leg Mites", "Healthy", "Feather Mites", "Ringworm", "Fowl Cholera", "Fowl Pox"],
    "horse": ["Rain Rot", "Healthy", "Sarcoids", "Sweet Itch", "Mud Fever", "Caseous Lymphadenitis (CLA)"],
    "goat": ["Dermatophilosis", "Healthy", "Lice Infestations", "Mange", "Ringworm", "Extra Class"],
    "freshwater_fish": ["Healthy", "Parasitic Diseases", "Viral Diseases", "Bacterial Gill Disease", "Bacterial Red Disease", "Fungal Diseases", "Extra Class"],
    "donkeys": ["Ringworm", "Sweet Itch", "Mange", "Fungal Infections", "Aeromonas"],
    "dogs": ["Hypersensitivity Allergic Dermatosis", "White Tail Disease", "Healthy", "Habronemiasis"],
    "cat": ["Healthy", "Flea Allergy", "Bacterial Dermatosis", "Ringworm"],
    "cow": ["Healthy", "Lumpy Skin Disease"]
}

# Model paths for skin condition prediction (ViT models)
SECONDARY_MODELS = {
    "sheep": os.path.join(MODEL_DIR, "Sheep.pth"),
    "rabbits": os.path.join(MODEL_DIR, "Rabbit.pth"),
    "freshwater_fish": os.path.join(MODEL_DIR, "Freshwate Fish.pth"),
    "poultry": os.path.join(MODEL_DIR, "Poultry.pth"),
    "horse": os.path.join(MODEL_DIR, "Horse.pth"),
    "goat": os.path.join(MODEL_DIR, "Goat.pth"),
    "donkeys": os.path.join(MODEL_DIR, "Donkeys.pth"),
    "cat": os.path.join(MODEL_DIR, "Cat.pth"),
    "cow": os.path.join(MODEL_DIR, "Cow.pth"),
    "dogs": os.path.join(MODEL_DIR, "Dogs.pth")
}

# Load environment variables
GEMINI_API_KEY = ""
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise RuntimeError("Gemini API configuration failed")

# Load secondary models for skin condition prediction
secondary_models = {}

def load_model(model_path: str, num_classes: int):
    """Load a ViT model from a .pth file."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None

def initialize_models():
    """Initialize secondary ViT models for skin condition prediction."""
    global secondary_models
    for animal in ANIMAL_CLASSES:
        animal_key = animal.lower()
        model_path = SECONDARY_MODELS.get(animal_key)
        num_classes = len(SKIN_CLASSES.get(animal_key, []))
        if num_classes == 0:
            logger.warning(f"No skin classes defined for {animal}")
            secondary_models[animal_key] = None
            continue
        if model_path and os.path.exists(model_path):
            model = load_model(model_path, num_classes)
            secondary_models[animal_key] = model
            if model is None:
                logger.warning(f"Failed to load model for {animal} at {model_path}")
            else:
                logger.info(f"Model for {animal} loaded from {model_path}")
        else:
            logger.warning(f"No model defined or file missing for {animal} at {model_path}")
            secondary_models[animal_key] = None

initialize_models()

def predict_image(image_path, model, class_names):
    """Predict class using the ViT model."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)
        top_class = torch.argmax(probs, dim=1).item()
    
    return class_names[top_class], probs[0][top_class].item()

def predict_animal(image: Image.Image):
    """Predict animal using the ViT model."""
    try:
        # Save the PIL image to a temporary file
        temp_image_path = os.path.join(UPLOAD_DIR, "temp_animal_image.jpg")
        image.save(temp_image_path, format="JPEG")

        # Use the predict_image function
        predicted_class, confidence = predict_image(temp_image_path, model, class_names)

        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return predicted_class, None, confidence
    except Exception as e:
        logger.error(f"Error predicting animal: {str(e)}")
        return None, f"Prediction failed: {str(e)}", 0.0

def predict_skin_condition(image: Image.Image, animal: str):
    """Predict skin condition using the ViT model for the given animal."""
    animal_key = animal.lower()
    model = secondary_models.get(animal_key)
    if model is None:
        return None, False, 0.0, f"No model available for {animal}"
    try:
        # Save the PIL image to a temporary file
        temp_image_path = os.path.join(UPLOAD_DIR, f"temp_{animal_key}_image.jpg")
        image.save(temp_image_path, format="JPEG")

        # Get skin condition classes
        skin_classes = SKIN_CLASSES.get(animal_key, [])
        if not skin_classes:
            return None, False, 0.0, f"No skin condition classes defined for {animal}"

        # Predict using the ViT model
        condition, confidence = predict_image(temp_image_path, model, skin_classes)

        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        model_filename = SECONDARY_MODELS.get(animal_key, "")
        is_substring = check_substring(condition, model_filename)
        return condition, is_substring, confidence, None
    except Exception as e:
        logger.error(f"Error predicting skin condition for {animal}: {str(e)}")
        return None, False, 0.0, f"Prediction failed: {str(e)}"

def check_substring(predicted_class: str, model_filename: str):
    """Check if the predicted class is a substring of the model filename."""
    if not model_filename:
        return False
    cleaned_class = predicted_class.lower().replace(" ", "").replace("_", "")
    model_name = os.path.basename(model_filename).lower().replace(".pth", "")
    return cleaned_class in model_name

def get_gemini_response(message: str, animal: str = None, condition: str = None):
    """Generate a response using Gemini model."""
    try:
        prompt = ""
        if animal and condition:
            prompt = (
                f"The user asked: '{message}'. Based on an image analysis, the animal is a {animal} "
                f"with a predicted skin condition of {condition}.\n\n"
                "Respond with:\n"
                "- A short title for the condition\n"
                "- A brief summary of the condition\n"
                "- Possible causes\n"
                "- Common symptoms\n"
                "- Recommended actions (home care or vet visit)\n"
                "- Answer the user query in a concise and clear manner\n"
                "Use bullet points or numbered lists where possible. Avoid long paragraphs."
            )
        elif message:
            prompt = (
                f"The user asked: '{message}'.\n\n"
                "Provide a helpful and easy-to-understand answer. "
                "Use structured formatting (like bullet points) and avoid long paragraphs."
            )
        else:
            prompt = (
                "No specific question or image provided.\n\n"
                "Provide general advice on pet skin health in a structured format with headers or bullet points."
            )

        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating Gemini response: {str(e)}")
        return f"Failed to generate response:"


@app.post("/api/chat")
async def chat(
    message: str = Form(None),  # Optional text message
    image: UploadFile = File(None)  # Optional image file
):
    """Handle image uploads and messages, returning predictions and Gemini response."""
    try:
        response_text = ""
        image_path = None
        animal_prediction = None
        animal_confidence = 0.0
        skin_condition = None
        skin_confidence = 0.0
        substring_match = False
        gemini_response = ""

        if message:
            response_text += f"Received message: {message}. "

        if image:
            image_filename = os.path.join(UPLOAD_DIR, image.filename)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            image_bytes = await image.read()
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            image_path = image_filename

            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Predict animal
                animal, animal_error, animal_conf = predict_animal(img)
                if animal_error:
                    response_text += f"Animal prediction failed: {animal_error}. "
                else:
                    animal_prediction = animal
                    animal_confidence = animal_conf
                    response_text += f"Predicted animal: {animal} (confidence: {animal_conf:.2f}). "

                # Predict skin condition
                if animal:
                    condition, is_substring, condition_conf, condition_error = predict_skin_condition(img, animal)
                    if condition_error:
                        response_text += f"Skin condition prediction failed: {condition_error}. "
                    else:
                        skin_condition = condition
                        skin_confidence = condition_conf
                        substring_match = is_substring
                        response_text += f"Predicted skin condition: {condition} (confidence: {condition_conf:.2f}). "
                        if is_substring:
                            response_text += f"Note: The predicted condition '{condition}' matches part of the model filename for {animal}. "
                        response_text += "Consult a vet for confirmation."
            finally:
                if os.path.exists(image_filename):
                    os.remove(image_filename)

        # Generate Gemini response
        gemini_response = get_gemini_response(message, animal_prediction, skin_condition)
        response_text += f"\nAI Assistant: {gemini_response}"

        if not message and not image:
            response_text = "No input provided. Please send a message or image to analyze your pet's skin condition."
            gemini_response = get_gemini_response("")

        return JSONResponse({
            "status": "success",
            "response": response_text,
            "imagePath": image_path,
            "animalPrediction": animal_prediction,
            "animalConfidence": float(animal_confidence),
            "skinCondition": skin_condition,
            "skinConfidence": float(skin_confidence),
            "substringMatch": substring_match,
            "geminiResponse": gemini_response
        })

    except Exception as e:
        logger.exception("Error in /api/chat endpoint")
        return JSONResponse({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }, status_code=500)

def find_available_port(start_port=8000, max_port=8100):
    """Find an available port within the given range."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    logger.error(f"No available ports found between {start_port} and {max_port}")
    return None

if __name__ == "__main__":
    import uvicorn
    port = find_available_port()
    if port is None:
        logger.error("Could not find an available port. Please free up ports or adjust the range.")
        exit(1)
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
