import functions_framework
import tensorflow as tf
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage # Import necessary modules

# --- Global variables for TFLite model and class names (loaded once for warm starts) ---
# These remain as global variables, but are initialized to None.
# They will be populated by _load_model_and_class_names() on the first function invocation.
interpreter = None
input_details = None
output_details = None
class_names = []

# --- Firebase Admin SDK Clients (also initialized lazily) ---
# These will hold the client instances (Firestore, Storage) after initialization.
# They are initialized to None in the global scope.
_db_client = None
_bucket_client = None

# --- Paths to TFLite model and class names in Firebase Storage ---
FIREBASE_STORAGE_BUCKET_NAME = "agroai-phoenix.firebasestorage.app" # Confirmed from your input
TFLITE_MODEL_GCS_PATH_FULL = f"gs://{FIREBASE_STORAGE_BUCKET_NAME}/ml_models/fp32_mvp_model.tflite"
CLASS_NAMES_GCS_PATH_FULL = f"gs://{FIREBASE_STORAGE_BUCKET_NAME}/ml_models/class_names.txt"


def _load_model_and_class_names():
    """
    Loads the TFLite model and class names from Firebase Storage if not already loaded.
    This function also handles lazy initialization of Firebase Admin SDK clients.
    Designed for warm starts: executes heavy ops only on the first invocation of an instance.
    """
    global interpreter, input_details, output_details, class_names, _db_client, _bucket_client

    # --- Lazy Firebase Admin SDK Initialization ---
    # This ensures firebase_admin.initialize_app() and client instantiation
    # only happen on the first function invocation for a given instance.
    if not firebase_admin._apps:
        print(f"[{os.getpid()}] Initializing Firebase Admin SDK...")
        # Cloud Functions environment automatically provides credentials.
        firebase_admin.initialize_app()
        print(f"[{os.getpid()}] Firebase Admin SDK initialized.")

    if _db_client is None:
        print(f"[{os.getpid()}] Initializing Firestore client...")
        _db_client = firestore.client()
        print(f"[{os.getpid()}] Firestore client initialized.")

    if _bucket_client is None:
        print(f"[{os.getpid()}] Initializing Storage bucket client...")
        _bucket_client = storage.bucket()
        print(f"[{os.getpid()}] Storage bucket client initialized.")


    # --- TFLite Model Loading ---
    if interpreter is None:
        print(f"[{os.getpid()}] Starting model download from {TFLITE_MODEL_GCS_PATH_FULL}...")
        try:
            model_path = tf.keras.utils.get_file(
                os.path.basename(TFLITE_MODEL_GCS_PATH_FULL), TFLITE_MODEL_GCS_PATH_FULL, cache_dir="/tmp/"
            )
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(f"[{os.getpid()}] TFLite model loaded and allocated.")
        except Exception as e:
            print(f"[{os.getpid()}] Failed to load TFLite model from GCS: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    # --- Class Names Loading ---
    if not class_names:
        print(f"[{os.getpid()}] Starting class names download from {CLASS_NAMES_GCS_PATH_FULL}...")
        try:
            class_names_local_path = tf.keras.utils.get_file(
                os.path.basename(CLASS_NAMES_GCS_PATH_FULL), CLASS_NAMES_GCS_PATH_FULL, cache_dir="/tmp/"
            )
            with open(class_names_local_path, 'r') as f:
                class_names = [line.strip() for line in f]
            print(f"[{os.getpid()}] Loaded {len(class_names)} class names.")
        except Exception as e:
            print(f"[{os.getpid()}] Failed to load class names from GCS: {e}")
            raise RuntimeError(f"Class names loading failed: {e}")


@functions_framework.http
def predict_plant_disease(request):
    """
    Cloud Function for Module 1 TFLite inference.
    Triggered by an HTTP request (called from Node.js orchestrator).
    """
    # Call the lazy loader on function invocation.
    # This is the FIRST time any heavy initialization will run for this instance.
    _load_model_and_class_names()

    # Now that clients are guaranteed to be initialized, use them.
    db = _db_client # Use the lazily initialized Firestore client
    # bucket_client = _bucket_client # If you needed the bucket_client directly here

    # Handle CORS preflight requests (important for web/mobile app calls)
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json(silent=True)
    if not request_json or 'imageUrl' not in request_json:
        print(f"[{os.getpid()}] Bad Request: Missing imageUrl in JSON body.")
        return ('{"error": "Missing imageUrl in request body"}', 400, headers)

    image_url = request_json['imageUrl']
    print(f"[{os.getpid()}] Received request for image URL: {image_url}")

    try:
        # Download image from the provided URL (could be Firebase Storage or any public URL)
        # tf.keras.utils.get_file handles downloading to /tmp/ and caching
        # Clean filename from URL query parameters (e.g., "?alt=media&token=...")
        img_local_path = tf.keras.utils.get_file(
            os.path.basename(image_url).split('?')[0],
            image_url,
            cache_dir="/tmp/"
        )

        # Read, decode, resize, and preprocess the image
        img = tf.io.read_file(img_local_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False) # Ensure 3 channels, no GIFs
        img = tf.image.resize(img, [224, 224]) # TFLite model input size (EfficientNetV2-B0 default)
        img = tf.cast(img, tf.float32) # Ensure float32 for model input (if not int8)
        img = np.expand_dims(img, axis=0) # Add batch dimension

        # Get input and output tensors details for TFLite interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Handle quantization if the model input type is INT8
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.int8:
            print(f"[{os.getpid()}] Input type is INT8, quantizing image for TFLite model...")
            input_scale, input_zero_point = input_details[0]["quantization"]
            img = (img / input_scale + input_zero_point).astype(input_dtype)
        elif input_dtype == np.float32:
            pass # No specific normalization for EfficientNetV2's internal preprocessing
        else:
            print(f"[{os.getpid()}] Warning: Unexpected TFLite input dtype: {input_dtype}")


        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Invoke inference
        interpreter.invoke()

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle dequantization if the model output type is INT8
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.int8:
            output_scale, output_zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        # Apply softmax if the model outputs logits (raw scores) - common for classification models
        predictions = tf.nn.softmax(output[0]).numpy()

        predicted_class_idx = np.argmax(predictions)
        predicted_confidence = np.max(predictions)

        diagnosis_result = {
            "class_name": class_names[predicted_class_idx],
            "confidence": float(predicted_confidence),
            "full_prediction_scores": predictions.tolist(), # Store all class probabilities
            "message": f"Detected: {class_names[predicted_class_idx]} with {predicted_confidence*100:.2f}% confidence."
        }

        print(f"[{os.getpid()}] Inference result: {diagnosis_result}")

        diagnosis_doc_id = f"diagnosis_{os.path.basename(image_url).split('?')[0].replace('.', '_')}_{tf.timestamp().numpy().astype(int)}"
        
        # Save relevant parts to Firestore using the lazily initialized client
        db.collection('diagnoses').document(diagnosis_doc_id).set({
            "imageUrl": image_url,
            "diagnosis": diagnosis_result,
            "timestamp": firestore.FieldValue.server_timestamp(),
            "module": "module1"
        })
        print(f"[{os.getpid()}] Diagnosis stored in Firestore with ID: {diagnosis_doc_id}")

        return (json.dumps({"diagnosis": diagnosis_result, "diagnosisId": diagnosis_doc_id}), 200, headers)

    except Exception as e:
        print(f"[{os.getpid()}] Error during image processing or inference: {e}")
        import traceback
        traceback.print_exc()
        return (json.dumps({"error": str(e), "message": "Failed to process image for diagnosis."}), 500, headers)