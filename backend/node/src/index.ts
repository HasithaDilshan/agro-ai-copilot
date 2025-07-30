import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import { https } from 'firebase-functions';
import fetch from 'node-fetch'; // For Node.js 20, use 'node-fetch' version 2.x

// Initialize Firebase Admin SDK
if (!admin.apps.length) {
  admin.initializeApp();
}

// IMPORTANT: Replace <YOUR_FIREBASE_PROJECT_ID> with your actual Firebase Project ID.
// Example: if your project ID is 'my-agro-app-123', replace it.
const YOUR_FIREBASE_PROJECT_ID = "agroai-phoenix"; // <--- CUSTOMIZE THIS!

// Main callable function triggered by the mobile app.
export const processPlantImage = functions.https.onCall(async (data: any, context) => {
  const imageUrl = data.imageUrl as string;

  if (!imageUrl) {
    throw new https.HttpsError('invalid-argument', 'Image URL is required.');
  }

  functions.logger.info(`Node.js orchestrator received image: ${imageUrl}`);

  try {
    // Determine the Python Inference Function URL dynamically.
    // This handles both local emulator and deployed environments.
    const pythonFunctionBaseUrl = process.env["FUNCTIONS_EMULATOR"] === 'true'
      ? `http://localhost:5001/${YOUR_FIREBASE_PROJECT_ID}/us-central1`
      : `https://us-central1-${YOUR_FIREBASE_PROJECT_ID}.cloudfunctions.net`;

    const pythonFunctionUrl = `${pythonFunctionBaseUrl}/predict_plant_disease`;

    functions.logger.info(`Calling Python inference function at: ${pythonFunctionUrl}`);

    const response = await fetch(pythonFunctionUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageUrl: imageUrl })
    });

    if (!response.ok) {
        const errorText = await response.text();
        functions.logger.error(`Python inference failed (status ${response.status}): ${errorText}`);
        throw new https.HttpsError('internal', `Python inference failed: ${response.status} - ${errorText}`);
    }

    const pythonResult = await response.json();
    functions.logger.info(`Python inference successful. Result: ${JSON.stringify(pythonResult)}`);

    return pythonResult;

  } catch (e: any) {
    functions.logger.error("Error in Node.js orchestrator:", e);
    throw new https.HttpsError('internal', `Failed to process image: ${e.message || e}`);
  }
});