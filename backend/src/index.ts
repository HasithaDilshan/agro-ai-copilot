import * as functions from "firebase-functions";
import * as admin from "firebase-admin";

// Initialize Firebase Admin SDK (only once)
// The emulator will handle the credentials automatically in local dev
if (!admin.apps.length) {
  admin.initializeApp();
}

// This function acts as a mock for Module 1's inference trigger
// It's an HTTPS callable function for simplicity in this mock
export const processPlantImage = functions.https.onCall(async (data: any, context) => {
  const imageUrl = data.imageUrl as string;
  // const imagePath = data.imagePath as string; // Path in storage, if needed

  if (!imageUrl) {
    throw new functions.https.HttpsError('invalid-argument', 'The image URL is required.');
  }

  functions.logger.info(`Received image for processing: ${imageUrl}`);

  // --- MOCK AI INFERENCE (Module 1) ---
  // In a real scenario, you'd trigger a TFLite inference or a cloud AI model here.
  // For now, let's simulate a diagnosis.
  let mockDiagnosis = "Healthy plant with good growth.";
  let mockConfidence = 0.95;

  // Simulate some mock logic based on the image URL or other factors
  if (imageUrl.includes("disease")) { // Just a simple string check for mock
    mockDiagnosis = "Early stage fungal infection detected. Recommend immediate treatment.";
    mockConfidence = 0.78;
  } else if (imageUrl.includes("pest")) {
    mockDiagnosis = "Signs of pest infestation. Consider organic pest control.";
    mockConfidence = 0.85;
  } else if (Math.random() > 0.7) { // Randomly simulate a "stress"
    mockDiagnosis = "Mild nutrient deficiency observed. Recommend soil test.";
    mockConfidence = 0.65;
  }

  functions.logger.info(`Mock diagnosis generated: ${mockDiagnosis}`);

  // --- Store Diagnosis in Firestore (Simulating Module 2 output) ---
  const diagnosisRef = admin.firestore().collection('diagnoses');
  const newDiagnosisDoc = await diagnosisRef.add({
    imageUrl: imageUrl,
    mockDiagnosis: mockDiagnosis,
    mockConfidence: mockConfidence,
    timestamp: admin.firestore.FieldValue.serverTimestamp(),
    // Potentially add fields for Module 3 insights later:
    // weatherConditions: {},
    // soilProperties: {},
    // stressScore: 0.0,
  });

  functions.logger.info(`Diagnosis stored in Firestore with ID: ${newDiagnosisDoc.id}`);

  // Return the document ID so the app can fetch the result
  return { diagnosisId: newDiagnosisDoc.id, status: 'success' };
});

// Optional: Add a Cloud Storage trigger function, though for this mock flow
// an HTTPS callable function is simpler to demonstrate the sequence.
// exports.onImageUpload = functions.storage.object().onFinalize(async (object) => {
//   // This would be triggered when an image is uploaded to Storage
//   // You could then call your AI processing here.
//   if (object.name && object.contentType?.startsWith('image/')) {
//     functions.logger.info(`New image uploaded: ${object.name}`);
//     // Call the processing logic directly or an internal callable function
//   }
//   return null;
// });