"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.processPlantImage = void 0;
var functions = require("firebase-functions");
var admin = require("firebase-admin");
// Initialize Firebase Admin SDK (only once)
// The emulator will handle the credentials automatically in local dev
if (!admin.apps.length) {
    admin.initializeApp();
}
// This function acts as a mock for Module 1's inference trigger
// It's an HTTPS callable function for simplicity in this mock
exports.processPlantImage = functions.https.onCall(function (data, context) { return __awaiter(void 0, void 0, void 0, function () {
    var imageUrl, mockDiagnosis, mockConfidence, diagnosisRef, newDiagnosisDoc;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                imageUrl = data.imageUrl;
                // const imagePath = data.imagePath as string; // Path in storage, if needed
                if (!imageUrl) {
                    throw new functions.https.HttpsError('invalid-argument', 'The image URL is required.');
                }
                functions.logger.info("Received image for processing: ".concat(imageUrl));
                mockDiagnosis = "Healthy plant with good growth.";
                mockConfidence = 0.95;
                // Simulate some mock logic based on the image URL or other factors
                if (imageUrl.includes("disease")) { // Just a simple string check for mock
                    mockDiagnosis = "Early stage fungal infection detected. Recommend immediate treatment.";
                    mockConfidence = 0.78;
                }
                else if (imageUrl.includes("pest")) {
                    mockDiagnosis = "Signs of pest infestation. Consider organic pest control.";
                    mockConfidence = 0.85;
                }
                else if (Math.random() > 0.7) { // Randomly simulate a "stress"
                    mockDiagnosis = "Mild nutrient deficiency observed. Recommend soil test.";
                    mockConfidence = 0.65;
                }
                functions.logger.info("Mock diagnosis generated: ".concat(mockDiagnosis));
                diagnosisRef = admin.firestore().collection('diagnoses');
                return [4 /*yield*/, diagnosisRef.add({
                        imageUrl: imageUrl,
                        mockDiagnosis: mockDiagnosis,
                        mockConfidence: mockConfidence,
                        timestamp: admin.firestore.FieldValue.serverTimestamp(),
                        // Potentially add fields for Module 3 insights later:
                        // weatherConditions: {},
                        // soilProperties: {},
                        // stressScore: 0.0,
                    })];
            case 1:
                newDiagnosisDoc = _a.sent();
                functions.logger.info("Diagnosis stored in Firestore with ID: ".concat(newDiagnosisDoc.id));
                // Return the document ID so the app can fetch the result
                return [2 /*return*/, { diagnosisId: newDiagnosisDoc.id, status: 'success' }];
        }
    });
}); });
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
