import 'dart:io';

import 'package:flutter/foundation.dart'; // Import for kDebugMode
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:cloud_functions/cloud_functions.dart';
import 'package:image_picker/image_picker.dart'; // Add this package later for actual image picker

// Ensure you have a firebase_options.dart from `flutterfire configure`
// If not, run `flutterfire configure` in your mobile-app directory.
import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  // --- IMPORTANT: CONFIGURE FIREBASE EMULATORS ---
  if (kDebugMode) {
    try {
      // Adjust IP for Android emulator (10.0.2.2) vs. iOS/Web/Desktop (localhost)
      final emulatorHost = defaultTargetPlatform == TargetPlatform.android
          ? "10.0.2.2"
          : "localhost";

      FirebaseFirestore.instance.useFirestoreEmulator(emulatorHost, 8080);
      FirebaseStorage.instance.useStorageEmulator(emulatorHost, 9199);
      FirebaseFunctions.instance.useFunctionsEmulator(emulatorHost, 5001);
      print('--- Using Firebase Emulators ---');
    } catch (e) {
      print('Error configuring Firebase Emulators: $e');
    }
  }
  // --- END EMULATOR CONFIG ---

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgroAI Co-Pilot',
      theme: ThemeData(primarySwatch: Colors.green),
      home: ImageCaptureScreen(), // Our new screen for the mock flow
    );
  }
}

// --- New Widget for Image Capture and Display ---
class ImageCaptureScreen extends StatefulWidget {
  @override
  _ImageCaptureScreenState createState() => _ImageCaptureScreenState();
}

class _ImageCaptureScreenState extends State<ImageCaptureScreen> {
  String? _imageUrl;
  String _diagnosisResult = "No diagnosis yet.";
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickAndUploadImage() async {
    setState(() {
      _isLoading = true;
      _diagnosisResult = "Processing image...";
    });

    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
    ); // Or ImageSource.gallery

    if (image != null) {
      try {
        // 1. Upload image to Firebase Storage
        final storageRef = FirebaseStorage.instance.ref().child(
          'plant_images/${DateTime.now().millisecondsSinceEpoch}.jpg',
        );
        await storageRef.putFile(File(image.path));
        _imageUrl = await storageRef.getDownloadURL();
        print('Image uploaded to: $_imageUrl');

        // 2. Call the Firebase Function (this will be the mock AI inference)
        final callable = FirebaseFunctions.instance.httpsCallable(
          'processPlantImage',
        );
        final result = await callable.call(<String, dynamic>{
          'imageUrl': _imageUrl,
          'imagePath': storageRef
              .fullPath, // Pass path for the function to know where to find it
        });

        // 3. Get diagnosis from Firestore (triggered by the function)
        // The function will create a document in 'diagnoses' collection
        final diagnosisDocId =
            result.data['diagnosisId']; // Function returns the doc ID
        if (diagnosisDocId != null) {
          final diagnosisSnapshot = await FirebaseFirestore.instance
              .collection('diagnoses')
              .doc(diagnosisDocId)
              .get();
          if (diagnosisSnapshot.exists) {
            setState(() {
              _diagnosisResult =
                  diagnosisSnapshot.data()?['mockDiagnosis'] ??
                  'Diagnosis data not found.';
            });
            print('Diagnosis retrieved from Firestore: $_diagnosisResult');
          } else {
            setState(() {
              _diagnosisResult = 'Diagnosis document not found in Firestore.';
            });
          }
        } else {
          setState(() {
            _diagnosisResult = 'Function did not return a diagnosis ID.';
          });
        }
      } catch (e) {
        print('Error during image upload or processing: $e');
        setState(() {
          _diagnosisResult = 'Error: ${e.toString()}';
        });
      } finally {
        setState(() {
          _isLoading = false;
        });
      }
    } else {
      setState(() {
        _isLoading = false;
        _diagnosisResult = "Image capture cancelled.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AgroAI Mock Co-Pilot')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _imageUrl != null
                  ? Image.network(
                      _imageUrl!,
                      height: 200,
                      width: 200,
                      fit: BoxFit.cover,
                    )
                  : Container(
                      height: 200,
                      width: 200,
                      color: Colors.grey[300],
                      child: Icon(
                        Icons.photo,
                        size: 100,
                        color: Colors.grey[600],
                      ),
                    ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isLoading ? null : _pickAndUploadImage,
                child: _isLoading
                    ? const CircularProgressIndicator(color: Colors.white)
                    : const Text('Capture & Get Mock Diagnosis'),
              ),
              const SizedBox(height: 20),
              Text(
                'Diagnosis: $_diagnosisResult',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: _diagnosisResult.startsWith('Error')
                      ? Colors.red
                      : Colors.green[800],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
