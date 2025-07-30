import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgroAI Co-Pilot',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const ImageClassifier(),
    );
  }
}

class ImageClassifier extends StatefulWidget {
  const ImageClassifier({super.key});

  @override
  State<ImageClassifier> createState() => _ImageClassifierState();
}

class _ImageClassifierState extends State<ImageClassifier> {
  final String _modelPath = 'assets/fp32_mvp_model.tflite'; // Fixed path
  final String _labelsPath = 'assets/class_names.txt'; // Fixed path

  tfl.Interpreter? _interpreter;
  List<String>? _labels;
  File? _image;
  String _classificationResult = '';
  bool _isLoading = true;

  // Expected model input dimensions
  // CRITICAL: Update these to match your EfficientNetV2-B0 model's training configuration.
  final int _inputSize = 224;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  /// Loads the TFLite model and labels from the assets folder.
  Future<void> _loadModel() async {
    try {
      // Create the interpreter from the asset
      _interpreter = await tfl.Interpreter.fromAsset(_modelPath);

      // Load the labels from the asset
      final labelsData = await rootBundle.loadString(_labelsPath);
      _labels = labelsData
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      print("Model loaded successfully!");
      print("Number of classes: ${_labels!.length}");
      print("Model expects 38 outputs, we have ${_labels!.length} labels");
      print("First few classes: ${_labels!.take(3).join(', ')}");
      print("Last few classes: ${_labels!.skip(_labels!.length - 3).join(', ')}");

      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      print("Error loading model: $e");
      setState(() {
        _isLoading = false;
        _classificationResult = "Failed to load model.";
      });
    }
  }

  /// Preprocesses the image to the format expected by the model.
  Float32List _preprocessImage(img.Image image) {
    // 1. Resize the image
    final resizedImage = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
    );

    // 2. Convert to Float32List and normalize to [0, 1] for EfficientNet
    final float32List = Float32List(_inputSize * _inputSize * 3);
    int pixelIndex = 0;
    
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        // Normalize pixel values to [0, 1] range
        float32List[pixelIndex++] = pixel.r / 255.0;
        float32List[pixelIndex++] = pixel.g / 255.0;
        float32List[pixelIndex++] = pixel.b / 255.0;
      }
    }
    
    return float32List;
  }

  /// Runs inference on the provided image file.
  Future<void> _runInference(File imageFile) async {
    if (_interpreter == null || _labels == null) {
      print("Interpreter or labels not loaded.");
      return;
    }

    setState(() {
      _isLoading = true;
    });

    // Read image file as bytes and decode it
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);

    if (image == null) {
      return;
    }

    // Preprocess the image
    final preprocessedImage = _preprocessImage(image);
    
    // Reshape into 4D tensor [1, height, width, channels]
    final input = List.generate(1, (batch) => 
      List.generate(_inputSize, (y) => 
        List.generate(_inputSize, (x) => 
          List.generate(3, (c) => 
            preprocessedImage[(y * _inputSize + x) * 3 + c]
          )
        )
      )
    );

    // Define the model's output - let's use a flexible approach
    final output = [List<double>.filled(38, 0.0)]; // Use 38 since model outputs [1, 38]

    // Run inference
    _interpreter!.run(input, output);

    // Print raw model output for debugging
    print("Raw model output:");
    print("Output shape: [1, ${output[0].length}]");
    print("Output values: ${output[0]}");
    print("Sum of probabilities: ${output[0].reduce((a, b) => a + b)}");

    // Process the output - this is now a list of probabilities
    final probabilities = output[0];

    // Find the index with the highest probability
    double maxScore = 0;
    int maxIndex = -1;
    for (int i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxScore) {
        maxScore = probabilities[i];
        maxIndex = i;
      }
    }

    setState(() {
      _image = imageFile;
      if (maxIndex != -1 && maxIndex < _labels!.length) {
        // Your model outputs probabilities (0.0 to 1.0), so convert to percentage
        final confidence = (maxScore * 100).toStringAsFixed(1);
        _classificationResult = "'${_labels![maxIndex]}' ($confidence%)";
        print("Classification result: ${_labels![maxIndex]} with $confidence% confidence");
      } else {
        _classificationResult = "Could not classify image.";
        print("Classification failed - maxIndex: $maxIndex, labels length: ${_labels!.length}");
      }
      _isLoading = false;
    });
  }

  /// Handles picking an image from the gallery or camera.
  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      _runInference(File(pickedFile.path));
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("AgroAI - Disease Recognition")),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              if (_isLoading && _image == null)
                const CircularProgressIndicator()
              else if (_image != null)
                Container(
                  margin: const EdgeInsets.all(16.0),
                  constraints: BoxConstraints(
                    maxHeight: MediaQuery.of(context).size.height * 0.4,
                  ),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.green.shade300, width: 2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Image.file(_image!),
                  ),
                )
              else
                Text(
                  'Capture or select an image to classify',
                  style: Theme.of(context).textTheme.headlineSmall,
                  textAlign: TextAlign.center,
                ),
              const SizedBox(height: 20),
              if (_isLoading && _image != null)
                const Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 10),
                    Text("Classifying..."),
                  ],
                ),
              if (!_isLoading && _classificationResult.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0),
                  child: Text(
                    _classificationResult,
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                ),
              const SizedBox(height: 30),
            ],
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            FloatingActionButton.extended(
              heroTag: 'camera_fab',
              onPressed: () => _pickImage(ImageSource.camera),
              label: const Text("Camera"),
              icon: const Icon(Icons.camera_alt),
            ),
            FloatingActionButton.extended(
              heroTag: 'gallery_fab',
              onPressed: () => _pickImage(ImageSource.gallery),
              label: const Text("Gallery"),
              icon: const Icon(Icons.photo_library),
            ),
          ],
        ),
      ),
    );
  }
}
