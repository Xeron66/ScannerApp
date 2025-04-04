import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart'; // Import for tflite_flutter
import 'package:image/image.dart' as img; // For image manipulation
import 'dart:typed_data'; // For working with Uint8List

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Scanner',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ScannerScreen(),
    );
  }
}

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  File? _imageFile;
  String _extractedText = "";
  bool _isProcessing = false;
  late Interpreter _interpreter; // TensorFlow Lite interpreter

  @override
  void initState() {
    super.initState();
    _loadModel(); // Load the model when the app starts
  }

  // Load the TFLite model
  Future<void> _loadModel() async {
    try {
      print("Loading model...");
      _interpreter = await Interpreter.fromAsset('assets/best-fp16.tflite'); //mnist_mobilenetv2.tflite
      print("Model loaded successfully.");

      // Print input and output tensor details
      var inputTensors = _interpreter.getInputTensors();
      var outputTensors = _interpreter.getOutputTensors();
      print("Input Tensors: $inputTensors");
      print("Output Tensors: $outputTensors");
    } catch (e) {
      print("Failed to load model: $e");
      setState(() {
        _isProcessing = false;
        _extractedText = "Failed to load model: $e";
      });
    }
  }

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _extractedText = ""; // Clear previous text
        _isProcessing = true;
      });
      await _processImage();
    }
  }

// Function to process the images!
  Future<void> _processImage() async {
    if (_imageFile == null || _interpreter == null) {
      print("Error: Image file or interpreter is null.");
      setState(() {
        _isProcessing = false;
        _extractedText = "Error: Image file or interpreter is null.";
      });
      return;
    }

    print("Reading image file...");
    final imageBytes = await _imageFile!.readAsBytes();
    print("Image file read successfully.");

    print("Decoding image...");
    final image = img.decodeImage(Uint8List.fromList(imageBytes));
    if (image == null) {
      print("Error: Failed to decode image.");
      setState(() {
        _isProcessing = false;
        _extractedText = "Error: Failed to decode image.";
      });
      return;
    }
    print("Image decoded successfully.");

    print("Resizing and preprocessing image...");
    var input = imageToByteListFloat32(image, 640); // Resize to 640x640
    print("Image resized and preprocessed successfully.");

    print("Running inference...");
    try {
      // Reshape the input to match the model's expected shape [1, 640, 640, 3]
      var inputTensor = input.reshape([1, 640, 640, 3]);

      // Prepare the output tensor
      var output = Float32List(1 * 25200 * 15); // Assuming output shape [1, 25200, 15]
      var outputTensor = output.reshape([1, 25200, 15]);

      // Run inference
      _interpreter.run(inputTensor, outputTensor);
      print("Inference successful.");

      // Extract and display the top predictions
      String result = "Top Predictions:\n";
      for (int i = 0; i < 5; i++) { // Display top 5 predictions
        result += "Prediction $i: ${outputTensor[0][i]}\n";
      }
      print("Inference result: $result");

      // Get the result
      setState(() {
        _isProcessing = false;
        _extractedText = result; // Display the top predictions
      });
    } catch (e) {
      print("Inference failed: $e");
      setState(() {
        _isProcessing = false;
        _extractedText = "Inference failed: $e";
      });
    }
  }

  Float32List imageToByteListFloat32(img.Image image, int inputSize) {
    // Create a Float32List to store the normalized pixel values
    var convertedBytes = Float32List(inputSize * inputSize * 3);

    // Resize the image to match the model's input dimensions
    final resizedImage = img.copyResize(image, width: inputSize, height: inputSize);

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        // Get the pixel at (x, y) as a Pixel object
        img.Pixel pixel = resizedImage.getPixel(x, y);

        // Extract the red, green, and blue values
        int r = pixel.r.toInt(); // Red channel
        int g = pixel.g.toInt(); // Green channel
        int b = pixel.b.toInt(); // Blue channel

        // Normalize the pixel values to the range [0, 1]
        double rNormalized = r / 255.0;
        double gNormalized = g / 255.0;
        double bNormalized = b / 255.0;

        // Store the normalized RGB values in the Float32List
        convertedBytes[(y * inputSize + x) * 3] = rNormalized;
        convertedBytes[(y * inputSize + x) * 3 + 1] = gNormalized;
        convertedBytes[(y * inputSize + x) * 3 + 2] = bNormalized;
      }
    }
    return convertedBytes;
  }



  @override
  void dispose() {
    _interpreter.close(); // Close the interpreter when no longer needed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scanner'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: _imageFile == null
                  ? const Center(
                child: Text(
                  'Select an image to analyze.',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
                ),
              )
                  : Column(
                children: [
                  Expanded(
                    child: Image.file(_imageFile!),
                  ),
                  const SizedBox(height: 10),
                  _isProcessing
                      ? const CircularProgressIndicator()
                      : Expanded(
                    child: SingleChildScrollView(
                      child: Text(
                        _extractedText.isEmpty ? 'No text found' : _extractedText,
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImage,
        tooltip: 'Pick Image',
        child: const Icon(Icons.add_a_photo),
      ),
    );
  }
}
