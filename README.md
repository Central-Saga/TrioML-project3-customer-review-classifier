# Customer Review Classification

An Android application for classifying customer reviews into positive and negative sentiment using IndoBERT and ONNX Runtime.

The application performs model inference locally on the Android device, without requiring a prediction API.

## Features

- Classifies customer reviews as positive or negative.
- Displays the predicted sentiment and confidence score.
- Uses an IndoBERT model exported to ONNX and quantized to INT8.
- Performs inference offline after the application and model are installed.

## Project Structure

```text
CustomerReviewClassifier/
+-- android/
¦   +-- app/
¦   ¦   +-- src/main/
¦   ¦       +-- assets/
¦   ¦       ¦   +-- model.onnx
¦   ¦       ¦   +-- vocab.txt
¦   ¦       ¦   +-- tokenizer.json
¦   ¦       ¦   +-- ...
¦   ¦       +-- java/com/trio/customerreview/
¦   ¦           +-- MainActivity.kt
¦   ¦           +-- OnnxPredictor.kt
¦   ¦           +-- TextTokenizer.kt
¦   +-- gradle/
¦   +-- build.gradle.kts
¦   +-- settings.gradle.kts
+-- ml_training/
¦   +-- notebooks/
¦   ¦   +-- customer_review_classification_ml_pipeline.ipynb
¦   +-- README.md
+-- .gitattributes
+-- .gitignore
```

## Machine Learning Pipeline

The training notebook documents the following workflow:

1. Exploratory Data Analysis (EDA).
2. Text classification baseline using TF-IDF and Logistic Regression.
3. IndoBERT fine-tuning for sentiment classification.
4. Model evaluation.
5. Export to ONNX and INT8 quantization.
6. Preparation of the model and tokenizer for Android deployment.

For details, see [`ml_training/README.md`](ml_training/README.md).

## Android Application

### Requirements

- Android Studio
- Android SDK compatible with the project configuration
- Git LFS to download the ONNX model from the repository

### Run the application

1. Clone the repository and switch to the project branch.
2. Download the Git LFS files.
3. Open the `android/` directory in Android Studio.
4. Wait for Gradle Sync to finish.
5. Select an emulator or Android device.
6. Build and run the `app` configuration.

### How inference works

```text
Customer review
      |
      v
TextTokenizer.kt
      |
      v
input_ids, attention_mask, token_type_ids
      |
      v
OnnxPredictor.kt
      |
      v
ONNX INT8 model
      |
      v
Positive / Negative + confidence
```

The Android application runs inference locally and does not require FastAPI for sentiment prediction.

## Git LFS

The ONNX model is stored using Git LFS because of its file size.

After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

## Repository Branch

`feature/android-ml-integration`
