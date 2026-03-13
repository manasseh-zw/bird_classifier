# Implementation Breakdown

This document explains how the bird classification project works from first principles, then connects those ideas to the actual code in the notebook and the Streamlit app.

## 1. The Core Idea

The goal of this project is to take an input image of a bird and predict which one of 525 bird species it belongs to.

At a high level, the system does four things:

1. Read bird images from a labeled dataset.
2. Convert each image into numbers that a computer can process.
3. Train a neural network to connect image patterns to bird species labels.
4. Use the trained model inside a small web app for local inference.

## 2. What A Computer Sees In An Image

Humans see a bird, a beak, feathers, and color patterns.
A computer does not see those concepts directly.

An image is stored as a grid of pixels.
Each pixel contains numeric color values, usually in three channels:

- Red
- Green
- Blue

For example, one pixel might be represented as:

```text
[R=120, G=180, B=210]
```

A full image is therefore a large collection of numbers.
Deep learning works by learning patterns inside those numbers.

In image classification, the model tries to learn which visual patterns usually appear for each class.
Examples:

- a curved beak
- a long neck
- white plumage
- wing markings
- head shape

These low-level and high-level visual features help the model decide which bird species is most likely present in the image.

## 3. Why We Need A Neural Network

It is very difficult to write rules by hand for 525 bird species.

For example, a rule-based approach would need to answer questions like:

- How long is the beak?
- What if the bird is turned sideways?
- What if the lighting is poor?
- What if two species look almost identical?

Instead of hard-coding rules, we train a neural network.

A neural network learns from examples.
During training, it repeatedly sees:

- an input image
- the correct label for that image

Over time, it adjusts its internal weights so that its predictions become more accurate.

## 4. Why Transfer Learning Was Used

This project uses **transfer learning** with **ResNet18**.

Transfer learning means we start with a model that has already learned useful visual features from a very large dataset, then adapt it to our own problem.

This is better than training from scratch because:

- it needs less training time
- it usually gives better accuracy
- it already understands general visual patterns such as edges, textures, and shapes

In this project, the early layers of ResNet18 act like a strong visual feature extractor, and the final layer is replaced so the model can predict exactly **525 bird classes**.

## 5. Notebook Workflow

The main experimental and training work happens in `bird_classifer.ipynb`.

The notebook follows a standard deep learning pipeline.

### 5.1 Data Preparation

The notebook first prepares the dataset and points to the train, validation, and test folders.

These splits serve different purposes:

- **train**: used to learn the model weights
- **validation**: used during training to monitor performance
- **test**: used at the end to measure final performance

### 5.2 Image Transforms

Before images are passed into the model, they are transformed into the format the network expects.

The notebook applies:

- `ToTensor()`
- `Resize((224, 224), antialias=True)`
- ImageNet normalization with:
  - mean = `[0.485, 0.456, 0.406]`
  - std = `[0.229, 0.224, 0.225]` 

Why this matters:

- `ToTensor()` converts image data into PyTorch tensors
- `Resize((224, 224))` makes all images the same size
- normalization scales pixel values into a range that works well with a pre-trained ResNet model

### 5.3 Dataset Loading With ImageFolder

The notebook uses `torchvision.datasets.ImageFolder`.

This is important because `ImageFolder` automatically:

- reads images from folders
- treats each folder name as a class label
- assigns each class an integer index

Those integer indices are what the model actually learns to predict.

The class order is based on the folder names in alphabetical order.
This detail becomes very important later when mapping predicted indices back to species names.

### 5.4 DataLoader

The notebook then wraps the datasets in `DataLoader` objects.

`DataLoader` helps by:

- loading images in batches
- shuffling training data
- feeding data efficiently into the model

Training in batches is much more efficient than processing one image at a time.

### 5.5 Model Creation

The notebook loads:

```python
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

This gives a pre-trained ResNet18 model.

Then the final fully connected layer is replaced so that the network outputs **525 values**, one for each bird species.

Conceptually:

- earlier layers learn visual features
- the final layer converts those features into class scores

### 5.6 Loss Function And Optimizer

The notebook uses:

- `nn.CrossEntropyLoss()`
- `torch.optim.Adam(...)`

`CrossEntropyLoss` measures how far the model's prediction is from the correct class.
The optimizer then updates the weights to reduce that error.

This repeated update process is the core of learning.

### 5.7 Training Loop

The training loop repeatedly does the following:

1. load a batch of images and labels
2. run a forward pass through the model
3. compute the loss
4. clear old gradients
5. run backpropagation
6. update the weights

This loop is repeated for multiple epochs.

An **epoch** means one full pass through the training dataset.

During training, the notebook also tracks:

- training loss
- validation loss
- training accuracy
- validation accuracy

These metrics help show whether the model is learning well.

### 5.8 Testing And Saving

After training, the notebook evaluates the final model on the test set.
It then saves the trained weights using `torch.save(...)`.

That saved `.pth` file is what the Streamlit app later loads for inference.

## 6. How Prediction Works

When the trained model receives an image, it outputs a vector of 525 scores.

Each score represents how strongly the model believes the image belongs to one class.

Those raw scores are called **logits**.

To convert logits into interpretable probabilities, we apply **softmax**.
Softmax turns the 525 scores into values between 0 and 1 that sum to 1.

Then we choose the class with the highest probability.

So the prediction process is:

1. preprocess the image
2. run the model
3. compute softmax probabilities
4. take the highest-probability class index
5. map that index back to the bird species name

## 7. Why The Label Mapping Needed Care

The app does not only need a trained model.
It also needs a reliable way to convert predicted class indices into human-readable bird names.

The raw `app/birds.csv` file contains repeated rows for many images, so it is too large and repetitive to use directly in the UI.

More importantly, the app must use the same label order that the notebook used during training.

Because training used `ImageFolder`, the real model output order is based on **alphabetical class order**, not just the original `class id` column from the CSV.

This is why `app/generate_bird_labels.py` exists.

## 8. What `generate_bird_labels.py` Does

The script in `app/generate_bird_labels.py` performs three main jobs:

### 8.1 Load And Deduplicate

It reads `app/birds.csv` and removes repeated entries by keeping one label per `class id`.

### 8.2 Rebuild Model Order

It sorts the unique labels alphabetically so the output order matches the way `ImageFolder` assigned class indices during training.

It then writes:

- `model_index`: the index actually used by the model
- `class_id`: the original dataset class id
- `label`: the bird species name

### 8.3 Validate The Mapping

The script checks that:

- there are exactly 525 classes
- `model_index` covers `0` to `524`
- `class_id` also covers `0` to `524`

This prevents a silent mismatch between the model and the UI.

## 9. How The Streamlit App Works

The deployment code lives in `app/main.py`.

Its job is not to train the model.
Its job is to load the already-trained model and use it to make predictions on new images.

### 9.1 File Paths And Constants

The app defines:

- the path to the model weights
- the path to `bird_labels.csv`
- the total class count

This keeps the code organized and avoids hard-coded values scattered throughout the file.

### 9.2 Preprocessing

The app uses the same preprocessing pipeline as the notebook:

- `ToTensor()`
- `Resize((224, 224), antialias=True)`
- ImageNet normalization

This consistency is critical.
If preprocessing at inference time is very different from preprocessing during training, accuracy will usually drop.

### 9.3 Cached Resource Loading

The app uses `@st.cache_resource` for:

- `load_label_mapping()`
- `load_model()`

This means Streamlit loads these heavy resources once and reuses them, instead of reloading them on every interaction.

That makes the UI much faster, especially because model weights are relatively large compared to normal UI state.

### 9.4 Model Reconstruction

Before loading the saved weights, the app rebuilds the ResNet18 architecture and replaces the final layer with a `Linear` layer that outputs 525 classes.

Only after rebuilding the correct structure can the saved state dictionary be loaded successfully.

### 9.5 Inference

`predict_bird(...)` performs the actual prediction:

1. load the model
2. load the label mapping
3. convert the uploaded image to RGB
4. preprocess the image
5. add a batch dimension with `unsqueeze(0)`
6. run inference with gradients disabled
7. compute softmax probabilities
8. return the best label and confidence score

Using `torch.inference_mode()` is efficient because the app is only doing prediction, not training.

### 9.6 User Interface

The Streamlit UI is intentionally minimal:

- a header
- a new upload button
- an upload panel
- a result panel

The uploaded image is previewed on the left.
The predicted species and confidence are shown on the right.

Custom CSS is used to simplify the default Streamlit look so the app feels more like a clean demo interface.

## 10. Why This Design Works Well

This project separates responsibilities clearly:

- the notebook handles experimentation, training, and evaluation
- the helper script prepares a correct label mapping for deployment
- the Streamlit app focuses only on inference and presentation

This is good software design because each component has one clear responsibility.

## 11. End-To-End Flow

From start to finish, the system works like this:

1. the notebook trains a ResNet18 model on bird images
2. the notebook saves the trained weights as a `.pth` file
3. the helper script converts the raw bird metadata into a clean model-aligned label mapping
4. the Streamlit app loads both the model weights and the label mapping
5. the user uploads a new image
6. the app preprocesses the image and runs inference
7. the app displays the predicted bird species and confidence

## 12. Final Summary

This project is an example of taking a machine learning experiment and turning it into a usable software product.

From a computer science point of view, it combines:

- image representation
- deep learning
- transfer learning
- model evaluation
- data preprocessing
- deployment
- user interface design

The result is a complete bird species classification system that is understandable, reproducible, and ready for demonstration.