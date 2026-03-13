# Bird Species Classification

This repository contains a deep learning project for classifying 525 different bird species from images. It includes the experimental training pipeline as well as a production-ready Streamlit web application for running local inference.

## Dataset Attribution

This project uses the [chriamue/bird-species-dataset](https://huggingface.co/datasets/chriamue/bird-species-dataset) published on Hugging Face. The dataset contains 525 bird species and is the source of the images and label metadata used for training, evaluation, and local inference.

## Project Architecture and Methodology

The core of this project utilizes Transfer Learning via the ResNet18 convolutional neural network architecture.

During the initial experimental phases of this project, we attempted to train a custom image classification model entirely from scratch. However, the resulting accuracy was highly sub-optimal. The custom model struggled because it lacked a foundational understanding of basic visual features (such as edge detection, texture recognition, and shape processing), which require massive amounts of data and compute time to develop.

Instead of continuing with a model that lacked basic visual comprehension, we pivoted to leveraging existing, pre-trained models. By using ResNet18—which has already developed a deep, basic understanding of the visual world through extensive prior training—we were able to bypass the foundational feature-learning stage. We then fine-tuned this architecture and modified the final fully connected layers to specialize specifically in classifying our 525 distinct bird species. This approach dramatically improved both training efficiency and final inference accuracy.

## Repository Structure

- `bird_classifer.ipynb`: The primary Jupyter notebook containing the data pipeline, training loop, and evaluation metrics used to train the model.
- `app/`: The directory containing the local web application and its dependencies.
  - `main.py`: The entry point for the Streamlit web interface.
  - `generate_bird_labels.py`: A data processing script used to condense the large, raw dataset index into a lightweight, unique mapping file.
  - `bird_labels.csv`: The clean, 525-row mapping file generated for the application UI.
  - `pyproject.toml` & `uv.lock`: Dependency management files configured for the `uv` package manager.
  - `model/`: The target directory for the serialized PyTorch model weights (`.pth`).

## Local Setup and Installation

This project uses `uv` for fast, reproducible Python environment management.

1. **Navigate to the application directory:**

```bash
cd app

```

2. **Install dependencies:**
   Using `uv`, you can install all required packages (including PyTorch, Streamlit, and Pandas) directly from the configuration file:

```bash
uv sync

```

3. **Provide the Model Weights:**
   Ensure that your trained PyTorch weights file is placed in the model directory at the following path:
   `app/model/bird_resnet18_model.pth`

4. **Data Preparation (If necessary):**
   If you need to regenerate the label mappings from the raw dataset index (`birds.csv`), run the utility script:

```bash
uv run python generate_bird_labels.py

```

5. **Run the Application:**
   Launch the local Streamlit server to access the user interface:

```bash
uv run streamlit run main.py

```

## Usage

Once the Streamlit application is running, open the provided local URL in your web browser. Upload a `.jpg`, `.jpeg`, or `.png` image of a bird. The application will process the image through the standard ImageNet transforms, run inference via the local CPU, and output the predicted bird species along with a confidence percentage.
