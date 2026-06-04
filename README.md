# END-TO-END CHEST CANCER CLASSIFICATION

An end-to-end deep learning project for classifying chest CT scan images to detect adenocarcinoma cancer using VGG16 transfer learning, MLflow experiment tracking, DVC pipelines, and Flask deployment.

---

## Project Overview

This project builds a complete MLOps pipeline that:
- Ingests chest CT scan image data
- Trains a VGG16-based CNN classifier
- Tracks experiments with MLflow
- Manages pipelines with DVC
- Serves predictions via a Flask REST API

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras, VGG16 |
| Experiment Tracking | MLflow |
| Pipeline Management | DVC |
| Web Framework | Flask |
| Language | Python 3.11 |
| Deployment | Docker, AWS EC2/ECR |

---

## Project Structure

```
├── .github/workflows/       # CI/CD pipeline
├── config/
│   └── config.yaml          # Project configuration
├── research/
│   └── trials.ipynb         # Experimentation notebook
├── src/
│   └── cnnClassifier/
│       ├── components/      # Pipeline components
│       ├── config/          # Configuration manager
│       ├── constants/       # Project constants
│       ├── entity/          # Data classes
│       ├── pipeline/        # Training & prediction pipelines
│       └── utils/           # Helper utilities
├── templates/
│   └── index.html           # Frontend UI
├── main.py                  # Pipeline runner
├── app.py                   # Flask application
├── params.yaml              # Model hyperparameters
├── dvc.yaml                 # DVC pipeline stages
├── requirements.txt
└── setup.py
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/gben007/END-TO-END-CHEST-CANCER-CLASSIFICATION.git
cd END-TO-END-CHEST-CANCER-CLASSIFICATION

# Create and activate virtual environment
python3.11 -m venv chest
source chest/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
python main.py
```

---

## MLflow Tracking

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to view experiments.

---

## DVC Pipeline

```bash
dvc init
dvc repro
dvc dag
```

---

## Author

**Gaurav Beniwal**
- GitHub: [@gben007](https://github.com/gben007)
