# PySeas 🌊

**Ocean Buoy Camera Image Processing and Analysis System**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)

## 📖 Overview

PySeas is a comprehensive Python-based system for collecting, processing, and analyzing images from NOAA buoy cameras deployed across the world's oceans. The system automatically scrapes images from NDBC (National Data Buoy Center) BuoyCAM stations, processes them using machine learning models, and creates beautiful panoramic collages of ocean conditions.

### 🌟 Key Features

- **Automated Image Collection**: Scrapes images from multiple NOAA buoy cameras simultaneously
- **Machine Learning Processing**: Uses trained models to classify ocean conditions and detect objects
- **Image Quality Control**: Automatically removes white/blank images and duplicates
- **Panoramic Creation**: Generates stunning panoramic views from multiple buoy images
- **Real-time Monitoring**: Continuous monitoring of buoy camera availability and image quality
- **Data Export**: Exports processed data and statistics in various formats
- **Docker Support**: Easy deployment and containerization

### 🎯 Use Cases

- **Marine Research**: Monitor ocean conditions and weather patterns
- **Environmental Monitoring**: Track marine ecosystem changes over time
- **Weather Forecasting**: Analyze visual weather data from ocean buoys
- **Educational Purposes**: Study oceanography and marine environments
- **Art and Photography**: Create beautiful ocean panoramas

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended for image processing

### Installation

#### Option 1: Direct Python Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PySeas.git
cd PySeas

# Create virtual environment
python -m venv pyseas_env
source pyseas_env/bin/activate  # On Windows: pyseas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR use conda
conda env create -f environment.yml
conda activate pyseas_env
```

#### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t pyseas .
docker run -p 8000:8000 pyseas
```

### Basic Usage

```python
from src.pyseas import PySeas

# Initialize the system
pyseas = PySeas()

# Collect images from all available buoys
pyseas.collect_images()

# Process and classify images
pyseas.process_images()

# Generate panoramic views
pyseas.create_panoramas()

# Export results
pyseas.export_data()
```

## 📁 Project Structure

```
PySeas/
├── 📁 src/                    # Main source code
│   ├── 📁 legacy_src/        # Legacy implementations
│   ├── 📁 rough_drafts/      # Development drafts
│   ├── helper_functions.py   # Utility functions
│   ├── phase_one.py          # Core processing logic
│   └── pyseas.py            # Main application class
├── 📁 models/                # Trained ML models
│   ├── buoy_model/           # Buoy detection model
│   ├── blank_or_not_model/   # Image quality model
│   └── improved_model/       # Enhanced classification model
├── 📁 data/                  # Data storage
│   ├── 📁 collected_data/    # Scraped buoy images
│   ├── 📁 balanced_training_data/  # Training datasets
│   └── 📁 supplemental_buoy_data/  # Additional buoy information
├── 📁 notebooks/             # Jupyter notebooks for analysis
├── 📁 docs/                  # Documentation
├── 📁 images/                # Generated images and outputs
├── 📁 assets/                # Static assets (fonts, etc.)
├── main.py                   # Main execution script
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── Dockerfile                # Docker configuration
└── docker-compose.yml        # Docker Compose setup
```

## 🔧 Configuration

### Environment Variables

```bash
# Image storage directory
IMAGES_DIR=/path/to/images

# Logging level
LOG_LEVEL=INFO

# Buoy camera IDs to monitor
BUOY_IDS=41001,41002,41008,41010,41012,41013,41025,41040,41043,41044,41046,41047,41048,41049
```

### Configuration Files

- `config.py`: Main configuration settings
- `src/helper_functions.py`: Processing parameters
- `models/*/labels.txt`: Model classification labels

## 📊 Data Sources

### NOAA Buoy Cameras

PySeas integrates with the National Data Buoy Center (NDBC) network, accessing cameras from:

- **Atlantic Coast**: Stations 41001-41098
- **Gulf of Mexico**: Stations 42001-42099
- **Pacific Coast**: Stations 46001-46099
- **Great Lakes**: Stations 45001-45099
- **Alaska**: Stations 51001-51099

### Image Categories

The system automatically classifies images into:

- **Weather Conditions**: Clear, cloudy, stormy, beautiful
- **Time of Day**: Day, night, sunset, dusk
- **Special Events**: Objects detected, interesting phenomena
- **Quality Issues**: Blank, white, or corrupted images

## 🤖 Machine Learning Models

### Current Models

1. **Buoy Detection Model** (`models/buoy_model/`)
   - Identifies and locates buoys in images
   - Trained on thousands of buoy camera images

2. **Image Quality Model** (`models/blank_or_not_model/`)
   - Classifies images as usable or blank/white
   - Reduces storage of poor quality images

3. **Enhanced Classification Model** (`models/improved_model/`)
   - Multi-class classification of ocean conditions
   - Weather, time, and event detection

### Model Training

```python
# Train a new model
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_model(
    data_dir="data/balanced_training_data",
    model_output="models/new_model"
)
```

## 📈 Performance & Monitoring

### System Metrics

- **Image Processing Speed**: ~100-500 images/minute
- **Memory Usage**: 2-8GB depending on batch size
- **Storage Requirements**: 10-100GB for image collections
- **Network Usage**: Moderate during active collection

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pyseas.log'),
        logging.StreamHandler()
    ]
)
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_image_processing.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/PySeas.git
cd PySeas

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Make changes and test
python -m pytest

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)**: How to use PySeas
- **[API Reference](docs/API_REFERENCE.md)**: Code documentation
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Contributing and extending
- **[Architecture](docs/ARCHITECTURE.md)**: System design overview
- **[Data Guide](docs/DATA_GUIDE.md)**: Understanding data structures

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size in configuration
2. **Network Timeouts**: Check internet connection and NDBC availability
3. **Model Loading Issues**: Verify model files are present and compatible
4. **Permission Errors**: Ensure write access to output directories

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Search [existing issues](https://github.com/yourusername/PySeas/issues)
- 💬 Ask questions in [discussions](https://github.com/yourusername/PySeas/discussions)
- 📧 Contact: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NOAA NDBC**: For providing buoy camera access and data
- **OpenCV Community**: For computer vision capabilities
- **TensorFlow Team**: For machine learning framework
- **Contributors**: All who have helped improve PySeas

## 📊 Project Status

- **Version**: 1.0.0
- **Status**: Active Development
- **Last Updated**: December 2024
- **Python Support**: 3.8 - 3.11
- **License**: MIT

---

**Made with ❤️ for ocean exploration and research**

*If you find PySeas useful, please consider giving it a ⭐ on GitHub!*