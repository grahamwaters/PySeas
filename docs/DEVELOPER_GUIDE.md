# PySeas Developer Guide 🛠️

**Complete guide for developers contributing to and extending PySeas**

## 🎯 Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Code Standards](#code-standards)
4. [Adding New Features](#adding-new-features)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Deployment](#deployment)
8. [Contributing Guidelines](#contributing-guidelines)

## 🚀 Development Setup

### Prerequisites

- **Python 3.10+** with pip/conda
- **Git** for version control
- **Docker** (optional, for containerized development)
- **8GB+ RAM** for development and testing
- **IDE** (VS Code, PyCharm, or similar)

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/PySeas.git
cd PySeas

# Create development virtual environment
python -m venv pyseas_dev
source pyseas_dev/bin/activate  # On Windows: pyseas_dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
python -m pytest tests/ --version
```

### Development Dependencies

Create `requirements-dev.txt`:

```txt
# Core dependencies
-r requirements.txt

# Development tools
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0

# Testing utilities
factory-boy>=3.2.0
faker>=18.0.0
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./pyseas_dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm Configuration

1. Set project interpreter to `pyseas_dev/bin/python`
2. Enable pytest as test runner
3. Configure code style to use Black
4. Enable type checking with mypy

## 🏗️ Project Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   API Layer     │    │  Core Engine    │
│   (Optional)    │◄──►│   (REST/GraphQL)│◄──►│   (PySeas)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Data Storage   │    │  ML Models      │
                       │  (Images/CSV)   │    │  (TensorFlow)   │
                       └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Main Application (`src/pyseas.py`)

```python
class PySeas:
    """Main application orchestrator"""

    def __init__(self, config: Config):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.data_collector = DataCollector(config)
        self.model_manager = ModelManager(config)
        self.panorama_creator = PanoramaCreator(config)
```

#### 2. Image Processing Pipeline

```python
class ImageProcessingPipeline:
    """Handles image processing workflow"""

    def process_batch(self, images: List[str]) -> List[ProcessingResult]:
        results = []
        for image in images:
            # 1. Quality assessment
            quality = self.assess_quality(image)

            # 2. Classification
            if quality > self.threshold:
                classification = self.classify_image(image)

                # 3. Enhancement (optional)
                if self.enhance_enabled:
                    enhanced = self.enhance_image(image)

                results.append(ProcessingResult(image, quality, classification))

        return results
```

#### 3. Data Collection System

```python
class BuoyDataCollector:
    """Manages buoy data collection"""

    def collect_from_buoy(self, buoy_id: str) -> BuoyData:
        # 1. Check buoy availability
        if not self.is_buoy_available(buoy_id):
            raise BuoyUnavailableError(buoy_id)

        # 2. Download latest image
        image_path = self.download_latest_image(buoy_id)

        # 3. Extract metadata
        metadata = self.extract_metadata(buoy_id, image_path)

        # 4. Create buoy data object
        return BuoyData(buoy_id, image_path, metadata)
```

### Design Patterns

#### 1. Strategy Pattern (Image Processing)

```python
from abc import ABC, abstractmethod

class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

class ContrastEnhancer(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        # Implement contrast enhancement
        return enhanced_image

class NoiseReducer(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        # Implement noise reduction
        return denoised_image

class ImageProcessingContext:
    def __init__(self, processor: ImageProcessor):
        self.processor = processor

    def process_image(self, image: np.ndarray) -> np.ndarray:
        return self.processor.process(image)
```

#### 2. Observer Pattern (Event System)

```python
from abc import ABC, abstractmethod
from typing import List

class EventObserver(ABC):
    @abstractmethod
    def update(self, event: str, data: Any):
        pass

class ImageCollectionObserver(EventObserver):
    def update(self, event: str, data: Any):
        if event == "image_collected":
            self.log_collection(data)
        elif event == "collection_failed":
            self.handle_failure(data)

class EventManager:
    def __init__(self):
        self.observers: List[EventObserver] = []

    def add_observer(self, observer: EventObserver):
        self.observers.append(observer)

    def notify_observers(self, event: str, data: Any):
        for observer in self.observers:
            observer.update(event, data)
```

#### 3. Factory Pattern (Model Creation)

```python
class ModelFactory:
    """Creates ML models based on configuration"""

    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]):
        if model_type == "buoy_detection":
            return BuoyDetectionModel(config)
        elif model_type == "weather_classification":
            return WeatherClassificationModel(config)
        elif model_type == "quality_assessment":
            return QualityAssessmentModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

## 📝 Code Standards

### Python Style Guide

Follow **PEP 8** with these additional rules:

```python
# ✅ Good
def process_buoy_images(buoy_ids: List[str],
                       quality_threshold: float = 0.7,
                       enable_enhancement: bool = False) -> Dict[str, Any]:
    """
    Process images from specified buoys.

    Args:
        buoy_ids: List of buoy IDs to process
        quality_threshold: Minimum quality score (0.0-1.0)
        enable_enhancement: Whether to apply image enhancement

    Returns:
        Dictionary containing processing results

    Raises:
        ValueError: If buoy_ids is empty
        ProcessingError: If image processing fails
    """
    if not buoy_ids:
        raise ValueError("buoy_ids cannot be empty")

    results = {}
    for buoy_id in buoy_ids:
        try:
            result = _process_single_buoy(buoy_id, quality_threshold, enable_enhancement)
            results[buoy_id] = result
        except ProcessingError as e:
            logger.error(f"Failed to process buoy {buoy_id}: {e}")
            results[buoy_id] = {"error": str(e)}

    return results

# ❌ Bad
def process_buoy_images(buoy_ids,qt=0.7,enh=False):
    if not buoy_ids:raise ValueError("empty")
    r={}
    for bid in buoy_ids:
        try:r[bid]=_process_single_buoy(bid,qt,enh)
        except:r[bid]={"error":"failed"}
    return r
```

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from pathlib import Path

def analyze_image_quality(
    image_path: Union[str, Path],
    metrics: List[str],
    threshold: Optional[float] = None
) -> Dict[str, Union[float, bool]]:
    """
    Analyze image quality using specified metrics.

    Args:
        image_path: Path to the image file
        metrics: List of quality metrics to calculate
        threshold: Optional quality threshold

    Returns:
        Dictionary mapping metric names to values
    """
    pass
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def classify_weather_condition(image: np.ndarray) -> Dict[str, float]:
    """Classify weather conditions in an image.

    This function uses a pre-trained neural network to classify weather
    conditions in ocean buoy camera images.

    Args:
        image: Input image as numpy array (RGB format)

    Returns:
        Dictionary mapping weather conditions to confidence scores.
        Keys: 'clear', 'cloudy', 'stormy', 'foggy'
        Values: Confidence scores between 0.0 and 1.0

    Raises:
        ValueError: If image is not a valid numpy array
        ModelError: If the classification model fails to load

    Example:
        >>> image = cv2.imread('buoy_image.jpg')
        >>> result = classify_weather_condition(image)
        >>> print(f"Clear: {result['clear']:.2f}")
        Clear: 0.85
    """
    pass
```

#### Inline Comments

```python
# Calculate image sharpness using Laplacian variance
laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

# Normalize quality score to 0-1 range
quality_score = min(1.0, laplacian_var / MAX_SHARPNESS_THRESHOLD)

# Skip processing if quality is below threshold
if quality_score < self.quality_threshold:
    logger.debug(f"Image quality {quality_score:.3f} below threshold {self.quality_threshold}")
    return None
```

### Error Handling

#### Custom Exceptions

```python
class PySeasError(Exception):
    """Base exception for PySeas errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now()

class ImageProcessingError(PySeasError):
    """Raised when image processing fails."""
    pass

class NetworkError(PySeasError):
    """Raised when network operations fail."""
    pass

class ModelError(PySeasError):
    """Raised when ML model operations fail."""
    pass
```

#### Error Handling Patterns

```python
def process_image_safely(image_path: str) -> Optional[ProcessingResult]:
    """Process image with comprehensive error handling."""
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ImageProcessingError(f"Failed to load image: {image_path}")

        # Process image
        result = process_image(image)
        return result

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return None

    except ImageProcessingError as e:
        logger.error(f"Processing error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None
```

## 🚀 Adding New Features

### Feature Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-image-filter
   ```

2. **Implement Feature**
   ```python
   # Add new filter class
   class CustomImageFilter(ImageProcessor):
       def __init__(self, parameters: Dict[str, Any]):
           self.parameters = parameters

       def process(self, image: np.ndarray) -> np.ndarray:
           # Implement custom filtering logic
           return filtered_image
   ```

3. **Add Tests**
   ```python
   def test_custom_image_filter():
       filter = CustomImageFilter({"param1": "value1"})
       test_image = create_test_image((100, 100))
       result = filter.process(test_image)
       assert result.shape == test_image.shape
   ```

4. **Update Documentation**
   ```python
   def apply_custom_filter(self, filter_name: str, parameters: Dict[str, Any]):
       """
       Apply a custom image filter.

       Args:
           filter_name: Name of the filter to apply
           parameters: Filter-specific parameters

       Returns:
           Path to the filtered image
       """
       pass
   ```

5. **Submit Pull Request**

### Adding New ML Models

#### 1. Model Structure

```
models/
├── new_model/
│   ├── model.h5              # Trained model file
│   ├── labels.txt            # Class labels
│   ├── config.json           # Model configuration
│   ├── preprocessor.py       # Data preprocessing
│   └── README.md             # Model documentation
```

#### 2. Model Integration

```python
class NewModelWrapper(ModelWrapper):
    """Wrapper for the new ML model."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()
        self.labels = self.load_labels()

    def predict(self, input_data: np.ndarray) -> Dict[str, float]:
        """Make predictions using the model."""
        # Preprocess input
        processed_input = self.preprocessor.transform(input_data)

        # Make prediction
        prediction = self.model.predict(processed_input)

        # Format output
        return self.format_prediction(prediction)

    def load_model(self):
        """Load the trained model."""
        return tf.keras.models.load_model(self.model_path)
```

#### 3. Model Registration

```python
# In models/__init__.py
AVAILABLE_MODELS = {
    "buoy_detection": BuoyDetectionModel,
    "weather_classification": WeatherClassificationModel,
    "quality_assessment": QualityAssessmentModel,
    "new_model": NewModelWrapper,  # Add new model here
}

def get_model(model_name: str) -> ModelWrapper:
    """Get model instance by name."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = AVAILABLE_MODELS[model_name]
    return model_class()
```

### Adding New Data Sources

#### 1. Data Source Interface

```python
from abc import ABC, abstractmethod

class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available."""
        pass

    @abstractmethod
    def collect_data(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from the source."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        pass
```

#### 2. Implementation

```python
class NewBuoyDataSource(DataSource):
    """Data source for new buoy network."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def is_available(self) -> bool:
        """Check if the API is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def collect_data(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect buoy data from the API."""
        buoy_ids = parameters.get("buoy_ids", [])
        data = []

        for buoy_id in buoy_ids:
            try:
                response = self.session.get(f"{self.base_url}/buoy/{buoy_id}")
                if response.status_code == 200:
                    buoy_data = response.json()
                    data.append(buoy_data)
            except requests.RequestException as e:
                logger.warning(f"Failed to collect data for buoy {buoy_id}: {e}")

        return data

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        return {
            "name": "New Buoy Network",
            "type": "API",
            "base_url": self.base_url,
            "supported_formats": ["JSON"],
            "rate_limit": "100 requests/hour"
        }
```

#### 3. Integration

```python
# In data_collector.py
def add_data_source(self, source_name: str, source: DataSource):
    """Add a new data source."""
    self.data_sources[source_name] = source

def collect_from_source(self, source_name: str, parameters: Dict[str, Any]):
    """Collect data from a specific source."""
    if source_name not in self.data_sources:
        raise ValueError(f"Unknown data source: {source_name}")

    source = self.data_sources[source_name]
    if not source.is_available():
        raise DataSourceUnavailableError(source_name)

    return source.collect_data(parameters)
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_image_processing.py
│   ├── test_data_collection.py
│   └── test_models.py
├── integration/             # Integration tests
│   ├── test_full_pipeline.py
│   └── test_api_endpoints.py
├── fixtures/                # Test fixtures
│   ├── sample_images/
│   └── test_data.json
└── conftest.py             # Pytest configuration
```

### Unit Testing

#### Test Naming Convention

```python
# ✅ Good
def test_image_quality_assessment_with_valid_image():
    """Test image quality assessment with valid input."""
    pass

def test_image_quality_assessment_with_invalid_image():
    """Test image quality assessment with invalid input."""
    pass

def test_image_quality_assessment_edge_cases():
    """Test image quality assessment edge cases."""
    pass

# ❌ Bad
def test1():
    pass

def test_image_quality():
    pass
```

#### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from src.image_processor import ImageProcessor

class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance for testing."""
        return ImageProcessor(quality_threshold=0.7)

    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_quality_assessment_valid_image(self, processor, sample_image):
        """Test quality assessment with valid image."""
        # Arrange
        expected_quality = 0.8

        # Act
        actual_quality = processor.assess_quality(sample_image)

        # Assert
        assert actual_quality == expected_quality
        assert 0.0 <= actual_quality <= 1.0

    def test_quality_assessment_invalid_input(self, processor):
        """Test quality assessment with invalid input."""
        # Arrange
        invalid_input = None

        # Act & Assert
        with pytest.raises(ValueError, match="Image cannot be None"):
            processor.assess_quality(invalid_input)

    @patch('cv2.Laplacian')
    def test_quality_assessment_uses_laplacian(self, mock_laplacian, processor, sample_image):
        """Test that quality assessment uses Laplacian operator."""
        # Arrange
        mock_laplacian.return_value = np.ones((100, 100))

        # Act
        processor.assess_quality(sample_image)

        # Assert
        mock_laplacian.assert_called_once()
```

### Integration Testing

```python
import pytest
from src.pyseas import PySeas

class TestFullPipeline:
    """Test the complete PySeas pipeline."""

    @pytest.fixture
    def pyseas(self, tmp_path):
        """Create PySeas instance with temporary directory."""
        return PySeas(
            image_directory=str(tmp_path / "images"),
            buoy_ids=["test_buoy_1", "test_buoy_2"]
        )

    def test_full_image_processing_pipeline(self, pyseas, sample_images):
        """Test complete image processing pipeline."""
        # Arrange
        image_paths = [str(img) for img in sample_images]

        # Act
        results = pyseas.process_images(image_paths=image_paths)

        # Assert
        assert len(results) == len(image_paths)
        for result in results.values():
            assert result["processed"] == True
            assert "quality_score" in result
            assert "classification" in result

    def test_panorama_creation(self, pyseas, sample_images):
        """Test panorama creation from multiple images."""
        # Arrange
        image_paths = [str(img) for img in sample_images[:4]]

        # Act
        panorama_path = pyseas.create_panorama(
            buoy_ids=["test_buoy_1"],
            layout="grid",
            image_size=(800, 600)
        )

        # Assert
        assert os.path.exists(panorama_path)
        assert panorama_path.endswith(".jpg")
```

### Test Fixtures

```python
# conftest.py
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_images(tmp_path):
    """Create sample images for testing."""
    images = []
    for i in range(5):
        # Create test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"

        # Save image
        cv2.imwrite(str(img_path), img_array)
        images.append(img_path)

    return images

@pytest.fixture
def mock_buoy_data():
    """Create mock buoy data for testing."""
    return {
        "buoy_id": "test_buoy_1",
        "timestamp": "2024-01-01T12:00:00Z",
        "image_url": "https://example.com/buoy_image.jpg",
        "location": {"lat": 40.0, "lon": -70.0}
    }

@pytest.fixture
def mock_ml_model():
    """Create mock ML model for testing."""
    model = Mock()
    model.predict.return_value = np.array([[0.8, 0.1, 0.1]])  # Clear weather
    return model
```

### Performance Testing

```python
import time
import pytest
from src.image_processor import ImageProcessor

class TestPerformance:
    """Performance tests for critical operations."""

    def test_image_processing_performance(self, large_image):
        """Test image processing performance with large images."""
        processor = ImageProcessor()

        # Measure processing time
        start_time = time.time()
        result = processor.process_image(large_image)
        processing_time = time.time() - start_time

        # Assert performance requirements
        assert processing_time < 1.0  # Should process in under 1 second
        assert result is not None

    def test_batch_processing_performance(self, many_images):
        """Test batch processing performance."""
        processor = ImageProcessor()

        # Measure batch processing time
        start_time = time.time()
        results = processor.process_batch(many_images)
        batch_time = time.time() - start_time

        # Calculate throughput
        throughput = len(many_images) / batch_time

        # Assert performance requirements
        assert throughput > 10  # Should process at least 10 images per second
        assert len(results) == len(many_images)
```

## ⚡ Performance Optimization

### Profiling and Benchmarking

#### Using cProfile

```python
import cProfile
import pstats
from src.image_processor import ImageProcessor

def profile_image_processing():
    """Profile image processing performance."""
    processor = ImageProcessor()

    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Run operation
    for _ in range(100):
        processor.process_image("sample_image.jpg")

    profiler.disable()

    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

#### Using memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    """Monitor memory usage during operation."""
    large_array = np.zeros((1000, 1000, 3))
    processed = process_large_image(large_array)
    return processed
```

### Optimization Techniques

#### 1. Vectorization

```python
# ❌ Slow - loops
def calculate_pixel_differences(image1, image2):
    height, width = image1.shape[:2]
    differences = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            diff = np.abs(image1[y, x] - image2[y, x])
            differences[y, x] = np.mean(diff)

    return differences

# ✅ Fast - vectorized
def calculate_pixel_differences_vectorized(image1, image2):
    return np.mean(np.abs(image1 - image2), axis=2)
```

#### 2. Caching

```python
from functools import lru_cache
import hashlib

class ImageProcessor:
    def __init__(self):
        self._cache = {}

    @lru_cache(maxsize=1000)
    def process_image_cached(self, image_hash: str):
        """Process image with caching."""
        return self._process_image(image_hash)

    def get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def process_image(self, image_path: str):
        """Process image with automatic caching."""
        image_hash = self.get_image_hash(image_path)
        return self.process_image_cached(image_hash)
```

#### 3. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class ParallelImageProcessor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def process_images_parallel(self, image_paths: List[str]) -> List[ProcessingResult]:
        """Process multiple images in parallel."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image, path): path
                for path in image_paths
            }

            # Collect results
            results = []
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results.append(ProcessingResult(path, error=str(e)))

            return results
```

### Memory Management

#### 1. Generator Functions

```python
def process_images_generator(image_paths: List[str]):
    """Process images one at a time to save memory."""
    for image_path in image_paths:
        try:
            result = process_single_image(image_path)
            yield result
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            yield ProcessingResult(image_path, error=str(e))

# Usage
for result in process_images_generator(image_paths):
    save_result(result)
```

#### 2. Context Managers

```python
class ImageContext:
    """Context manager for image operations."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None

    def __enter__(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return self.image

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.image = None  # Release memory

# Usage
with ImageContext("large_image.jpg") as image:
    result = process_image(image)
    # Image is automatically released after this block
```

## 🚀 Deployment

### Production Configuration

#### Environment Variables

```bash
# Production settings
export PYSEAS_ENV=production
export IMAGES_DIR=/data/ocean_images
export MAX_WORKERS=8
export QUALITY_THRESHOLD=0.8
export LOG_LEVEL=WARNING
export ENABLE_MONITORING=true
export BACKUP_ENABLED=true
export ARCHIVE_ENABLED=true

# Database settings
export DATABASE_URL=postgresql://user:pass@localhost/pyseas
export REDIS_URL=redis://localhost:6379

# Monitoring
export PROMETHEUS_ENABLED=true
export GRAFANA_ENABLED=true
```

#### Configuration File

```python
# config/production.py
PRODUCTION_CONFIG = {
    "system": {
        "max_workers": 8,
        "memory_limit": "8GB",
        "enable_monitoring": True,
        "enable_backup": True
    },
    "processing": {
        "quality_threshold": 0.8,
        "batch_size": 100,
        "enable_enhancement": False,
        "parallel_processing": True
    },
    "storage": {
        "images_directory": "/data/ocean_images",
        "backup_directory": "/data/backups",
        "archive_directory": "/data/archives",
        "max_storage_size": "1TB"
    },
    "monitoring": {
        "metrics_enabled": True,
        "health_check_interval": 300,
        "alert_on_failure": True,
        "log_retention_days": 30
    }
}
```

### Docker Deployment

#### Production Dockerfile

```dockerfile
# Use multi-stage build for optimization
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash pyseas
USER pyseas

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=pyseas:pyseas src/ ./src/
COPY --chown=pyseas:pyseas main.py .

# Create data directories
RUN mkdir -p /data/images /data/logs /data/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "main.py"]
```

#### Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  pyseas:
    build:
      context: .
      dockerfile: Dockerfile
    image: pyseas:latest
    container_name: pyseas-app
    restart: unless-stopped
    environment:
      - PYSEAS_ENV=production
      - IMAGES_DIR=/data/images
      - MAX_WORKERS=8
      - LOG_LEVEL=WARNING
    volumes:
      - ocean_images:/data/images
      - ocean_logs:/data/logs
      - ocean_models:/data/models
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    networks:
      - pyseas-network

  postgres:
    image: postgres:13
    container_name: pyseas-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=pyseas
      - POSTGRES_USER=pyseas
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - pyseas-network

  redis:
    image: redis:6-alpine
    container_name: pyseas-cache
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - pyseas-network

  nginx:
    image: nginx:alpine
    container_name: pyseas-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - pyseas
    networks:
      - pyseas-network

volumes:
  ocean_images:
    driver: local
  ocean_logs:
    driver: local
  ocean_models:
    driver: local
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  pyseas-network:
    driver: bridge
```

### Monitoring and Logging

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
IMAGES_PROCESSED = Counter('pyseas_images_processed_total', 'Total images processed')
PROCESSING_TIME = Histogram('pyseas_processing_duration_seconds', 'Image processing time')
ACTIVE_WORKERS = Gauge('pyseas_active_workers', 'Number of active workers')
MEMORY_USAGE = Gauge('pyseas_memory_bytes', 'Memory usage in bytes')

class MonitoredImageProcessor:
    def process_image(self, image_path: str):
        start_time = time.time()

        try:
            result = self._process_image(image_path)
            IMAGES_PROCESSED.inc()
            return result
        finally:
            processing_time = time.time() - start_time
            PROCESSING_TIME.observe(processing_time)
```

#### Structured Logging

```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("Processing image",
           image_path="buoy_image.jpg",
           buoy_id="41001",
           quality_score=0.85)
```

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** following code standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run tests**: `python -m pytest tests/`
7. **Check code style**: `black src/ && flake8 src/`
8. **Commit changes**: `git commit -m "Add amazing feature"`
9. **Push to branch**: `git push origin feature/amazing-feature`
10. **Create Pull Request**

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and cover new functionality
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Error handling is appropriate
- [ ] Logging is informative
- [ ] Dependencies are minimal and justified

### Issue Reporting

When reporting issues, include:

- **Environment**: OS, Python version, dependencies
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback
- **Screenshots**: If applicable
- **Additional context**: Any relevant information

---

## 📞 Developer Support

For developer questions:

- 📖 **Documentation**: Check other guides in the `docs/` folder
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussions**: Ask technical questions in GitHub Discussions
- 📧 **Contact**: Email the development team

---

*Happy coding with PySeas! 🚀*
