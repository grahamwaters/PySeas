# PySeas API Reference 📚

**Complete API documentation for the PySeas system**

## 🎯 Table of Contents

1. [Core Classes](#core-classes)
2. [Image Processing](#image-processing)
3. [Data Collection](#data-collection)
4. [Machine Learning](#machine-learning)
5. [Utility Functions](#utility-functions)
6. [Configuration](#configuration)
7. [Data Structures](#data-structures)

## 🏗️ Core Classes

### PySeas

Main application class that orchestrates all PySeas operations.

```python
class PySeas:
    def __init__(self,
                 image_directory: str = "./images",
                 buoy_ids: List[str] = None,
                 max_workers: int = 4,
                 verbose: bool = False,
                 cleanup_enabled: bool = True)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_directory` | str | `"./images"` | Directory to store images |
| `buoy_ids` | List[str] | None | List of buoy IDs to monitor |
| `max_workers` | int | 4 | Maximum parallel workers |
| `verbose` | bool | False | Enable verbose logging |
| `cleanup_enabled` | bool | True | Enable automatic cleanup |

#### Methods

##### Image Collection

```python
def collect_images(self,
                  buoy_ids: List[str] = None,
                  max_retries: int = 3,
                  timeout: int = 30,
                  save_metadata: bool = True,
                  force_refresh: bool = False,
                  show_progress: bool = False) -> Dict[str, Any]
```

**Parameters:**
- `buoy_ids`: Specific buoy IDs to collect from
- `max_retries`: Number of retry attempts
- `timeout`: Network timeout in seconds
- `save_metadata`: Save buoy information
- `force_refresh`: Download even if exists
- `show_progress`: Show progress bars

**Returns:** Collection statistics dictionary

**Example:**
```python
pyseas = PySeas()
stats = pyseas.collect_images(
    buoy_ids=["41001", "41002"],
    max_retries=5,
    timeout=60
)
print(f"Collected {stats['total_images']} images")
```

##### Image Processing

```python
def process_images(self,
                  image_paths: List[str] = None,
                  quality_threshold: float = 0.7,
                  remove_duplicates: bool = True,
                  classify_weather: bool = True,
                  detect_objects: bool = False,
                  enhance_images: bool = False,
                  batch_size: int = 50) -> Dict[str, Any]
```

**Parameters:**
- `image_paths`: Specific images to process
- `quality_threshold`: Minimum quality score
- `remove_duplicates`: Remove duplicate images
- `classify_weather`: Classify weather conditions
- `detect_objects`: Detect objects in images
- `enhance_images`: Apply image enhancement
- `batch_size`: Processing batch size

**Returns:** Processing results dictionary

##### Data Analysis

```python
def generate_report(self,
                   report_type: str = "comprehensive",
                   start_date: str = None,
                   end_date: str = None,
                   buoy_ids: List[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `report_type`: Type of report to generate
- `start_date`: Start date for analysis
- `end_date`: End date for analysis
- `buoy_ids`: Buoys to include in report

**Returns:** Report data dictionary

##### Panoramic Creation

```python
def create_panorama(self,
                   buoy_ids: List[str] = None,
                   layout: str = "vertical",
                   image_size: Tuple[int, int] = (1920, 1080),
                   quality: str = "medium") -> str
```

**Parameters:**
- `buoy_ids`: Buoys to include in panorama
- `layout`: Panorama layout ("vertical", "horizontal", "grid")
- `image_size`: Output image dimensions
- `quality`: Output quality ("low", "medium", "high")

**Returns:** Path to created panorama image

### BuoyCamScraper

Handles individual buoy camera image scraping.

```python
class BuoyCamScraper:
    def __init__(self, image_directory: str, buoycam_ids: List[str])
```

#### Methods

```python
def scrape_buoycam_images(self, buoycam_id: str) -> bool
def get_latest_image(self, buoycam_id: str) -> Optional[str]
def check_buoycam_status(self, buoycam_id: str) -> Dict[str, Any]
```

## 🖼️ Image Processing

### ImageProcessor

Handles image processing operations.

```python
class ImageProcessor:
    def __init__(self, quality_threshold: float = 0.7)
```

#### Methods

##### Quality Assessment

```python
def assess_image_quality(self, image_path: str) -> float
def is_blank_image(self, image_path: str) -> bool
def is_white_image(self, image_path: str) -> bool
def calculate_sharpness(self, image_path: str) -> float
```

**Returns:** Quality metrics (0.0 to 1.0)

##### Image Enhancement

```python
def enhance_contrast(self, image_path: str) -> str
def remove_noise(self, image_path: str) -> str
def adjust_brightness(self, image_path: str, factor: float) -> str
def resize_image(self, image_path: str, dimensions: Tuple[int, int]) -> str
```

##### Duplicate Detection

```python
def find_duplicates(self, image_directory: str, similarity_threshold: float = 0.95) -> List[List[str]]
def remove_duplicates(self, image_directory: str, keep_best: bool = True) -> int
```

### PanoramaCreator

Creates panoramic images from multiple buoy images.

```python
class PanoramaCreator:
    def __init__(self, output_directory: str = "./panoramas")
```

#### Methods

```python
def create_vertical_panorama(self, image_paths: List[str], output_path: str) -> bool
def create_horizontal_panorama(self, image_paths: List[str], output_path: str) -> bool
def create_grid_panorama(self, image_paths: List[str], grid_size: Tuple[int, int], output_path: str) -> bool
def stitch_images(self, image_paths: List[str], method: str = "opencv") -> np.ndarray
```

## 📡 Data Collection

### DataCollector

Manages data collection and storage.

```python
class DataCollector:
    def __init__(self, base_directory: str = "./data")
```

#### Methods

```python
def collect_buoy_data(self, buoy_id: str) -> Dict[str, Any]
def save_metadata(self, buoy_id: str, metadata: Dict[str, Any]) -> bool
def load_metadata(self, buoy_id: str) -> Optional[Dict[str, Any]]
def export_to_csv(self, filename: str, data: List[Dict[str, Any]]) -> bool
def export_to_json(self, filename: str, data: List[Dict[str, Any]]) -> bool
```

### NetworkManager

Handles network operations and connectivity.

```python
class NetworkManager:
    def __init__(self, timeout: int = 30, max_retries: int = 3)
```

#### Methods

```python
def check_connectivity(self, url: str) -> bool
def download_file(self, url: str, local_path: str) -> bool
def get_response_time(self, url: str) -> float
def is_accessible(self, url: str) -> bool
```

## 🤖 Machine Learning

### ModelManager

Manages machine learning models.

```python
class ModelManager:
    def __init__(self, models_directory: str = "./models")
```

#### Methods

```python
def load_model(self, model_name: str) -> Any
def predict(self, model_name: str, input_data: Any) -> Any
def get_model_info(self, model_name: str) -> Dict[str, Any]
def list_available_models(self) -> List[str]
def validate_model(self, model_name: str) -> bool
```

### ImageClassifier

Classifies images using trained models.

```python
class ImageClassifier:
    def __init__(self, model_path: str)
```

#### Methods

```python
def classify_weather(self, image_path: str) -> Dict[str, Any]
def classify_time_of_day(self, image_path: str) -> str
def detect_objects(self, image_path: str) -> List[Dict[str, Any]]
def get_confidence_scores(self, image_path: str) -> Dict[str, float]
```

## 🛠️ Utility Functions

### File Operations

```python
def ensure_directory(directory_path: str) -> bool
def get_file_size(file_path: str) -> int
def get_file_extension(file_path: str) -> str
def is_image_file(file_path: str) -> bool
def list_image_files(directory_path: str, extensions: List[str] = None) -> List[str]
```

### Date and Time

```python
def parse_timestamp(timestamp: str) -> datetime
def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d_%H-%M-%S") -> str
def get_time_difference(timestamp1: str, timestamp2: str) -> timedelta
def is_recent(timestamp: str, hours: int = 24) -> bool
```

### Image Utilities

```python
def get_image_dimensions(image_path: str) -> Tuple[int, int]
def get_image_format(image_path: str) -> str
def convert_image_format(image_path: str, target_format: str) -> str
def validate_image_file(image_path: str) -> bool
```

## ⚙️ Configuration

### ConfigManager

Manages system configuration.

```python
class ConfigManager:
    def __init__(self, config_file: str = "config.py")
```

#### Methods

```python
def load_config(self) -> Dict[str, Any]
def save_config(self, config: Dict[str, Any]) -> bool
def get_setting(self, key: str, default: Any = None) -> Any
def set_setting(self, key: str, value: Any) -> bool
def validate_config(self) -> List[str]
```

### Environment Variables

```python
# Core settings
IMAGES_DIR = os.getenv("IMAGES_DIR", "./images")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Buoy settings
BUOY_IDS = os.getenv("BUOY_IDS", "").split(",")
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))

# Network settings
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Processing settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
ENABLE_ENHANCEMENT = os.getenv("ENABLE_ENHANCEMENT", "false").lower() == "true"
```

## 📊 Data Structures

### BuoyData

```python
@dataclass
class BuoyData:
    buoy_id: str
    timestamp: datetime
    image_path: str
    weather_condition: str
    time_of_day: str
    quality_score: float
    metadata: Dict[str, Any]
```

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    image_path: str
    processed: bool
    quality_score: float
    classification: Dict[str, Any]
    errors: List[str]
    processing_time: float
```

### CollectionStats

```python
@dataclass
class CollectionStats:
    total_images: int
    successful: int
    failed: int
    skipped: int
    total_size: int
    collection_time: float
    errors: List[str]
```

### ReportData

```python
@dataclass
class ReportData:
    report_type: str
    generated_at: datetime
    data_summary: Dict[str, Any]
    charts: List[str]
    recommendations: List[str]
    export_paths: List[str]
```

## 🔧 Error Handling

### Custom Exceptions

```python
class PySeasError(Exception):
    """Base exception for PySeas errors"""
    pass

class ImageProcessingError(PySeasError):
    """Raised when image processing fails"""
    pass

class NetworkError(PySeasError):
    """Raised when network operations fail"""
    pass

class ModelError(PySeasError):
    """Raised when ML model operations fail"""
    pass

class ConfigurationError(PySeasError):
    """Raised when configuration is invalid"""
    pass
```

### Error Handling Examples

```python
try:
    result = pyseas.process_images()
except ImageProcessingError as e:
    logger.error(f"Image processing failed: {e}")
    # Handle gracefully
except NetworkError as e:
    logger.error(f"Network error: {e}")
    # Retry or use cached data
except PySeasError as e:
    logger.error(f"PySeas error: {e}")
    # General error handling
```

## 📝 Logging

### Logger Configuration

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

# Create logger
logger = logging.getLogger(__name__)
```

### Logging Examples

```python
# Info logging
logger.info(f"Processing {len(image_paths)} images")

# Warning logging
logger.warning(f"Low quality image detected: {image_path}")

# Error logging
logger.error(f"Failed to process image: {image_path}", exc_info=True)

# Debug logging
logger.debug(f"Image metadata: {metadata}")
```

## 🧪 Testing

### Test Utilities

```python
def create_test_image(dimensions: Tuple[int, int], color: Tuple[int, int, int]) -> str
def create_test_buoy_data(buoy_id: str) -> BuoyData
def mock_network_response(status_code: int, content: bytes) -> MockResponse
def cleanup_test_files(directory: str) -> None
```

### Testing Examples

```python
import pytest
from unittest.mock import Mock, patch

def test_image_processing():
    with patch('cv2.imread') as mock_imread:
        mock_imread.return_value = np.zeros((100, 100, 3))
        result = process_image("test.jpg")
        assert result.success == True

def test_buoy_collection():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"fake_image_data"
        result = collect_buoy_image("41001")
        assert result.downloaded == True
```

---

## 📞 API Support

For questions about the API:

- 📖 **Documentation**: Check other guides in the `docs/` folder
- 🐛 **Issues**: Report API problems on GitHub
- 💬 **Discussions**: Ask API questions in GitHub Discussions
- 📧 **Contact**: Email the development team

---

*Complete API reference for PySeas development* 🚀
