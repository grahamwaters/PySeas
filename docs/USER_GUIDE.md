# PySeas User Guide 📚

**Complete guide to using PySeas for ocean buoy camera image processing**

## 🎯 Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Image Collection](#image-collection)
4. [Image Processing](#image-processing)
5. [Data Analysis](#data-analysis)
6. [Output Management](#output-management)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

## 🚀 Getting Started

### Prerequisites

Before using PySeas, ensure you have:

- **Python 3.10+** installed
- **8GB+ RAM** available for image processing
- **Stable internet connection** for buoy camera access
- **50GB+ free disk space** for image storage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PySeas.git
cd PySeas

# Create virtual environment
python -m venv pyseas_env
source pyseas_env/bin/activate  # On Windows: pyseas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

```python
from src.pyseas import PySeas

# Initialize the system
pyseas = PySeas()

# Check system status
status = pyseas.check_system_status()
print(f"System Status: {status}")
```

## 🔧 Basic Operations

### System Initialization

```python
import logging
from src.pyseas import PySeas

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize with custom settings
pyseas = PySeas(
    image_directory="/path/to/images",
    buoy_ids=["41001", "41002", "41008"],
    max_workers=4,
    verbose=True
)
```

### Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `image_directory` | Where to save images | `./images` | `/Volumes/SSD/images` |
| `buoy_ids` | List of buoy IDs to monitor | All available | `["41001", "41002"]` |
| `max_workers` | Number of parallel workers | 4 | 8 |
| `verbose` | Enable detailed logging | False | True |
| `cleanup_enabled` | Auto-remove poor images | True | False |

## 📸 Image Collection

### Collecting Images from Buoys

```python
# Collect from all available buoys
pyseas.collect_images()

# Collect from specific buoys
pyseas.collect_images(buoy_ids=["41001", "41002"])

# Collect with custom parameters
pyseas.collect_images(
    buoy_ids=["41001"],
    max_retries=3,
    timeout=30,
    save_metadata=True
)
```

### Collection Parameters

- **`max_retries`**: Number of retry attempts for failed downloads
- **`timeout`**: Network timeout in seconds
- **`save_metadata`**: Save buoy information and timestamps
- **`force_refresh`**: Download even if image exists

### Monitoring Collection Progress

```python
# Enable progress bars
pyseas.collect_images(show_progress=True)

# Get collection statistics
stats = pyseas.get_collection_stats()
print(f"Images collected: {stats['total_images']}")
print(f"Successful: {stats['successful']}")
print(f"Failed: {stats['failed']}")
```

## 🖼️ Image Processing

### Basic Image Processing

```python
# Process all collected images
pyseas.process_images()

# Process specific images
pyseas.process_images(image_paths=["path/to/image1.jpg", "path/to/image2.jpg"])

# Process with custom parameters
pyseas.process_images(
    quality_threshold=0.8,
    remove_duplicates=True,
    classify_weather=True
)
```

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `quality_threshold` | Minimum image quality score | 0.7 |
| `remove_duplicates` | Remove duplicate images | True |
| `classify_weather` | Classify weather conditions | True |
| `detect_objects` | Detect objects in images | False |
| `enhance_images` | Apply image enhancement | False |

### Image Classification

```python
# Classify images by weather conditions
classifications = pyseas.classify_images()

# View classification results
for image_path, classification in classifications.items():
    print(f"{image_path}: {classification['weather']} - {classification['confidence']}")

# Get classification statistics
stats = pyseas.get_classification_stats()
print(f"Clear: {stats['clear']}")
print(f"Cloudy: {stats['cloudy']}")
print(f"Stormy: {stats['stormy']}")
```

## 📊 Data Analysis

### Generating Reports

```python
# Generate comprehensive report
report = pyseas.generate_report()

# Save report to file
pyseas.save_report(report, "ocean_conditions_report.html")

# Generate specific report types
weather_report = pyseas.generate_weather_report()
quality_report = pyseas.generate_quality_report()
timeline_report = pyseas.generate_timeline_report()
```

### Data Export

```python
# Export to CSV
pyseas.export_to_csv("buoy_data.csv")

# Export to JSON
pyseas.export_to_json("buoy_data.json")

# Export to Excel
pyseas.export_to_excel("buoy_data.xlsx")

# Custom export format
pyseas.export_data(
    format="csv",
    filename="custom_export.csv",
    include_metadata=True,
    filter_by_quality=0.8
)
```

### Statistical Analysis

```python
# Get basic statistics
stats = pyseas.get_statistics()

# Analyze weather patterns
weather_patterns = pyseas.analyze_weather_patterns()

# Analyze image quality trends
quality_trends = pyseas.analyze_quality_trends()

# Generate time series data
time_series = pyseas.generate_time_series(
    buoy_id="41001",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

## 🎨 Output Management

### Panoramic Creation

```python
# Create panoramas from multiple buoys
panoramas = pyseas.create_panoramas()

# Create panorama from specific buoys
panorama = pyseas.create_panorama(buoy_ids=["41001", "41002", "41008"])

# Custom panorama settings
panorama = pyseas.create_panorama(
    buoy_ids=["41001", "41002"],
    layout="horizontal",
    image_size=(1920, 1080),
    quality="high"
)
```

### Image Organization

```python
# Organize images by date
pyseas.organize_by_date()

# Organize images by buoy
pyseas.organize_by_buoy()

# Organize images by weather
pyseas.organize_by_weather()

# Custom organization
pyseas.organize_images(
    primary_sort="date",
    secondary_sort="buoy_id",
    create_symlinks=True
)
```

### Backup and Archiving

```python
# Create backup of processed images
pyseas.create_backup("/path/to/backup")

# Archive old images
pyseas.archive_images(
    older_than_days=30,
    archive_path="/path/to/archive"
)

# Compress archived images
pyseas.compress_archive("/path/to/archive")
```

## 🔍 Advanced Features

### Custom Image Processing

```python
# Apply custom filters
pyseas.apply_custom_filter(
    filter_function=my_custom_filter,
    filter_name="custom_filter"
)

# Batch processing with custom pipeline
pipeline = [
    "resize",
    "enhance_contrast",
    "remove_noise",
    "custom_filter"
]

pyseas.process_with_pipeline(pipeline)
```

### Real-time Monitoring

```python
# Start real-time monitoring
pyseas.start_monitoring(
    check_interval=300,  # 5 minutes
    alert_on_failure=True
)

# Stop monitoring
pyseas.stop_monitoring()

# Get monitoring status
status = pyseas.get_monitoring_status()
```

### API Integration

```python
# Start API server
pyseas.start_api_server(port=8000)

# Make API requests
import requests

response = requests.get("http://localhost:8000/api/status")
data = response.json()

# Stop API server
pyseas.stop_api_server()
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Errors

```python
# Reduce batch size
pyseas = PySeas(max_workers=2)

# Process images in smaller batches
pyseas.process_images(batch_size=10)
```

#### 2. Network Timeouts

```python
# Increase timeout values
pyseas.collect_images(timeout=60, max_retries=5)

# Check network connectivity
if pyseas.check_network_connectivity():
    print("Network is accessible")
else:
    print("Network issues detected")
```

#### 3. Image Quality Issues

```python
# Adjust quality threshold
pyseas.process_images(quality_threshold=0.6)

# Enable detailed quality logging
pyseas.enable_quality_logging()

# Check image quality manually
quality_score = pyseas.assess_image_quality("path/to/image.jpg")
```

### Debug Mode

```python
# Enable debug mode
pyseas.enable_debug_mode()

# Get detailed error information
errors = pyseas.get_error_log()

# Test individual components
pyseas.test_image_processing()
pyseas.test_network_connectivity()
pyseas.test_model_loading()
```

### Performance Optimization

```python
# Profile performance
pyseas.profile_performance()

# Optimize settings
optimal_settings = pyseas.optimize_settings()

# Monitor resource usage
usage = pyseas.monitor_resources()
print(f"CPU: {usage['cpu']}%")
print(f"Memory: {usage['memory']}%")
print(f"Disk: {usage['disk']}%")
```

## 📚 Additional Resources

### Helpful Commands

```python
# Get system information
pyseas.get_system_info()

# Check available buoys
available_buoys = pyseas.get_available_buoys()

# Get buoy details
buoy_info = pyseas.get_buoy_info("41001")

# Check model status
model_status = pyseas.check_models()
```

### Configuration Examples

```python
# Production configuration
production_config = {
    "image_directory": "/data/ocean_images",
    "max_workers": 8,
    "quality_threshold": 0.8,
    "backup_enabled": True,
    "monitoring_enabled": True
}

# Development configuration
dev_config = {
    "image_directory": "./test_images",
    "max_workers": 2,
    "quality_threshold": 0.5,
    "verbose": True
}
```

---

## 📞 Need Help?

- 📖 **Documentation**: Check other guides in the `docs/` folder
- 🐛 **Issues**: Report problems on GitHub
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 📧 **Contact**: Email the development team

---

*Happy ocean exploring with PySeas! 🌊*
