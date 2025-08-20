# PySeas Data Guide 📊

**Understanding data structures and organization in PySeas**

## 🎯 Overview

PySeas manages several types of data:
- **Buoy Images**: Camera captures from NOAA buoys
- **Metadata**: Buoy information and image details
- **Processing Results**: ML classifications and quality scores
- **System Data**: Logs, configurations, and statistics

## 📁 Data Directory Structure

```
data/
├── collected_data/           # Raw buoy images
│   └── output/              # Processed data by buoy ID
│       ├── 41001/           # Buoy 41001 data
│       ├── 41002/           # Buoy 41002 data
│       └── ...              # Other buoys
├── balanced_training_data/   # ML training datasets
│   ├── clear/               # Clear weather images
│   ├── cloudy/              # Cloudy weather images
│   ├── stormy/              # Stormy weather images
│   ├── night/               # Night time images
│   └── ...                  # Other categories
└── supplemental_buoy_data/  # Additional buoy information
    ├── all_buoy_list_2023.csv
    ├── blacklisted_buoy_ids.csv
    └── locations_with_cameras.csv
```

## 🖼️ Image Data

### Image Naming Convention

```
{buoy_id}_{weather_condition}_{timestamp}.{extension}

Examples:
- 41001_clear_20230521_115244.png
- 41002_stormy_20230521_120947.png
- 41008_night_20230521_193106.png
```

### Image Categories

| Category | Description | Example Filenames |
|----------|-------------|-------------------|
| **clear** | Clear weather conditions | `41001_clear_*.png` |
| **cloudy** | Cloudy skies | `41002_cloudy_*.png` |
| **stormy** | Stormy weather | `41008_stormy_*.png` |
| **night** | Night time images | `41001_night_*.png` |
| **sunset** | Sunset/dusk images | `41002_sunset_*.png` |
| **beautiful** | Aesthetically pleasing | `41040_beautiful_*.png` |
| **object** | Objects detected | `41008_object_*.png` |

## 📊 Data Formats

### CSV Data Structure

```csv
buoy_id,timestamp,image_path,weather_condition,quality_score,processing_time
41001,2023-05-21 11:52:44,41001_clear_20230521_115244.png,clear,0.85,2.3
41002,2023-05-21 12:09:47,41002_stormy_20230521_120947.png,stormy,0.72,1.8
```

### JSON Metadata

```json
{
  "buoy_id": "41001",
  "location": {
    "latitude": 32.4,
    "longitude": -79.1
  },
  "image_info": {
    "filename": "41001_clear_20230521_115244.png",
    "size_bytes": 245760,
    "dimensions": [1920, 1080],
    "format": "PNG"
  },
  "weather_data": {
    "condition": "clear",
    "confidence": 0.85,
    "temperature": 22.5,
    "wind_speed": 8.2
  },
  "processing_info": {
    "processed_at": "2023-05-21T11:52:44Z",
    "processing_time": 2.3,
    "quality_score": 0.85
  }
}
```

## 🔄 Data Flow

### 1. Collection Phase
```
NOAA API → Image Download → Local Storage → Metadata Creation
```

### 2. Processing Phase
```
Raw Image → Quality Assessment → Weather Classification → Result Storage
```

### 3. Analysis Phase
```
Processed Data → Statistical Analysis → Report Generation → Export
```

## 📈 Data Statistics

### Typical Data Volumes

- **Daily Images**: 100-500 per buoy
- **Storage per Image**: 100KB - 2MB
- **Daily Storage**: 50MB - 1GB per buoy
- **Monthly Storage**: 1.5GB - 30GB per buoy

### Data Retention

- **Raw Images**: 30 days (configurable)
- **Processed Results**: 1 year (configurable)
- **Metadata**: Indefinite
- **Logs**: 90 days (configurable)

## 🗄️ Database Schema

### Core Tables

```sql
-- Buoy information
CREATE TABLE buoys (
    id VARCHAR(5) PRIMARY KEY,
    name VARCHAR(100),
    latitude DECIMAL(10,7),
    longitude DECIMAL(10,7),
    status VARCHAR(20),
    last_updated TIMESTAMP
);

-- Image records
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    buoy_id VARCHAR(5) REFERENCES buoys(id),
    filename VARCHAR(255),
    file_path TEXT,
    file_size BIGINT,
    dimensions VARCHAR(20),
    captured_at TIMESTAMP,
    processed_at TIMESTAMP,
    weather_condition VARCHAR(50),
    quality_score DECIMAL(3,2),
    metadata JSONB
);

-- Processing results
CREATE TABLE processing_results (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    model_name VARCHAR(100),
    predictions JSONB,
    confidence_scores JSONB,
    processing_time DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔍 Data Access Patterns

### Common Queries

```python
# Get recent images from a specific buoy
def get_recent_buoy_images(buoy_id: str, hours: int = 24):
    query = """
    SELECT * FROM images
    WHERE buoy_id = %s
    AND captured_at > NOW() - INTERVAL '%s hours'
    ORDER BY captured_at DESC
    """
    return execute_query(query, (buoy_id, hours))

# Get weather statistics for a time period
def get_weather_stats(start_date: str, end_date: str):
    query = """
    SELECT weather_condition, COUNT(*) as count
    FROM images
    WHERE captured_at BETWEEN %s AND %s
    GROUP BY weather_condition
    ORDER BY count DESC
    """
    return execute_query(query, (start_date, end_date))

# Find high-quality images
def get_high_quality_images(min_score: float = 0.8):
    query = """
    SELECT * FROM images
    WHERE quality_score >= %s
    ORDER BY quality_score DESC
    """
    return execute_query(query, (min_score,))
```

## 📤 Data Export

### Export Formats

```python
# CSV Export
def export_to_csv(filename: str, data: List[Dict]):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# JSON Export
def export_to_json(filename: str, data: List[Dict]):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# Excel Export
def export_to_excel(filename: str, data: List[Dict]):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
```

### Export Examples

```python
# Export buoy statistics
buoy_stats = get_buoy_statistics()
export_to_csv("buoy_statistics.csv", buoy_stats)

# Export image metadata
image_metadata = get_image_metadata()
export_to_json("image_metadata.json", image_metadata)

# Export processing results
results = get_processing_results()
export_to_excel("processing_results.xlsx", results)
```

## 🧹 Data Cleanup

### Automatic Cleanup

```python
class DataCleanupManager:
    def cleanup_old_images(self, days_old: int = 30):
        """Remove images older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Find old images
        old_images = self.find_old_images(cutoff_date)

        # Remove files and database records
        for image in old_images:
            self.remove_image_file(image.file_path)
            self.remove_database_record(image.id)

    def cleanup_low_quality_images(self, min_score: float = 0.5):
        """Remove images below quality threshold."""
        low_quality = self.find_low_quality_images(min_score)

        for image in low_quality:
            self.remove_image_file(image.file_path)
            self.remove_database_record(image.id)
```

## 📊 Data Monitoring

### Health Checks

```python
def check_data_health():
    """Monitor data system health."""
    checks = {
        "total_images": count_total_images(),
        "recent_collections": count_recent_collections(),
        "storage_usage": get_storage_usage(),
        "processing_backlog": get_processing_backlog(),
        "database_connections": check_database_connections()
    }

    return checks
```

### Performance Metrics

```python
def get_data_performance_metrics():
    """Get data processing performance metrics."""
    return {
        "images_per_hour": calculate_images_per_hour(),
        "average_processing_time": calculate_avg_processing_time(),
        "storage_growth_rate": calculate_storage_growth(),
        "error_rate": calculate_error_rate()
    }
```

---

## 📞 Data Support

For data-related questions:

- 📖 **Documentation**: Check other guides in the `docs/` folder
- 🗄️ **Schema**: Review database schema definitions
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 📧 **Contact**: Email the development team

---

*Complete data guide for PySeas* 📊
