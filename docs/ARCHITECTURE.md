# PySeas Architecture 🏗️

**System design and architecture overview for PySeas**

## 🎯 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [High-Level Architecture](#high-level-architecture)
4. [Component Architecture](#component-architecture)
5. [Data Flow](#data-flow)
6. [Technology Stack](#technology-stack)
7. [Scalability Considerations](#scalability-considerations)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)

## 🌟 System Overview

PySeas is a distributed, scalable system designed to collect, process, and analyze ocean buoy camera images from NOAA's National Data Buoy Center (NDBC) network. The system employs a microservices architecture with event-driven processing pipelines to handle high-throughput image processing requirements.

### Core Objectives

- **Scalability**: Handle thousands of buoy cameras simultaneously
- **Reliability**: Ensure robust operation in marine environments
- **Performance**: Process images in real-time with minimal latency
- **Extensibility**: Support new data sources and processing algorithms
- **Maintainability**: Clear separation of concerns and modular design

### System Characteristics

- **Type**: Distributed image processing system
- **Architecture**: Microservices with event-driven messaging
- **Processing Model**: Asynchronous, batch-oriented
- **Data Model**: Event-sourced with CQRS pattern
- **Deployment**: Containerized with Kubernetes support

## 🏛️ Architecture Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

```python
# ✅ Good - Single responsibility
class ImageQualityAssessor:
    """Assesses image quality using various metrics."""
    def assess_quality(self, image: np.ndarray) -> float:
        pass

class WeatherClassifier:
    """Classifies weather conditions in images."""
    def classify_weather(self, image: np.ndarray) -> Dict[str, float]:
        pass

# ❌ Bad - Multiple responsibilities
class ImageProcessor:
    """Processes images, classifies weather, and creates panoramas."""
    def process_image(self, image: np.ndarray):
        # Quality assessment
        # Weather classification
        # Panorama creation
        pass
```

### 2. Loose Coupling

Components communicate through well-defined interfaces:

```python
from abc import ABC, abstractmethod

class DataSource(ABC):
    """Abstract interface for data sources."""

    @abstractmethod
    def collect_data(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

class NOAADataSource(DataSource):
    """Concrete implementation for NOAA data."""
    def collect_data(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # NOAA-specific implementation
        pass

class CustomDataSource(DataSource):
    """Concrete implementation for custom data."""
    def collect_data(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Custom implementation
        pass
```

### 3. High Cohesion

Related functionality is grouped together:

```python
# ✅ Good - High cohesion
class BuoyDataManager:
    """Manages all buoy-related data operations."""

    def get_buoy_info(self, buoy_id: str) -> BuoyInfo:
        pass

    def update_buoy_status(self, buoy_id: str, status: str):
        pass

    def get_buoy_history(self, buoy_id: str) -> List[BuoyData]:
        pass

# ❌ Bad - Low cohesion
class UtilityFunctions:
    """Miscellaneous utility functions."""

    def get_buoy_info(self, buoy_id: str) -> BuoyInfo:
        pass

    def calculate_image_entropy(self, image: np.ndarray) -> float:
        pass

    def format_timestamp(self, timestamp: datetime) -> str:
        pass
```

### 4. Dependency Inversion

High-level modules don't depend on low-level modules:

```python
# ✅ Good - Dependency inversion
class ImageProcessingService:
    def __init__(self, processor: ImageProcessor, storage: StorageInterface):
        self.processor = processor
        self.storage = storage

    def process_and_store(self, image_path: str):
        result = self.processor.process(image_path)
        self.storage.save(result)

# ❌ Bad - Direct dependency
class ImageProcessingService:
    def __init__(self):
        self.processor = OpenCVProcessor()  # Direct instantiation
        self.storage = FileStorage()        # Direct instantiation
```

## 🏗️ High-Level Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Web UI      │  │ REST API    │  │ GraphQL API        │ │
│  │ (Optional)  │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Image       │  │ Data        │  │ Analysis            │ │
│  │ Processing  │  │ Collection  │  │ Services            │ │
│  │ Services    │  │ Services    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Image       │  │ Buoy        │  │ Weather             │ │
│  │ Processing  │  │ Management  │  │ Classification      │ │
│  │ Domain      │  │ Domain      │  │ Domain              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ File        │  │ Database    │  │ External            │ │
│  │ Storage     │  │ Storage     │  │ Services            │ │
│  │             │  │             │  │ (NOAA API)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web UI    │◄──►│  API Layer  │◄──►│ Core Engine│
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Event Bus   │    │ Processing  │
                   │             │    │ Pipeline    │
                   └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Data Store  │    │ ML Models   │
                   │             │    │             │
                   └─────────────┘    └─────────────┘
```

## 🔧 Component Architecture

### 1. Core Engine (`src/pyseas.py`)

The main orchestrator that coordinates all system operations.

```python
class PySeas:
    """Main application orchestrator."""

    def __init__(self, config: Config):
        # Initialize core components
        self.event_bus = EventBus()
        self.image_processor = ImageProcessor(config)
        self.data_collector = DataCollector(config)
        self.model_manager = ModelManager(config)
        self.panorama_creator = PanoramaCreator(config)

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers for system events."""
        self.event_bus.subscribe("image_collected", self._handle_image_collected)
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
        self.event_bus.subscribe("error_occurred", self._handle_error)

    def start(self):
        """Start the PySeas system."""
        self.event_bus.start()
        self.data_collector.start()
        self.image_processor.start()

    def stop(self):
        """Stop the PySeas system."""
        self.event_bus.stop()
        self.data_collector.stop()
        self.image_processor.stop()
```

### 2. Event Bus

Manages communication between components using an event-driven architecture.

```python
class EventBus:
    """Central event management system."""

    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = Queue()
        self.running = False

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        self.subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: Any):
        """Publish an event."""
        event = Event(event_type, data, datetime.now())
        self.event_queue.put(event)

    def start(self):
        """Start event processing."""
        self.running = True
        self._process_events()

    def _process_events(self):
        """Process events from the queue."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
            except Empty:
                continue

    def _dispatch_event(self, event: Event):
        """Dispatch event to all subscribers."""
        for handler in self.subscribers[event.type]:
            try:
                handler(event.data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
```

### 3. Image Processing Pipeline

Multi-stage pipeline for processing buoy images.

```python
class ImageProcessingPipeline:
    """Multi-stage image processing pipeline."""

    def __init__(self, config: Config):
        self.stages = [
            QualityAssessmentStage(config.quality_threshold),
            ClassificationStage(config.classification_enabled),
            EnhancementStage(config.enhancement_enabled),
            StorageStage(config.storage_config)
        ]
        self.event_bus = config.event_bus

    def process_image(self, image_path: str) -> ProcessingResult:
        """Process image through all stages."""
        context = ProcessingContext(image_path)

        try:
            for stage in self.stages:
                context = stage.process(context)
                if not context.continue_processing:
                    break

            # Publish completion event
            self.event_bus.publish("processing_completed", context.result)
            return context.result

        except Exception as e:
            error_result = ProcessingResult(
                image_path=image_path,
                success=False,
                error=str(e)
            )
            self.event_bus.publish("processing_error", error_result)
            return error_result

class ProcessingContext:
    """Context object passed between pipeline stages."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.metadata = {}
        self.continue_processing = True
        self.result = None
```

### 4. Data Collection System

Manages data collection from multiple buoy sources.

```python
class DataCollectionOrchestrator:
    """Orchestrates data collection from multiple sources."""

    def __init__(self, config: Config):
        self.sources = self._initialize_sources(config)
        self.scheduler = DataCollectionScheduler(config.schedule)
        self.event_bus = config.event_bus

    def _initialize_sources(self, config: Config) -> Dict[str, DataSource]:
        """Initialize data sources based on configuration."""
        sources = {}

        if config.noaa_enabled:
            sources["noaa"] = NOAADataSource(config.noaa_config)

        if config.custom_sources:
            for source_config in config.custom_sources:
                source = self._create_custom_source(source_config)
                sources[source_config.name] = source

        return sources

    def start_collection(self):
        """Start data collection from all sources."""
        for source_name, source in self.sources.items():
            if source.is_available():
                self.scheduler.schedule_collection(source_name, source)

    def collect_from_source(self, source_name: str) -> CollectionResult:
        """Collect data from a specific source."""
        source = self.sources[source_name]

        try:
            data = source.collect_data()
            result = CollectionResult(source_name, data, success=True)

            # Publish collection event
            self.event_bus.publish("data_collected", result)
            return result

        except Exception as e:
            result = CollectionResult(source_name, [], success=False, error=str(e))
            self.event_bus.publish("collection_failed", result)
            return result
```

### 5. Machine Learning Integration

Manages ML models and provides prediction services.

```python
class ModelManager:
    """Manages machine learning models."""

    def __init__(self, config: Config):
        self.models = {}
        self.model_configs = config.models
        self.load_models()

    def load_models(self):
        """Load all configured models."""
        for model_name, model_config in self.model_configs.items():
            try:
                model = self._load_single_model(model_config)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

    def _load_single_model(self, config: ModelConfig) -> MLModel:
        """Load a single ML model."""
        if config.type == "tensorflow":
            return TensorFlowModel(config)
        elif config.type == "pytorch":
            return PyTorchModel(config)
        elif config.type == "scikit":
            return ScikitModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.type}")

    def predict(self, model_name: str, input_data: Any) -> PredictionResult:
        """Make prediction using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        model = self.models[model_name]
        return model.predict(input_data)

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a model."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name].get_info()
```

## 🔄 Data Flow

### 1. Image Collection Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ NOAA API    │───►│ Data        │───►│ Event Bus   │───►│ Processing  │
│             │    │ Collector   │    │             │    │ Pipeline    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │ Local       │    │ Event       │    │ Quality     │
                   │ Storage     │    │ Queue       │    │ Assessment  │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Image Processing Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Image       │───►│ Quality     │───►│ Weather     │───►│ Storage     │
│ Input       │    │ Assessment  │    │ Classifier  │    │ & Metadata  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │ Quality     │    │ Confidence  │    │ Database    │
                   │ Score       │    │ Scores      │    │ Update      │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

### 3. Event Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Component   │───►│ Event Bus   │───►│ Event      │
│ Action      │    │             │    │ Handlers   │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Event       │    │ Side       │
                   │ Queue       │    │ Effects    │
                   └─────────────┘    └─────────────┘
```

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Language** | Python | 3.10+ | Core application logic |
| **Image Processing** | OpenCV | 4.8+ | Computer vision operations |
| **Machine Learning** | TensorFlow | 2.13+ | Neural network models |
| **Data Processing** | NumPy | 1.24+ | Numerical computations |
| **Web Framework** | FastAPI | 0.100+ | REST API endpoints |
| **Message Queue** | Redis | 6.2+ | Event queuing |
| **Database** | PostgreSQL | 13+ | Persistent storage |
| **Containerization** | Docker | 20.10+ | Application packaging |

### Supporting Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Logging** | structlog | Structured logging |
| **Configuration** | Pydantic | Configuration validation |
| **Testing** | pytest | Unit and integration testing |
| **Code Quality** | Black, flake8 | Code formatting and linting |
| **Type Checking** | mypy | Static type analysis |
| **Documentation** | Sphinx | API documentation |
| **Monitoring** | Prometheus | Metrics collection |
| **Visualization** | Grafana | Metrics visualization |

### External Dependencies

| Service | Purpose | Integration Method |
|---------|---------|-------------------|
| **NOAA NDBC API** | Buoy camera images | HTTP REST API |
| **Weather APIs** | Meteorological data | HTTP REST API |
| **Cloud Storage** | Image backup | S3-compatible API |
| **Email Service** | Notifications | SMTP API |

## 📈 Scalability Considerations

### 1. Horizontal Scaling

```python
class ScalableImageProcessor:
    """Horizontally scalable image processor."""

    def __init__(self, config: Config):
        self.worker_pool = WorkerPool(config.max_workers)
        self.task_queue = RedisQueue(config.redis_url)
        self.result_store = RedisStore(config.redis_url)

    def process_images(self, image_paths: List[str]) -> List[ProcessingResult]:
        """Process images using distributed workers."""
        # Submit tasks to queue
        task_ids = []
        for image_path in image_paths:
            task_id = self.task_queue.submit({
                "type": "process_image",
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            })
            task_ids.append(task_id)

        # Wait for completion
        results = []
        for task_id in task_ids:
            result = self.result_store.get(task_id)
            results.append(result)

        return results
```

### 2. Load Balancing

```python
class LoadBalancer:
    """Distributes load across multiple processing nodes."""

    def __init__(self, nodes: List[ProcessingNode]):
        self.nodes = nodes
        self.current_node = 0

    def get_next_node(self) -> ProcessingNode:
        """Get next node using round-robin strategy."""
        node = self.nodes[self.current_node]
        self.current_node = (self.current_node + 1) % len(self.nodes)
        return node

    def get_least_loaded_node(self) -> ProcessingNode:
        """Get node with lowest current load."""
        return min(self.nodes, key=lambda n: n.current_load)

    def get_fastest_node(self) -> ProcessingNode:
        """Get node with best performance metrics."""
        return max(self.nodes, key=lambda n: n.performance_score)
```

### 3. Caching Strategy

```python
class MultiLevelCache:
    """Multi-level caching system."""

    def __init__(self, config: Config):
        self.l1_cache = LRUCache(config.l1_size)      # In-memory
        self.l2_cache = RedisCache(config.redis_url)  # Redis
        self.l3_cache = DiskCache(config.disk_path)   # Disk

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try L1 cache
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            self.l1_cache.set(key, value)  # Promote to L1
            return value

        # Try L3 cache
        value = self.l3_cache.get(key)
        if value is not None:
            self.l2_cache.set(key, value)  # Promote to L2
            return value

        return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in all cache levels."""
        self.l1_cache.set(key, value, ttl)
        self.l2_cache.set(key, value, ttl)
        self.l3_cache.set(key, value, ttl)
```

### 4. Database Sharding

```python
class ShardedDatabase:
    """Sharded database for horizontal scaling."""

    def __init__(self, config: Config):
        self.shards = self._initialize_shards(config.shards)
        self.sharding_strategy = config.sharding_strategy

    def _get_shard(self, key: str) -> DatabaseShard:
        """Get appropriate shard for a key."""
        if self.sharding_strategy == "hash":
            shard_index = hash(key) % len(self.shards)
        elif self.sharding_strategy == "range":
            shard_index = self._get_range_shard(key)
        else:
            raise ValueError(f"Unknown sharding strategy: {self.sharding_strategy}")

        return self.shards[shard_index]

    def get(self, key: str) -> Optional[Any]:
        """Get value from appropriate shard."""
        shard = self._get_shard(key)
        return shard.get(key)

    def set(self, key: str, value: Any):
        """Set value in appropriate shard."""
        shard = self._get_shard(key)
        shard.set(key, value)
```

## 🔒 Security Architecture

### 1. Authentication & Authorization

```python
class SecurityManager:
    """Manages system security."""

    def __init__(self, config: Config):
        self.jwt_secret = config.jwt_secret
        self.api_keys = config.api_keys
        self.rate_limiters = self._initialize_rate_limiters()

    def authenticate_request(self, request: Request) -> Optional[User]:
        """Authenticate incoming request."""
        # Check API key
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in self.api_keys:
            return self.api_keys[api_key]

        # Check JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._verify_jwt_token(token)

        return None

    def authorize_action(self, user: User, action: str, resource: str) -> bool:
        """Check if user is authorized for action."""
        if not user:
            return False

        # Check user permissions
        if action in user.permissions.get(resource, []):
            return True

        # Check role-based permissions
        for role in user.roles:
            if action in role.permissions.get(resource, []):
                return True

        return False
```

### 2. Input Validation

```python
class InputValidator:
    """Validates and sanitizes input data."""

    def validate_image_path(self, path: str) -> bool:
        """Validate image file path."""
        # Check path traversal
        if ".." in path or path.startswith("/"):
            return False

        # Check file extension
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif"}
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return False

        # Check file size (max 50MB)
        try:
            if os.path.getsize(path) > 50 * 1024 * 1024:
                return False
        except OSError:
            return False

        return True

    def validate_buoy_id(self, buoy_id: str) -> bool:
        """Validate buoy ID format."""
        # Buoy IDs are typically 5-character alphanumeric
        import re
        pattern = r'^[A-Z0-9]{5}$'
        return bool(re.match(pattern, buoy_id))

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        return filename
```

### 3. Rate Limiting

```python
class RateLimiter:
    """Implements rate limiting for API endpoints."""

    def __init__(self, config: Config):
        self.redis_client = redis.Redis.from_url(config.redis_url)
        self.limits = config.rate_limits

    def is_allowed(self, client_id: str, endpoint: str) -> bool:
        """Check if request is allowed."""
        key = f"rate_limit:{client_id}:{endpoint}"
        current = self.redis_client.get(key)

        if current is None:
            # First request
            self.redis_client.setex(key, 3600, 1)  # 1 hour TTL
            return True

        current_count = int(current)
        limit = self.limits.get(endpoint, 100)  # Default 100 requests/hour

        if current_count < limit:
            self.redis_client.incr(key)
            return True

        return False

    def get_remaining_requests(self, client_id: str, endpoint: str) -> int:
        """Get remaining requests for client."""
        key = f"rate_limit:{client_id}:{endpoint}"
        current = self.redis_client.get(key)

        if current is None:
            return self.limits.get(endpoint, 100)

        remaining = self.limits.get(endpoint, 100) - int(current)
        return max(0, remaining)
```

## 🚀 Deployment Architecture

### 1. Container Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main application
  pyseas-app:
    build: .
    image: pyseas:latest
    container_name: pyseas-app
    restart: unless-stopped
    environment:
      - PYSEAS_ENV=production
      - DATABASE_URL=postgresql://pyseas:${DB_PASSWORD}@postgres:5432/pyseas
      - REDIS_URL=redis://redis:6379
    volumes:
      - ocean_images:/data/images
      - ocean_logs:/data/logs
    depends_on:
      - postgres
      - redis
    networks:
      - pyseas-network

  # Database
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

  # Cache and message queue
  redis:
    image: redis:6-alpine
    container_name: pyseas-cache
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - pyseas-network

  # Reverse proxy
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
      - pyseas-app
    networks:
      - pyseas-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: pyseas-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - pyseas-network

  grafana:
    image: grafana/grafana:latest
    container_name: pyseas-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - pyseas-network

volumes:
  ocean_images:
    driver: local
  ocean_logs:
    driver: local
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  pyseas-network:
    driver: bridge
```

### 2. Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pyseas-app
  namespace: pyseas
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pyseas-app
  template:
    metadata:
      labels:
        app: pyseas-app
    spec:
      containers:
      - name: pyseas-app
        image: pyseas:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYSEAS_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pyseas-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://pyseas-redis:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: ocean-images
          mountPath: /data/images
        - name: ocean-logs
          mountPath: /data/logs
      volumes:
      - name: ocean-images
        persistentVolumeClaim:
          claimName: ocean-images-pvc
      - name: ocean-logs
        persistentVolumeClaim:
          claimName: ocean-logs-pvc
```

### 3. Service Mesh

```yaml
# istio/virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: pyseas-vs
  namespace: pyseas
spec:
  hosts:
  - pyseas.example.com
  gateways:
  - pyseas-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1
    route:
    - destination:
        host: pyseas-app
        port:
          number: 8000
      weight: 100
  - match:
    - uri:
        prefix: /health
    route:
    - destination:
        host: pyseas-app
        port:
          number: 8000
      weight: 100
  - match:
    - uri:
        prefix: /metrics
    route:
    - destination:
        host: pyseas-app
        port:
          number: 8000
      weight: 100
```

---

## 📞 Architecture Support

For architecture questions:

- 📖 **Documentation**: Check other guides in the `docs/` folder
- 🏗️ **Design Decisions**: Review architecture discussions
- 💬 **Discussions**: Ask technical questions in GitHub Discussions
- 📧 **Contact**: Email the development team

---

*Comprehensive architecture overview for PySeas* 🚀
