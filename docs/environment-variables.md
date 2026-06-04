# Environment Variables

## Backend Configuration (`.env`):

```bash
# Environment (development, production)
ENV=development
DEBUG=true

# Server Configuration
HOST=localhost
PORT=8001

# CORS - Comma-separated list of allowed origins
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:4173

# File Upload Limits
MAX_FILE_SIZE_MB=50
MAX_PDF_PAGES=50
PDF_DPI=300
ALLOWED_IMAGE_EXTENSIONS=.jpg,.jpeg,.png,.webp,.bmp,.tiff

# Model Defaults
DEFAULT_CONF_THRESHOLD=0.8
DEFAULT_IMAGE_SIZE=640
DEFAULT_OVERLAP_RATIO=0.2

# Gemini / OpenRouter (required when using model=gemini)
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-3-flash-preview
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TEMPERATURE=0.7

# Cache Configuration
RESULTS_CACHE_MAX_SIZE=100
RESULTS_CACHE_TTL=3600
MODEL_CACHE_MAX_SIZE=10

# Cleanup Configuration
PREDICTION_IMAGE_TTL_HOURS=24
CLEANUP_INTERVAL_MINUTES=60

# OCR Configuration
OCR_CACHE_ENABLED=true
OCR_LANGUAGES=en
OCR_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=garnet.log

# Paths
PREDICTIONS_DIR=static/images/predictions
```

## Frontend Configuration (`frontend/.env.local`):

```bash
# API Configuration
VITE_API_URL=http://localhost:8001

# Development Server
VITE_PORT=5173
VITE_HOST=localhost

# Build Configuration
VITE_SOURCEMAP=false
VITE_OUT_DIR=dist
```