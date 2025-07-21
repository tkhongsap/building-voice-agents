# Multi-stage Dockerfile for LiveKit Voice Agents Platform
# Optimized for production deployment with security and performance considerations

# Build stage - Use Python slim image for smaller size
FROM python:3.11-slim as builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home agent

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the application
RUN pip install -e .

# Run tests to ensure build is valid
RUN python -m pytest --tb=short

# Production stage - Use smaller runtime image
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/agent/.local/bin:$PATH" \
    PYTHONPATH="/app"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing libraries
    ffmpeg \
    libsndfile1 \
    # SSL/TLS support
    ca-certificates \
    # Process monitoring
    procps \
    # Network tools for debugging
    curl \
    netcat-openbsd \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home agent

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=agent:agent . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R agent:agent /app

# Switch to non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8443

# Default command
CMD ["python", "-m", "src.agents.voice_assistant", "--config", "/app/configs/production.yaml"]

# Development stage - For local development
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER agent

# Set up development environment
WORKDIR /app

# Default command for development
CMD ["python", "-m", "src.agents.voice_assistant", "--config", "/app/configs/dev.yaml", "--hot-reload"]