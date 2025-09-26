FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    libopenblas-dev \\
    libhdf5-dev \\
    libxml2-dev \\
    libxslt1-dev \\
    zlib1g-dev \\
    gfortran \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .

# Install project dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -e .

# Copy project code
COPY src/ ./src/
COPY README.md README_zh.md LICENSE ./

# Create necessary directories for VASPilot
RUN mkdir -p /app/configs /app/data/uploads /app/data/record /app/data/work /app/data/downloads /app/data/mcp_work /app/data/memory

# Expose ports for the different services
EXPOSE 8933 51293

# Create a default config directory
RUN mkdir -p /app/configs

# Create a default start script
RUN echo '#!/bin/bash\\n\\
echo \"Starting VASPilot services...\"\\n\\
echo \"Usage: docker run -p 8933:8933 -p 51293:51293 -v $(pwd)/configs:/app/configs -v $(pwd)/data:/app/data -e PMG_VASP_PSP_DIR=/path/to/your/POTCARS -e MP_API_KEY=your_mp_key -e LLM_BASE_URL=http://your.llm.server:port/v1 -e LLM_API_KEY=your_api_key -e EMBEDDER_BASE_URL=http://your.embedder.server:port/v1/embeddings vaspilot\"\\n\\
exec \"$@\"' > /app/start.sh && chmod +x /app/start.sh

# Set the default command
CMD [\"/app/start.sh\"]