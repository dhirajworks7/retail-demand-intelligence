# Use Python 3.12 to match the project's supported and CI environment.
FROM python:3.12-slim

# Prevent Python from writing .pyc files and ensure logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# All following commands run from this directory inside the container.
WORKDIR /app

# Install system libraries required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first so Docker can cache this layer when source code changes.
COPY requirements.txt .

# Install the project's Python dependencies without retaining pip's download cache.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the image.
COPY api/ api/
COPY src/ src/

# Document the port used by the FastAPI service.
EXPOSE 8000

# Start the forecasting API when the container launches.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]