# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (for packages like torch, transformers, opencv, pyaudio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    portaudio19-dev \
    libasound-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements.txt first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project files into the container
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]