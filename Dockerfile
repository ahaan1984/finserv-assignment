FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only the dependency definition file first to leverage Docker cache
COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    "easyocr>=1.7.2",   \
    "fastapi>=0.115.12",\
    "ipykernel>=6.29.5",\
    "matplotlib>=3.10.1",\
    "numpy>=2.2.5",\
    "opencv-python>=4.11.0.86",\
    "pandas>=2.2.3",\
    "pillow>=11.2.1",\
    "pydantic>=2.11.3",\
    "pytesseract>=0.3.13",\
    "python-multipart>=0.0.20",\
    "torch>=2.7.0",\
    "transformers>=4.51.3",\
    "uvicorn>=0.34.2",

COPY hello.py .
COPY test.py .
COPY models.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the application using uvicorn
# Use 0.0.0.0 to ensure it's accessible from outside the container
# Do not use --reload in production images
CMD ["uvicorn", "hello:app", "--host", "0.0.0.0", "--port", "8000"]
