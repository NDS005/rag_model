FROM python:3.10-slim

# Install system dependencies required by Unstructured for PDF processing
# libglib2.0-0 provides the missing libgthread-2.0.so.0
# poppler-utils is essential for PDF parsing by Unstructured
# tesseract-ocr handles images within PDFs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variable and expose port
ENV PORT=8080
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]