# Lab Report OCR Processing API
A FastAPI service that extracts lab test results from medical report images using Optical Character Recognition (OCR) and structured data extraction.

# Overview
This service provides an API endpoint that accepts medical lab report images (like CBC reports) and returns structured data extracted from those images. It uses EasyOCR for text extraction and regex pattern matching to identify test results.

# Features
Image processing using OpenCV and EasyOCR
Structured data extraction for Complete Blood Count (CBC) reports
FastAPI REST API with CORS support
Docker containerization for easy deployment

# API Endpoint

## POST /get-lab-tests
Upload and process a lab report image.

Request:
Parameter: file (image file)

Response:
{
  "is_success": true,
  "data": {
    "results": [
      {
        "test_name": "HEMOGLOBIN",
        "test_value": "14.2",
        "bio_reference_range": "13.0 - 17.0",
        "test_unit": "g/dl",
        "lab_test_out_of_range": false
      },
      // Additional test results...
    ]
  },
  "error": null
}

# Limitations
The OCR process requires clear, high-resolution images
Currently optimized for specific CBC report formats
Processing may be slower without GPU acceleration

# Dependencies
Major dependencies include:

FastAPI - Web framework
EasyOCR - Optical Character Recognition
OpenCV - Image processing
PyTorch - Machine learning (used by EasyOCR)
Pydantic - Data validation