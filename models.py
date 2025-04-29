from pydantic import BaseModel, Base64Bytes, Field
from typing import List, Dict, Any, Optional

class ImageInput(BaseModel):
    """Model for accepting base64 encoded image data."""
    image_data: Base64Bytes = Field(..., description="Base64 encoded image data")

class TestResult(BaseModel):
    """Model for a single lab test result."""
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

class OutputData(BaseModel):
    """Model for the structured data part of the output."""
    results: List[TestResult]

class Output(BaseModel):
    """Model for the final API response."""
    is_success: bool
    data: Optional[OutputData] = None # Use the refined OutputData, allow None for failure
    error: Optional[str] = None # Add an optional error field