from pydantic import BaseModel, Base64Bytes, Field
from typing import List, Optional

class ImageInput(BaseModel):
    image_data: Base64Bytes = Field(..., description="Base64 encoded image data")

class TestResult(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

class OutputData(BaseModel):
    results: List[TestResult]

class Output(BaseModel):
    is_success: bool
    data: Optional[OutputData] = None
    error: Optional[str] = None