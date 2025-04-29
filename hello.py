from fastapi import FastAPI, HTTPException, File, UploadFile # Make sure File and UploadFile are imported
from fastapi.middleware.cors import CORSMiddleware
import base64
import traceback
import io

from models import ImageInput, Output, OutputData, TestResult # Import necessary models
# Keep ImageInput if other endpoints use it, otherwise it can be removed from imports if not used elsewhere
from test import process_single_image_bytes # Import the processing function

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers
)

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

# Modified endpoint to accept file upload
@app.post("/get-lab-tests", response_model=Output)
async def get_lab_tests(file: UploadFile = File(...)): # Changed input to UploadFile
    """
    Accepts an image file upload, processes it using OCR,
    and returns extracted lab test results.
    """
    try:
        # Read image bytes directly from the uploaded file
        image_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes") # Log received file info

        result_dict = process_single_image_bytes(image_bytes)

        if result_dict["is_success"] and result_dict.get("data"):
            try:
                validated_results = [TestResult(**item) for item in result_dict["data"]["results"]]
                output_data = OutputData(results=validated_results)
                response = Output(is_success=True, data=output_data, error=None)
                print("Successfully processed image and extracted data.")
            except Exception as validation_error:
                print(f"Data validation error: {validation_error}")
                print(f"Data causing validation error: {result_dict.get('data')}")
                raise HTTPException(status_code=500, detail=f"Internal server error during data validation: {validation_error}")
        else:
            error_message = result_dict.get("error", "Processing failed or no data found.")
            print(f"Processing failed or no data found: {error_message}")
            response = Output(is_success=False, data=None, error=error_message)

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /get-lab-tests endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred processing the image.")
    finally:
        # Ensure the file is closed
        await file.close()


if __name__ == "__main__":
    import uvicorn
    # Reload=True is useful for development
    uvicorn.run("hello:app", host="0.0.0.0", port=8000, reload=True)