from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
import traceback
import io

from models import ImageInput, Output, OutputData, TestResult
from test import process_single_image_bytes

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/get-lab-tests", response_model=Output)
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")

        result_dict = process_single_image_bytes(image_bytes)

        if result_dict["is_success"] and result_dict.get("data"):
            try:
                validated_results = [TestResult(**item) for item in result_dict["data"]["results"]]
                output_data = OutputData(results=validated_results)
                response = Output(is_success=True, data=output_data, error=None)
                print("Successfully processed image and extracted data.")
            except Exception as validation_error:
                print(f"Data validation error: {validation_error}")
                for item in result_dict.get("data", {}).get("results", []):
                    print(f"Problematic item keys: {list(item.keys())}")
                raise HTTPException(status_code=500, detail=f"Internal server error during data validation: {validation_error}")
        else:
            error_message = result_dict.get("error", "Processing failed or no data found.")
            print(f"Processing failed or no data found: {error_message}")
            response = Output(is_success=False, data=None, error=error_message)

        print(response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /get-lab-tests endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred processing the image.")
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hello:app", host="0.0.0.0", port=8000, reload=True)