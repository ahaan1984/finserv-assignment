import easyocr
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import os
import io 
from PIL import Image

def process_medical_reports(image_paths):
    # Initialize the EasyOCR reader with English language
    reader = easyocr.Reader(['en'])
    
    all_results = {}
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}: {os.path.basename(img_path)}")
        
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # Convert to RGB (EasyOCR expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform OCR with EasyOCR
        # Adjust parameters for better accuracy with medical documents
        results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False,
            decoder='greedy',
            beamWidth=5,
            batch_size=4,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4
        )
        
        # Extract text from results
        extracted_text = []
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            extracted_text.append((text, confidence))
            
        # Process specific content based on image type
        report_data = process_report_content(extracted_text, i+1)
        all_results[f"Image_{i+1}"] = report_data
        
        print(f"Completed processing image {i+1}\n")
    
    return all_results

def process_report_content(extracted_text, image_index=1):
    # Initialize result structure
    result = {
        "is_success": False,
        "data": []
    }
    
    # Extract texts only (without confidence scores)
    texts = [item[0] for item in extracted_text]
    
    # Find test names, values, units and reference ranges
    test_pattern = []
    
    # Locate CBC section header
    cbc_index = -1
    for i, text in enumerate(texts):
        if "COMPLETE BLOOD COUNT" in text:
            cbc_index = i
            break
    
    if cbc_index == -1:
        return result  # CBC section not found
    
    # Find column headers to determine start of data
    header_indices = []
    for i in range(cbc_index, len(texts)):
        if texts[i] == "Test":
            start_index = i
            # Find other headers
            for j in range(i, i+4):
                if j < len(texts):
                    if texts[j] == "Result":
                        result_index = j
                    elif texts[j] == "Unit":
                        unit_index = j
                    elif "Biological Ref" in texts[j]:
                        ref_index = j
            header_indices = [start_index, result_index, unit_index, ref_index]
            break
    
    if not header_indices:
        return result  # Column headers not found
    
    # Process tests
    i = header_indices[0] + 1  # Start after headers
    
    while i < len(texts):
        # End processing at certain known sections
        if "ABSQLUTE CQUNTS" in texts[i] or "Method" in texts[i] or "End Of Report" in texts[i]:
            break
            
        test_name = texts[i]
        
        # Skip non-test items
        if test_name in ["DHFFERENTIAL CQUNI:", "PLATELETS"]:
            i += 1
            continue
            
        # Check if this is likely a test name
        if i+1 < len(texts) and ('[' in texts[i+1] or texts[i+1].replace('.', '', 1).isdigit()):
            test_value = texts[i+1]
            out_of_range = False
            
            # Check if value is out of range
            if '[' in test_value:
                out_of_range = True
                
            # Clean test value
            test_value = test_value.replace('[L]', '').replace('[H]', '').strip()
            
            # Find unit (usually in the next position)
            unit = ""
            if i+2 < len(texts):
                potential_unit = texts[i+2]
                if potential_unit in ['%', 'gm/dl', 'g/dl', 'fl', 'Pg', 'fL', '/ul', '/uL', 'Iul', 'luL', 'millcmm']:
                    unit = potential_unit
                    i += 1  # Move to unit position
            
            # Find reference range (usually follows the unit)
            ref_range = ""
            if i+2 < len(texts):
                potential_ref = texts[i+2]
                # Reference ranges typically contain numbers and hyphens
                if '-' in potential_ref and any(c.isdigit() for c in potential_ref):
                    ref_range = potential_ref
                    i += 1  # Move to ref position
            
            # Add test to results
            test_data = {
                "test_name": test_name,
                "test_value": test_value,
                "bio_reference_range": ref_range,
                "test_unit": unit,
                "lab_test_out_of_range": out_of_range
            }
            
            result["data"].append(test_data)
            
        i += 1
    
    # Mark success if we found any tests
    result["is_success"] = len(result["data"]) > 0
    
    return result

def process_single_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Processes a single image provided as bytes and returns structured results."""
    try:
        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en']) # Add GPU=True if CUDA is available and configured

        # Decode image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: Failed to decode image bytes.")
            return {"is_success": False, "data": None, "error": "Failed to decode image data."}

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform OCR
        print("Performing OCR...")
        # Using default parameters first, adjust if needed based on results
        ocr_results = reader.readtext(image_rgb, detail=1, paragraph=False)
        print(f"OCR found {len(ocr_results)} text blocks.")

        # Extract text with confidence
        extracted_text = [(detection[1], detection[2]) for detection in ocr_results]
        print("Processing extracted text...")
        report_data = process_report_content(extracted_text)

        # Format the final output dictionary
        if report_data["is_success"]:
             # Structure matches Output model: {"is_success": True, "data": {"results": [...]}}
             return {"is_success": True, "data": {"results": report_data["data"]}, "error": None}
        else:
             print("No relevant data found by process_report_content.")
             return {"is_success": False, "data": None, "error": "No relevant lab test data found in the image."}

    except cv2.error as e:
        print(f"OpenCV Error processing image: {e}")
        return {"is_success": False, "data": None, "error": f"Image processing error: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error during image processing: {e}")
        import traceback
        traceback.print_exc() # Log detailed error for server-side debugging
        return {"is_success": False, "data": None, "error": f"An unexpected error occurred: {str(e)}"}

def display_results(results):
    """Print extracted information in a structured format"""
    for image_key, data in results.items():
        print(f"=== {image_key}: {data['report_type']} ===")
        
        print("\nPatient Information:")
        for key, value in data['patient_info'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\nTest Results:")
        for test in data['test_results']:
            print(f"  • {test.get('test_name', '')}")
            for k, v in test.items():
                if k != 'test_name':
                    print(f"    - {k.replace('_', ' ').title()}: {v}")
        
        print("\n" + "="*50 + "\n")

# Main execution
def main():
    # Define paths to the images
    image_paths = [
        'lbmaske/AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_7.png',  # Replace with actual paths
        # 'lbmaske/BLR-0425-PA-0037318_SASHANK P K 0037318 2 OF 2_28-04-2025_1007-19_AM@E.pdf_page_29.png',
        # 'lbmaske/BLR-0425-PA-0039192_05c45741fa5d4b5180df06f200423a00__2_files_merged__26-04-2025_0430-01_PM@E.pdf_page_104.png',
    ]
    
    # Process the images
    results = process_medical_reports(image_paths)
    
    # Display the results
    # display_results(results)
    
    print (results)

if __name__ == "__main__":
    main()