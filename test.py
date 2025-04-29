import easyocr
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import os
import io
from PIL import Image
import re 

def process_medical_reports(image_paths):
    reader = easyocr.Reader(['en'])

    all_results = {}

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}: {os.path.basename(img_path)}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False, # Process line by line
            decoder='greedy',
            beamWidth=5,
            batch_size=4,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4
        )

        extracted_text = []
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            extracted_text.append((text, confidence))

        report_data = process_report_content_regex(extracted_text, i+1) # Use the regex version
        all_results[f"Image_{i+1}"] = report_data

        print(f"Completed processing image {i+1}\n")

    return all_results

def process_report_content_regex(extracted_text, image_index=1):
    """Processes extracted text using regex to find CBC results."""
    result = {
        "is_success": False,
        "data": []
    }

    full_text = "\n".join([item[0] for item in extracted_text])
    cbc_header_pattern = re.compile(
        r"COMPLETE BLOOD COUNT.*?\n.*?" # Find the title and potentially following lines
        r"(?:Test\s+Result\s+Unit\s+Biological\s+Ref)", # Find the header row
        re.IGNORECASE | re.DOTALL # Ignore case and allow '.' to match newline
    )

    end_section_pattern = re.compile(
        r"(ABSOLUTE\s+COUNTS|Method|End\s+Of\s+Report|DIFFERENTIAL\s+COUNT)", # Keywords indicating end of CBC table
        re.IGNORECASE
    )

    test_row_pattern = re.compile(
        r"^\s*([A-Za-z\s\/().-]+?)\s+"       # 1: Test Name (letters, spaces, /, (), ., -)
        r"([\d.]+(?:\s*\[[LH]\])?)\s*"      # 2: Value (digits, dot) with optional [L] or [H] marker
        r"([%a-zA-Z\/dLμugmfLcm. ]*?)\s+"   # 3: Unit (optional, common chars/formats, non-greedy)
        r"([\d.\s-]+)$",                    # 4: Reference Range (digits, dot, space, hyphen) at end of line
        re.MULTILINE | re.IGNORECASE
    )
    # Simpler pattern if unit is often missing or merged with range
    test_row_pattern_no_unit = re.compile(
        r"^\s*([A-Za-z\s\/().-]+?)\s+"       # 1: Test Name
        r"([\d.]+(?:\s*\[[LH]\])?)\s+"      # 2: Value (with optional [L]/[H])
        r"([\d.\s-]+)$",                    # 3: Reference Range (assuming no distinct unit column)
        re.MULTILINE | re.IGNORECASE
    )

    header_match = cbc_header_pattern.search(full_text)
    if not header_match:
        print("CBC header pattern not found.")
        return result # CBC header not found

    start_pos = header_match.end()

    end_match = end_section_pattern.search(full_text, pos=start_pos)
    end_pos = end_match.start() if end_match else len(full_text)

    cbc_block = full_text[start_pos:end_pos]

    matches = test_row_pattern.finditer(cbc_block)
    found_tests = False
    for match in matches:
        found_tests = True
        test_name = match.group(1).strip()
        raw_value = match.group(2).strip()
        unit = match.group(3).strip() if match.group(3) else "" # Handle optional unit
        ref_range = match.group(4).strip()

        # Check for out-of-range markers and clean value
        out_of_range = False
        test_value = raw_value
        if '[L]' in raw_value or '[H]' in raw_value:
            out_of_range = True
            test_value = re.sub(r'\s*\[[LH]\]', '', raw_value).strip() # Remove marker

        if not test_name or not test_value or not ref_range:
            continue # Skip if essential parts are missing

        known_units = ['%', 'gm/dl', 'g/dl', 'fl', 'pg', 'fL', '/ul', '/uL', 'mill/cmm', '10^3/uL', '10^6/uL']
        if unit.lower() not in [u.lower() for u in known_units] and unit:
             if re.match(r'^[\d.\s-]+$', unit): # If it looks like part of the range
                 ref_range = f"{unit} {ref_range}".strip()
                 unit = "" # Reset unit

        test_data = {
            "test_name": test_name,
            "test_value": test_value,
            "bio_reference_range": ref_range,
            "test_unit": unit,
            "lab_test_out_of_range": out_of_range
        }
        result["data"].append(test_data)

    if not found_tests:
        print("Primary test row pattern found no matches, trying pattern without distinct unit...")
        matches = test_row_pattern_no_unit.finditer(cbc_block)
        for match in matches:
            found_tests = True
            test_name = match.group(1).strip()
            raw_value = match.group(2).strip()
            ref_range = match.group(3).strip()
            unit = "" # No unit captured by this pattern

            out_of_range = False
            test_value = raw_value
            if '[L]' in raw_value or '[H]' in raw_value:
                out_of_range = True
                test_value = re.sub(r'\s*\[[LH]\]', '', raw_value).strip()

            if not test_name or not test_value or not ref_range:
                continue

            test_data = {
                "test_name": test_name,
                "test_value": test_value,
                "bio_reference_range": ref_range,
                "test_unit": unit,
                "lab_test_out_of_range": out_of_range
            }
            result["data"].append(test_data)

    result["is_success"] = len(result["data"]) > 0
    if not result["is_success"]:
        print("No test data extracted using regex patterns.")

    return result


def process_single_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    try:
        reader = easyocr.Reader(['en']) # Add GPU=True if CUDA is available and configured
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: Failed to decode image bytes.")
            return {"is_success": False, "data": None, "error": "Failed to decode image data."}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Performing OCR...")
        ocr_results = reader.readtext(image_rgb, detail=1, paragraph=False) # Process line by line
        print(f"OCR found {len(ocr_results)} text blocks.")
        extracted_text = [(detection[1], detection[2]) for detection in ocr_results]
        print("Processing extracted text using regex...")
        report_data = process_report_content_regex(extracted_text)

        if report_data["is_success"]:
             return {"is_success": True, "data": {"results": report_data["data"]}, "error": None}
        else:
             print("No relevant data found by process_report_content_regex.")
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
        print(f"=== Results for {image_key} ===")
        if data.get("is_success"):
            print("Status: Success")
            print("\nTest Results:")
            if data.get("data"):
                for test in data["data"]:
                    print(f"  • Test: {test.get('test_name', 'N/A')}")
                    print(f"    - Value: {test.get('test_value', 'N/A')} {test.get('test_unit', '')}".strip())
                    print(f"    - Range: {test.get('bio_reference_range', 'N/A')}")
                    print(f"    - Out of Range: {test.get('lab_test_out_of_range', 'N/A')}")
            else:
                print("  No test data extracted.")
        else:
            print("Status: Failed or No Data Found")
            print(f"Error: {data.get('error', 'Unknown error')}")

        print("\n" + "="*50 + "\n")

#test
def main():
    image_paths = [
        'lbmaske/AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_7.png',  # Replace with actual paths
    ]

    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        print("Error: None of the specified image paths exist.")
        print("Please check the paths:")
        for p in image_paths:
            print(f"- {p}")
        return
    elif len(valid_paths) < len(image_paths):
        print("Warning: Some image paths were not found.")


    results = process_medical_reports(valid_paths) # Use only valid paths

    display_results(results)

if __name__ == "__main__":
    main()