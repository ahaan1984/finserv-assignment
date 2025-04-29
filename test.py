import easyocr
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import os
import io
from PIL import Image
import re # Import the regex module

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

        # Extract text from results
        extracted_text = []
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            extracted_text.append((text, confidence))

        # Process specific content based on image type using regex
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

    # Combine extracted text lines into a single string
    full_text = "\n".join([item[0] for item in extracted_text])

    # --- Regex Patterns ---
    # Pattern to find the start of the CBC section and the header row
    # Looks for "COMPLETE BLOOD COUNT" followed by lines until the header "Test Result Unit Biological Ref" (allowing variations)
    cbc_header_pattern = re.compile(
        r"COMPLETE BLOOD COUNT.*?\n.*?" # Find the title and potentially following lines
        r"(?:Test\s+Result\s+Unit\s+Biological\s+Ref)", # Find the header row
        re.IGNORECASE | re.DOTALL # Ignore case and allow '.' to match newline
    )

    # Pattern to find the end of the CBC section
    end_section_pattern = re.compile(
        r"(ABSOLUTE\s+COUNTS|Method|End\s+Of\s+Report|DIFFERENTIAL\s+COUNT)", # Keywords indicating end of CBC table
        re.IGNORECASE
    )

    # Pattern to extract individual test rows
    # Captures: Test Name, Value (with optional [L]/[H]), Unit (optional), Ref Range
    # Assumes Test Name starts the line, followed by value, optional unit, and range.
    # Allows for variations in spacing and common unit formats.
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


    # --- Extraction Logic ---
    header_match = cbc_header_pattern.search(full_text)
    if not header_match:
        print("CBC header pattern not found.")
        return result # CBC header not found

    # Start searching for tests after the header
    start_pos = header_match.end()

    # Find the end position of the CBC section
    end_match = end_section_pattern.search(full_text, pos=start_pos)
    end_pos = end_match.start() if end_match else len(full_text)

    # Extract the text block likely containing the CBC results
    cbc_block = full_text[start_pos:end_pos]

    # Find all test rows in the block using the primary pattern
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

        # Basic validation/cleanup (optional but recommended)
        if not test_name or not test_value or not ref_range:
            continue # Skip if essential parts are missing

        # Refine unit extraction if needed (e.g., check against known units)
        known_units = ['%', 'gm/dl', 'g/dl', 'fl', 'pg', 'fL', '/ul', '/uL', 'mill/cmm', '10^3/uL', '10^6/uL']
        if unit.lower() not in [u.lower() for u in known_units] and unit:
             # If the captured 'unit' doesn't look like a unit, it might be part of the range
             # This logic might need adjustment based on common formats
             if re.match(r'^[\d.\s-]+$', unit): # If it looks like part of the range
                 ref_range = f"{unit} {ref_range}".strip()
                 unit = "" # Reset unit
             # else: keep it as potential unit for now

        test_data = {
            "test_name": test_name,
            "test_value": test_value,
            "bio_reference_range": ref_range,
            "test_unit": unit,
            "lab_test_out_of_range": out_of_range
        }
        result["data"].append(test_data)

    # If the primary pattern found nothing, try the simpler pattern (no distinct unit column)
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


    # Mark success if we found any tests
    result["is_success"] = len(result["data"]) > 0
    if not result["is_success"]:
        print("No test data extracted using regex patterns.")

    return result


# This function is kept for potential use if the old method is needed elsewhere
# or for comparison, but it's not used by process_medical_reports anymore.
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
    start_index = -1
    result_index = -1
    unit_index = -1
    ref_index = -1

    # Search for headers starting from the CBC title line
    for i in range(cbc_index, len(texts)):
        # Check if the line contains the expected headers (flexible check)
        line_text = texts[i].lower() # Use lower case for matching
        if "test" in line_text and "result" in line_text and "unit" in line_text and "biological ref" in line_text:
            # Attempt to find relative positions (this is fragile)
            # A better approach would use regex on the line or fixed column assumptions if layout is consistent
            start_index = i # Assume header is on this line
            # Placeholder indices - real logic would need column detection
            result_index = i
            unit_index = i
            ref_index = i
            header_indices = [start_index, result_index, unit_index, ref_index] # Mark header found
            break # Found header row

    if not header_indices:
        print("Column headers not found using simple text search.")
        return result  # Column headers not found

    # Process tests
    i = header_indices[0] + 1  # Start after headers

    while i < len(texts):
        # End processing at certain known sections
        current_text_upper = texts[i].upper()
        if "ABSOLUTE COUNTS" in current_text_upper or \
           "METHOD" in current_text_upper or \
           "END OF REPORT" in current_text_upper or \
           "DIFFERENTIAL COUNT" in current_text_upper: # Added Differential Count as end marker
            break

        test_name = texts[i]

        # Skip known non-test items or subheaders more robustly
        if test_name.upper() in ["DIFFERENTIAL COUNT:", "PLATELETS", "ABSOLUTE COUNTS"]:
            i += 1
            continue

        # Check if the next item looks like a numeric value possibly with range markers
        # This logic is prone to errors if layout varies
        value_index = i + 1
        unit_index = i + 2
        ref_index = i + 3 # Assumed order

        test_value = ""
        out_of_range = False
        unit = ""
        ref_range = ""

        if value_index < len(texts):
            potential_value = texts[value_index]
            # Check if it looks like a number, possibly with [L] or [H]
            if re.match(r'^[\d.]+(\s*\[[LH]\])?$', potential_value):
                test_value = potential_value
                if '[' in test_value:
                    out_of_range = True
                # Clean test value
                test_value = re.sub(r'\s*\[[LH]\]', '', test_value).strip()

                # Try to find unit
                if unit_index < len(texts):
                    potential_unit = texts[unit_index]
                    # Check against known units or typical patterns
                    known_units = ['%', 'gm/dl', 'g/dl', 'fl', 'pg', 'fL', '/ul', '/uL', 'mill/cmm', '10^3/uL', '10^6/uL']
                    if potential_unit.lower() in [u.lower() for u in known_units] or re.match(r'^[a-zA-Z/%]+$', potential_unit):
                         unit = potential_unit
                    else:
                         ref_index = unit_index # Assume unit is missing, next item is ref range

                # Try to find reference range
                if ref_index < len(texts):
                    potential_ref = texts[ref_index]
                    # Reference ranges typically contain numbers and hyphens
                    if '-' in potential_ref and any(c.isdigit() for c in potential_ref):
                        ref_range = potential_ref
                    # Handle case where ref range might be on the same line as unit if unit was short
                    elif unit and '-' in unit and any(c.isdigit() for c in unit):
                         ref_range = unit # Unit was actually the start of the range
                         unit = ""

                # Add test to results only if value and range seem plausible
                if test_value and ref_range:
                    test_data = {
                        "test_name": test_name.strip(),
                        "test_value": test_value,
                        "bio_reference_range": ref_range.strip(),
                        "test_unit": unit.strip(),
                        "lab_test_out_of_range": out_of_range
                    }
                    result["data"].append(test_data)
                    # Advance index based on what was consumed (value, unit, range)
                    i = ref_index # Move past the consumed items
                else:
                    i += 1 # Move to next line if no valid test found here
            else:
                i += 1 # Next item didn't look like a value, move to next line
        else:
            break # Reached end of text

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
        ocr_results = reader.readtext(image_rgb, detail=1, paragraph=False) # Process line by line
        print(f"OCR found {len(ocr_results)} text blocks.")

        # Extract text with confidence
        extracted_text = [(detection[1], detection[2]) for detection in ocr_results]
        print("Processing extracted text using regex...")
        # Use the regex-based processing function
        report_data = process_report_content_regex(extracted_text)

        # Format the final output dictionary
        if report_data["is_success"]:
             # Structure matches Output model: {"is_success": True, "data": {"results": [...]}}
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


# Main execution
def main():
    # Define paths to the images
    image_paths = [
        'lbmaske/AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_7.png',  # Replace with actual paths
        # Add other image paths here if needed
        # 'lbmaske/BLR-0425-PA-0037318_SASHANK P K 0037318 2 OF 2_28-04-2025_1007-19_AM@E.pdf_page_29.png',
        # 'lbmaske/BLR-0425-PA-0039192_05c45741fa5d4b5180df06f200423a00__2_files_merged__26-04-2025_0430-01_PM@E.pdf_page_104.png',
    ]

    # Check if paths exist
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        print("Error: None of the specified image paths exist.")
        print("Please check the paths:")
        for p in image_paths:
            print(f"- {p}")
        return
    elif len(valid_paths) < len(image_paths):
        print("Warning: Some image paths were not found.")


    # Process the images
    results = process_medical_reports(valid_paths) # Use only valid paths

    # Display the results using the updated display function
    display_results(results)

    # Optionally print the raw results dictionary
    # print("\nRaw Results Dictionary:")
    # import json
    # print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()