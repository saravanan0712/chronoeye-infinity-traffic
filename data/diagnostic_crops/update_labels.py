import csv
import collections

gt_map = {
    "plate_crop_001.png": "KW527",
    "plate_crop_002.png": "SX8525",
    "plate_crop_003.png": "",
    "plate_crop_004.png": "XQ1147",
    "plate_crop_005.png": "KW527", 
    "plate_crop_006.png": "XQ1147",
    "plate_crop_007.png": "SX8525",
    "plate_crop_008.png": "",
    "plate_crop_009.png": "",
    "plate_crop_010.png": "",
    "plate_crop_011.png": "",
    "plate_crop_012.png": "",
    "plate_crop_013.png": "",
    "plate_crop_014.png": "",
    "plate_crop_015.png": "",
    "plate_crop_016.png": "",
    "plate_crop_017.png": ""
}

# The indian formats are AA00AA0000 etc. All the readable ones above are non-Indian.
category_map = {k: "non-Indian plate" if v != "" else "unreadable/poor crop" for k, v in gt_map.items()}

csv_file = r"E:\chronoeye\data\diagnostic_crops\anpr_verification.csv"
rows = []
with open(csv_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        crop_id = row['crop_id']
        gt = gt_map.get(crop_id, "")
        
        row['human_ground_truth'] = gt
        
        if gt == "":
            row['human_result'] = "UNCLEAR"
        else:
            if row['normalized_ocr'] == gt:
                row['human_result'] = "CORRECT"
            else:
                row['human_result'] = "WRONG"
                
        rows.append(row)

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV Updated successfully.")

# summary categories
summary = collections.defaultdict(int)

for row in rows:
    crop_id = row['crop_id']
    gt = row['human_ground_truth']
    res = row['human_result']
    indian_fmt = row['indian_format']
    
    if res == "UNCLEAR":
        summary["unreadable/poor crop"] += 1
    elif res == "CORRECT":
        # Check if it was rejected by indian validator
        # indian_format might be INVALID or FORMAT_MISMATCH
        if indian_fmt != "VALID":
            summary["OCR correct but Indian validator rejected"] += 1
        else:
            if gt in ("KW527", "SX8525", "XQ1147"):
                # it shouldn't be valid, but if it is, maybe it hallucinated.
                # But if res == CORRECT, then normalized_ocr is exactly "KW527".
                # A validator would reject "KW527" as INVALID.
                pass
            else:
                summary["correct Indian plate"] += 1
    elif res == "WRONG":
        summary["OCR wrong"] += 1
        
    # Also separate count by actual plate type
    if gt != "":
        # We manually know all in this benchmark are non-Indian formatted actual plates
        summary["non-Indian plate (total unique or occurances)"] += 1

print("\n--- SUMMARY ---")
for k, v in summary.items():
    print(f"{k}: {v}")
