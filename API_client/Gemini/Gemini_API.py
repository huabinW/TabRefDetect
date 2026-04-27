import json
import os
import time
import re
import ast
from openai import OpenAI
from PIL import Image
import traceback
import base64

# Base paths
base_path = "./TabRefError/citekeypdf"
json_file = "./TabRefError/biaoge.json"
originalkey_base_path = "./TabRefError/originalkey"

# Error log file path
ERROR_LOG_FILE = "./TabRefError/processing_errors2.jsonl"

# Initialize OpenAI client
client = OpenAI(
    api_key="your_api_key",
    base_url="https://aihubmix.com/v1",
)

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def image_to_base64(image_path):
    """Convert local image to base64 data URL"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def to_safe_string(value, field_name="value"):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    print(f"Warning: {field_name} is not a string (type: {type(value)}), converted to string")
    try:
        return str(value)
    except Exception as e:
        print(f"Failed to convert {field_name} to string: {e}")
        return ""


def remove_control_chars(s):
    return re.sub(r'[\x00-\x1f\x7f-\x9f]+', ' ', s)


def extract_json(text):
    """Extract the outermost valid JSON object from text using balanced brace matching."""
    text = remove_control_chars(text.strip())

    # Remove markdown code blocks
    if text.startswith("```"):
        parts = re.split(r"```(?:json)?", text, maxsplit=2)
        if len(parts) >= 2:
            text = parts[1].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # Find the outermost balanced JSON object
    depth = 0
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
        elif char == '[':
            if depth == 0:
                start = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return text  # fallback


def robust_json_loads(text, paper_id=None, context=""):
    cleaned_text = remove_control_chars(text)

    # Step 1: Try direct JSON load
    try:
        return json.loads(cleaned_text)
    except Exception:
        pass

    # Step 2: Heuristic cleanup
    try:
        fixed = cleaned_text.strip()
        # Remove common prefixes
        if not (fixed.startswith('{') or fixed.startswith('[')):
            idx = fixed.find('{')
            if idx == -1:
                idx = fixed.find('[')
            if idx != -1:
                fixed = fixed[idx:]

        # Ensure proper closing
        if fixed:
            if fixed.startswith('{') and not fixed.endswith('}'):
                last_brace = fixed.rfind('}')
                if last_brace != -1:
                    fixed = fixed[:last_brace + 1]
            elif fixed.startswith('[') and not fixed.endswith(']'):
                last_brace = fixed.rfind(']')
                if last_brace != -1:
                    fixed = fixed[:last_brace + 1]

        # Fix single quotes → double quotes (only if likely JSON)
        if fixed.count("'") > fixed.count('"'):
            fixed = re.sub(r"(?<!\\)'", '"', fixed)

        return json.loads(fixed)
    except Exception:
        pass

    # Step 3: Try ast.literal_eval
    try:
        obj = ast.literal_eval(cleaned_text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # Step 4: Use LLM to repair
    try:
        print("Attempting to repair JSON format using LLM...")
        safe_text_for_prompt = remove_control_chars(text)
        fix_prompt = f"""You are a JSON repair tool. Convert the following text into a strictly valid JSON object. Return ONLY the JSON, no explanations, no markdown.

Text:
{safe_text_for_prompt}

Valid JSON:"""
        completion = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.0
        )
        fixed_json_str = extract_json(completion.choices[0].message.content.strip())
        fixed_json_str = remove_control_chars(fixed_json_str)
        return json.loads(fixed_json_str)
    except Exception as e:
        log_error(paper_id, "JSONRepairFailed", f"LLM JSON repair failed: {str(e)}",
                  traceback_msg=traceback.format_exc(),
                  extra_info={"context": context, "raw_length": len(text)})

    raise ValueError("Unable to parse as valid JSON")


def is_valid_image(image_path):
    try:
        if not os.path.exists(image_path):
            return False
        SUPPORTED_FORMATS = ["JPEG", "JPG", "PNG", "WEBP"]
        with Image.open(image_path) as img:
            img.verify()
            if img.format not in SUPPORTED_FORMATS:
                print(f"Unsupported image format: {image_path}, format: {img.format}")
                return False
        return True
    except (OSError, PermissionError, ValueError) as e:
        print(f"Invalid, corrupted, or inaccessible image: {image_path}, error: {e}")
        return False


def log_error(paper_id, error_type, error_message, traceback_msg="", extra_info=None):
    error_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_id": paper_id or "unknown",
        "error_type": error_type,
        "error_message": str(error_message),
        "traceback": traceback_msg,
    }
    if extra_info:
        error_info.update(extra_info)

    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_info, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Failed to write error log: {e}")


def analyze_image(image_path, author=None, mode="citekey", paper_id=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            base64_url = image_to_base64(image_path)

            if mode == "citekey":
                prompt = f"""You are a JSON-only output system. Return ONLY a valid JSON object. Do NOT include any other text, explanation, or markdown.

Output format:
{{"description": "Detailed description including all key data, models, metrics, and datasets"}}

Analyze the image and output ONLY the JSON:"""
            else:
                prompt = f"""You are a JSON-only output system. Return ONLY a valid JSON object. No other text.

Output format:
{{"description": "Overall description of the image", "author_related_content": "Content related to author '{author}' (empty string if none)"}}

Analyze the image and output ONLY the JSON:"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            completion = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                temperature=0.0
            )

            output_text = completion.choices[0].message.content.strip()
            json_str = extract_json(output_text)

            try:
                response_json = robust_json_loads(json_str, paper_id=paper_id, context=f"analyze_image_{mode}")
                print(f"Analysis succeeded ({mode}, attempt {attempt + 1})")
                return response_json
            except Exception as je:
                print(f"JSON parsing failed (attempt {attempt + 1}): {je}")
                if attempt == max_retries - 1:
                    log_error(paper_id, "JSONParseFailed", f"Failed to parse LLM output: {str(je)}",
                              traceback_msg=traceback.format_exc(),
                              extra_info={"mode": mode, "raw_output": output_text[:500]})
                    if mode == "citekey":
                        return {"description": "Model failed to generate a valid description"}
                    else:
                        return {"description": "Model failed to generate description", "author_related_content": ""}

        except Exception as e:
            print(f"analyze_image failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                log_error(paper_id, "LLMCallFailed", f"Calling Gemini 2.5 Flash failed: {str(e)}",
                          traceback_msg=traceback.format_exc(),
                          extra_info={"mode": mode, "author": author, "image_path": image_path})
            time.sleep(1)
    return {"description": "Unknown error", "author_related_content": "Unknown error"}


def semantic_comparison(claim, originalkey_item, author, paper_id=None):
    claim = to_safe_string(claim, "claim").strip()
    if not isinstance(originalkey_item, dict):
        originalkey_item = {}

    author_content = to_safe_string(originalkey_item.get("author_related_content", ""),
                                    "author_related_content").strip()

    if not author_content:
        return {
            "match": False,
            "score": 0.0,
            "explanation": f"No content related to author '{author}' was found in the original document image; citation cannot be verified."
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = prompt = f"""You are a JSON-only output system. Your entire response must be a single, valid JSON object. Do not include any other text, explanations, markdown, code blocks, or formatting before or after the JSON.

You are a rigorous academic reviewer. Determine whether the following two pieces of content are semantically matched:

[Task Description]
- Content A (citation from current paper): "{claim}"
- Content B (author "{author}"-related content from original document): "{author_content}"

[Scoring Rules]
1. Core matching elements: numerical values (decimals or %), model names, datasets, evaluation metrics.
2. Allow reasonable abbreviations (e.g., "BERT" vs "Bert-base", "ImageNet" vs "IN", "Accuracy" vs "Acc").
3. If Content A fully covers the author-related data in Content B (consistent values, similar models/datasets) → match successful, score ≥ 0.8.
4. If Content A partially misses author-related data (e.g., missing a metric or value) → score between 0.3–0.7, penalized by missing extent.
5. If Content A completely omits author-related content → score = 0.0.
6. Ignore parts of Content B unrelated to the author.

[Output Requirements]
Return ONLY a strict JSON object with:
- "match": true/false
- "score": float between 0.0 and 1.0
- "explanation": concise reason

Now output ONLY the JSON object and nothing else."""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            completion = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                temperature=0.0
            )

            output_text = completion.choices[0].message.content.strip()
            json_str = extract_json(output_text)

            try:
                response_json = robust_json_loads(json_str, paper_id=paper_id, context="semantic_comparison")
                return {
                    "match": bool(response_json.get("match", False)),
                    "score": float(response_json.get("score", 0.0)),
                    "explanation": str(response_json.get("explanation", ""))
                }
            except Exception as je:
                print(f"semantic_comparison JSON parsing failed (attempt {attempt + 1}): {je}")
                if attempt == max_retries - 1:
                    log_error(paper_id, "SemanticComparisonFailed",
                              f"Failed to parse semantic comparison result: {str(je)}",
                              traceback_msg=traceback.format_exc())

        except Exception as e:
            print(f"semantic_comparison failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                log_error(paper_id, "SemanticComparisonFailed", f"Semantic comparison LLM call failed: {str(e)}",
                          traceback_msg=traceback.format_exc())
            time.sleep(1)
    return {"match": False, "score": 0.0, "explanation": "Unknown error"}


def find_content_list_file(citekey):
    citekey_folder_path = os.path.join(base_path, citekey)
    if not os.path.exists(citekey_folder_path):
        print(f"Citekey folder not found: {citekey_folder_path}")
        return None

    for subfolder_name in os.listdir(citekey_folder_path):
        subfolder_path = os.path.join(citekey_folder_path, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        auto_folder_path = os.path.join(subfolder_path, "auto")
        if not os.path.exists(auto_folder_path):
            continue

        for file_name in os.listdir(auto_folder_path):
            if file_name.endswith("content_list.json"):
                content_list_path = os.path.join(auto_folder_path, file_name)
                print(f"Found content_list.json: {content_list_path}")
                return content_list_path

    print(f"content_list.json not found for citekey={citekey}")
    return None


def group_originalkey_images(originalkey_folder_path, charts, author):
    groups = []
    for tab_id in charts:
        if not tab_id.startswith("TAB"):
            continue

        tab_image_path = None
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            path = os.path.join(originalkey_folder_path, tab_id + ext)
            if os.path.exists(path) and is_valid_image(path):
                tab_image_path = path
                break

        if not tab_image_path:
            print(f"TAB image not found: {tab_id}")
            continue

        tab_id_no_space = tab_id.replace(" ", "")
        caption_images = []

        base_caption_name = f"CAPTION {tab_id_no_space}"
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            path = os.path.join(originalkey_folder_path, base_caption_name + ext)
            if os.path.exists(path) and is_valid_image(path):
                caption_images.append(path)
                break

        idx = 1
        while True:
            found = False
            caption_name = f"CAPTION {tab_id_no_space}-{idx}"
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                path = os.path.join(originalkey_folder_path, caption_name + ext)
                if os.path.exists(path) and is_valid_image(path):
                    caption_images.append(path)
                    found = True
                    break
            if not found:
                break
            idx += 1

        if caption_images:
            print(f"Found {len(caption_images)} CAPTION files for {tab_id}")
        else:
            print(f"No CAPTION files found for {tab_id}")

        groups.append({
            "tab_id": tab_id,
            "tab_image": tab_image_path,
            "caption_images": caption_images
        })
    return groups


def process_claim_stream(entry, originalkey_folder_path, charts, author):
    paper_id = entry.get("paper_id", "unknown")
    try:
        label = entry.get("label", "")

        content_list_path = find_content_list_file(entry.get("citekey", ""))
        if not content_list_path:
            log_error(paper_id, "CiteKeyOrContentListNotFound",
                      "Citekey folder or content_list.json not found",
                      extra_info={"citekey": entry.get("citekey", "")})
            return {
                "id": paper_id,
                "label": label,
                "claim": [],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "content_list.json not found"
            }

        content_list_dir = os.path.dirname(content_list_path)
        with open(content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        if not content_list:
            return {
                "id": paper_id,
                "label": label,
                "claim": [],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "content_list is empty"
            }

        citekey_items = [item for item in content_list if item["type"] in ["table"]]
        citekey_images = [os.path.join(content_list_dir, item["img_path"]) for item in citekey_items if
                          "img_path" in item]

        claims_with_info = []
        for image_path in citekey_images:
            if not os.path.exists(image_path):
                print(f"Citekey image not found: {image_path}")
                continue
            if not is_valid_image(image_path):
                print(f"Invalid citekey image: {image_path}")
                continue
            analysis = analyze_image(image_path, mode="citekey", paper_id=paper_id)
            claim = to_safe_string(analysis.get("description", ""), "citekey_description").strip()
            if not claim:
                claim = "Model failed to generate a valid description, but image was processed"
            claims_with_info.append({
                "claim": claim,
                "citekey_image_name": os.path.basename(image_path)
            })

        if not claims_with_info:
            claims_with_info = [{"claim": "No valid citekey images found", "citekey_image_name": "N/A"}]

        tab_groups = group_originalkey_images(originalkey_folder_path, charts, author)
        originalkey_analysis = []
        for group in tab_groups:
            tab_id = group["tab_id"]
            tab_image = group["tab_image"]
            caption_images = group["caption_images"]

            tab_desc = ""
            tab_author = ""
            if os.path.exists(tab_image) and is_valid_image(tab_image):
                tab_analysis = analyze_image(tab_image, author=author, mode="originalkey", paper_id=paper_id)
                tab_desc = to_safe_string(tab_analysis.get("description", ""), "tab_description").strip()
                tab_author = to_safe_string(tab_analysis.get("author_related_content", ""), "tab_author").strip()

            caption_annotations = ""
            for cap_img in caption_images:
                if not os.path.exists(cap_img) or not is_valid_image(cap_img):
                    continue
                caption_prompt = f"""You are a JSON-only output system. Return ONLY valid JSON. No other text.

Output format:
{{"table_annotations": "Supplementary note (single sentence)"}}

Extract explanatory information about the table (e.g., dataset/metric abbreviations, symbol meanings). Do NOT include author conclusions or subjective evaluations.

Output ONLY the JSON:"""
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_to_base64(cap_img)}},
                            {"type": "text", "text": caption_prompt}
                        ]
                    }
                ]
                try:
                    completion = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=messages,
                        temperature=0.0
                    )
                    output_text = completion.choices[0].message.content.strip()
                    try:
                        json_str = extract_json(output_text)
                        parsed = robust_json_loads(json_str, paper_id=paper_id, context="caption_analysis")
                        ann_str = parsed.get("table_annotations", "").strip()
                        if ann_str and ann_str.lower() not in ["", "null", "none"]:
                            caption_annotations = ann_str
                            break
                    except Exception as e:
                        log_error(paper_id, "CaptionParseFailed", f"Failed to parse CAPTION annotation: {str(e)}",
                                  traceback_msg=traceback.format_exc())
                except Exception as e:
                    log_error(paper_id, "CaptionLLMFailed", f"LLM failed to analyze CAPTION: {str(e)}",
                              traceback_msg=traceback.format_exc())
                if caption_annotations:
                    break

            final_description = tab_desc if tab_desc else "Table content could not be extracted"
            if caption_annotations:
                final_description = f"{final_description} (Note: {caption_annotations})"

            final_author_content = tab_author

            if final_description or final_author_content:
                originalkey_analysis.append({
                    "tab_id": tab_id,
                    "analysis": {
                        "description": final_description,
                        "author_related_content": final_author_content
                    }
                })

        if not originalkey_analysis:
            return {
                "id": paper_id,
                "label": label,
                "claim": [info["claim"] for info in claims_with_info],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "No valid originalkey images found for analysis"
            }

        matching_results = []
        for claim_info in claims_with_info:
            for originalkey_item in originalkey_analysis:
                comparison_result = semantic_comparison(
                    claim_info["claim"],
                    originalkey_item["analysis"],
                    author,
                    paper_id=paper_id
                )
                matching_results.append({
                    "citekey_image_name": claim_info["citekey_image_name"],
                    "originalkey_tab_id": originalkey_item["tab_id"],
                    "claim": claim_info["claim"],
                    "originalkey_item": originalkey_item["analysis"],
                    "match_score": float(comparison_result.get("score", 0.0)),
                    "explanation": comparison_result.get("explanation", "")
                })

        matching_results.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "id": paper_id,
            "label": label,
            "claim": [info["claim"] for info in claims_with_info],
            "author": author,
            "matching_results": matching_results,
            "result": "true" if any(r["match_score"] > 0.5 for r in matching_results) else "false",
            "explanation": "Matching completed; see matching_results for details"
        }

    except Exception as e:
        log_error(paper_id, "ProcessFailed", f"Failed to process entry: {str(e)}",
                  traceback_msg=traceback.format_exc(),
                  extra_info={"entry_keys": list(entry.keys())})
        print(f"Processing failed (paper_id={paper_id}): {e}")
        return {
            "id": paper_id,
            "label": entry.get("label", ""),
            "claim": [],
            "author": entry.get("author", ""),
            "matching_results": [],
            "result": "false",
            "explanation": f"Processing error: {str(e)}"
        }


results = []
batch_size = 10

# Load existing intermediate results
last_counter = 0
batch_counter = 1
while True:
    batch_file = f"./gemini-2.5-flash_batch_{batch_counter}.json"
    if os.path.exists(batch_file):
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
                results.extend(batch_results)
            last_counter = batch_counter * batch_size
            print(f"Loaded batch {batch_counter}, processed {last_counter} entries")
            batch_counter += 1
        except Exception as e:
            print(f"Failed to load batch file: {batch_file}, error: {e}")
            break
    else:
        break

print(f"Currently processed: {last_counter} entries")

# Load full dataset
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Continue processing from last_counter
counter = last_counter
for i, entry in enumerate(data[last_counter:], start=last_counter):
    try:
        paper_id = entry.get("paper_id", "")
        citekey = entry.get("citekey", "")
        originalkey = entry.get("originalkey", "")
        charts = entry.get("charts", [])
        author = entry.get("author", "")

        originalkey_folder_path = os.path.join(originalkey_base_path, originalkey)
        if not os.path.exists(originalkey_folder_path):
            print(f"originalkey folder not found: {originalkey_folder_path}")
            result = {
                "id": paper_id,
                "label": entry.get("label", ""),
                "claim": [],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "originalkey folder not found"
            }
            log_error(paper_id, "OriginalKeyNotFound", "originalkey folder not found",
                      extra_info={"originalkey": originalkey})
            results.append(result)
            counter += 1
            continue

        result = process_claim_stream(entry, originalkey_folder_path, charts, author)
        results.append(result)

        print(f"\nProcessed result (entry {i + 1}):")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        counter += 1

        if counter % batch_size == 0:
            output_file_batch = f"./gemini-2.5-flash_batch_{counter // batch_size}.json"
            with open(output_file_batch, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"\nIntermediate results saved to {output_file_batch}")

    except Exception as e:
        paper_id = entry.get("paper_id", "unknown")
        log_error(paper_id, "MainLoopException", f"Main loop processing failed: {str(e)}",
                  traceback_msg=traceback.format_exc(),
                  extra_info={"index": i})
        print(f"Main loop failed on entry (index={i}): {e}")

# Final save
output_file = "./gemini-2.5-flash.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\nFinal results saved to {output_file}")
print(f"Error logs are being written to: {ERROR_LOG_FILE}")
