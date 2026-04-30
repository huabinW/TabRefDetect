import os
import time
import re
import ast
from openai import OpenAI
import oss2
from PIL import Image
import traceback
import json


# 基础路径
base_path = "./TabRefError/citekeypdf"
json_file = "./TabRefError/tabIndex.json"
originalkey_base_path = "./TabRefError/originalkey"
ERROR_LOG_FILE = "./processing_errors.jsonl"


client = OpenAI(
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 阿里云 OSS 配置
access_key_id = "your_access_key_id"
access_key_secret = "your_access_key_secret"
bucket_name = "your_bucket_name"
endpoint = "your_endpoint"

auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def to_safe_string(value, field_name="value"):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    print(f" 警告: {field_name} 不是字符串（类型: {type(value)}），已转为字符串")
    try:
        return str(value)
    except Exception as e:
        print(f" 转换 {field_name} 为字符串失败: {e}")
        return ""


def remove_control_chars(s):
    return re.sub(r'[\x00-\x1f]+', ' ', s)


def extract_json(text):
    text = remove_control_chars(text.strip())
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()

    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    return text


SYSTEM_PROMPTS = {
    "image_analyzer": (
        "你是一个文献表格解析器，帮我完成对该表格的解析任务"
    ),
    "semantic_judge": (
        "你是一个严谨的学术引用审查员。负责对比施引文献声明与被引文献原始数据，基于事实进行客观匹配评分。"
        "严格遵循数值一致性、模型/数据集对应原则，忽略无关内容。只输出要求的 JSON，不附加任何解释。"
    ),
    "caption_extractor": (
        "你是一名严谨的学术审查员，请判断以下两部分内容是否语义匹配："
    )
}


def robust_json_loads(text, paper_id=None, context=""):
    cleaned_text = remove_control_chars(text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    try:
        fixed_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned_text)
        if "'" in fixed_text and '"' not in fixed_text:
            fixed_text = fixed_text.replace("'", '"')
        return json.loads(fixed_text)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(cleaned_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:

        safe_text_for_prompt = remove_control_chars(text)
        fix_prompt = f"""你是一个JSON格式修复器。请将以下可能格式错误的文本转换为严格合法的JSON对象，只返回JSON，不要任何解释：

原始文本：
{safe_text_for_prompt}

合法JSON："""
        completion = client.chat.completions.create(
            model="qwen3vl/qwen-vl-max",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.0
        )
        fixed_json_str = extract_json(completion.choices[0].message.content.strip())
        fixed_json_str = remove_control_chars(fixed_json_str)
        return json.loads(fixed_json_str)
    except Exception as e:
        log_error(paper_id, "JSONRepairFailed", f"LLM 修复 JSON 失败: {str(e)}",
                 traceback_msg=traceback.format_exc(),
                 extra_info={"context": context, "raw_length": len(text)})

    raise ValueError("无法解析为有效 JSON")


def is_valid_image(image_path):
    try:
        if not os.path.exists(image_path):
            return False
        SUPPORTED_FORMATS = ["JPEG", "JPG", "PNG", "WEBP"]
        with Image.open(image_path) as img:
            img.verify()
            if img.format not in SUPPORTED_FORMATS:
                print(f"图片格式不支持: {image_path}, 格式: {img.format}")
                return False
        return True
    except (OSError, PermissionError, ValueError) as e:
        print(f"图片无效或损坏或无权限: {image_path}, 错误: {e}")
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
        print(f" 写入错误日志失败: {e}")


def upload_image_to_oss(image_path, paper_id=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not os.path.exists(image_path):
                print(f"文件不存在: {image_path}")
                log_error(paper_id, "FileNotFound", "图片文件不存在",
                         extra_info={"image_path": image_path})
                return None

            if not is_valid_image(image_path):
                print(f"图片无效，跳过上传: {image_path}")
                log_error(paper_id, "InvalidImage", "图片格式不支持或已损坏",
                         extra_info={"image_path": image_path})
                return None

            with open(image_path, "rb") as file:
                file_name = os.path.basename(image_path)
                bucket.put_object(file_name, file)
                http_url = bucket.sign_url('GET', file_name, 3600)
                https_url = http_url.replace("http://", "https://")
                print(f"图片上传成功: {file_name}, HTTPS URL: {https_url}")
                return https_url

        except PermissionError as pe:
            print(f"权限错误 (尝试 {attempt + 1}): {image_path}, 错误: {pe}")
            if attempt == max_retries - 1:
                log_error(paper_id, "PermissionError", str(pe),
                         extra_info={"image_path": image_path})

        except Exception as e:
            print(f"图片上传失败，第 {attempt + 1} 次尝试: {image_path}, 错误: {e}")
            if attempt == max_retries - 1:
                log_error(paper_id, "UploadFailed", f"OSS 上传失败: {str(e)}",
                         traceback_msg=traceback.format_exc(),
                         extra_info={"image_path": image_path, "oss_bucket": bucket_name})
            time.sleep(1)
    return None


def analyze_image(image_url, author=None, mode="citekey", paper_id=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if mode == "citekey":
                prompt = f"""请详细分析以下图片内容，特别注意以下几点：
1.图片中包含的所有数值（如准确率、F1值、BLEU分数等），明确区分是小数（如0.85）还是百分比（如85%），不可省略。
2.涉及的模型名称（如BERT,ResNet,Ours,Proposed Method等）、数据集名称（如ImageNet,COCO,SQuAD）、评估指标（如Accuracy,ROUGE-L,mAP）。
3.如果是表格，请逐行提取关键数据，尤其是模型对比结果。
4.不需要提取与特定作者相关的内容。
5.描述应尽可能完整、准确，不要遗漏任何关键数据。
请直接返回JSON格式：
{{
  "description": "详细描述，包含所有关键数据、模型、指标、数据集等"
}}
"""
            else:
                prompt = f"""请分析以下图片内容，并提取相关信息：
1.描述图片的整体内容，包括文字、图表、表格中的关键信息，特别注意数值是小数还是百分比，具体数值不可省略。
2.提取图片中与作者{author}相关的文字、图表或数据（如作者提出的方法、其模型名称、对应结果等）。如果没有相关内容，则返回空字符串。
3.如果图片中包含表格，请尽量提取表格中的关键数据，并标注与作者"{author}"的关联性。
4.关注模型名称、数据集、评估指标等关键信息。
示例（仅作格式参考）：
"AdaptCap(Wang et al.,2023)在COCO-Caption上：BLEU-4=38.5,METEOR=28.9,CIDEr=124.7"
返回以下字段：
{{
  "description":"图片的整体描述",
  "author_related_content":"与作者{author}相关的内容（如果没有则为空字符串）"
}}
"""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["image_analyzer"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            completion = client.chat.completions.create(
                model="qwen3vl/qwen-vl-max",
                messages=messages,
                temperature=0.0
            )

            output_text = completion.choices[0].message.content.strip()
            json_str = extract_json(output_text)

            try:
                response_json = robust_json_loads(json_str, paper_id=paper_id, context=f"analyze_image_{mode}")
                print(f"分析成功 ({mode}, 尝试 {attempt + 1})")
                return response_json
            except Exception as je:
                print(f"JSON 解析失败 (尝试 {attempt + 1}): {je}")
                if attempt == max_retries - 1:
                    log_error(paper_id, "JSONParseFailed", f"解析 LLM 输出失败: {str(je)}",
                             traceback_msg=traceback.format_exc(),
                             extra_info={"mode": mode, "raw_output": output_text[:500]})
                    if mode == "citekey":
                        return {"description": "模型未能生成有效描述"}
                    else:
                        return {"description": "模型未能生成描述", "author_related_content": ""}

        except Exception as e:
            print(f"analyze_image 第 {attempt + 1} 次失败: {e}")
            if attempt == max_retries - 1:
                log_error(paper_id, "LLMCallFailed", f"调用 Qwen3.5 失败: {str(e)}",
                         traceback_msg=traceback.format_exc(),
                         extra_info={"mode": mode, "author": author})
            time.sleep(1)
    return {"description": "未知错误", "author_related_content": "未知错误"}


def semantic_comparison(claim, originalkey_item, author, paper_id=None):
    claim = to_safe_string(claim, "claim").strip()
    if not isinstance(originalkey_item, dict):
        originalkey_item = {}

    author_content = to_safe_string(originalkey_item.get("author_related_content", ""), "author_related_content").strip()

    if not author_content:
        return {
            "match": False,
            "score": 0.0,
            "explanation": f"原始文献图片中未找到与作者 '{author}' 相关的内容，无法验证引用。"
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = f"""
【任务说明】
内容A（被引文献）：{claim}
内容B（施引文献中与作者{author}相关的内容）：{author_content}

【评分规则】
1.核心匹配项：数值（小数或%）、模型名称、数据集、评估指标。
2.允许合理缩写（如"BERT"vs"Bert-base","ImageNet"vs"IN","Accuracy"vs"Acc"）。
3.如果内容A完整覆盖内容B中的{author}相关数据（数值一致、模型/数据集基本一致）即匹配成功，score=1.0。
4.如果内容A部分缺失{author}相关数据（如缺少某个指标或数值）则score在0.1~0.9之间，按缺失程度扣分。
5.如果内容A完全未提及{author}相关内容则score=0.0。
6.不要关注内容B中与{author}无关的部分。

【输出要求】
请返回严格JSON格式，包含：
"match": true/false
"score": 0.0~1.0的浮点数
"explanation":简明说明匹配/不匹配原因，指出缺失或一致的关键点
"""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["semantic_judge"]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]

            completion = client.chat.completions.create(
                model="qwen3vl/qwen-vl-max",
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
                print(f" semantic_comparison JSON 解析失败 (尝试 {attempt + 1}): {je}")
                if attempt == max_retries - 1:
                    log_error(paper_id, "SemanticComparisonFailed", f"解析语义比对结果失败: {str(je)}",
                             traceback_msg=traceback.format_exc())

        except Exception as e:
            print(f"semantic_comparison 第 {attempt + 1} 次失败: {e}")
            if attempt == max_retries - 1:
                log_error(paper_id, "SemanticComparisonFailed", f"语义比对调用失败: {str(e)}",
                         traceback_msg=traceback.format_exc())
            time.sleep(1)
    return {"match": False, "score": 0.0, "explanation": "未知错误"}


def find_content_list_file(citekey):
    citekey_folder_path = os.path.join(base_path, citekey)
    if not os.path.exists(citekey_folder_path):
        print(f"citekey 文件夹未找到: {citekey_folder_path}")
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
                print(f"找到 content_list.json 文件: {content_list_path}")
                return content_list_path

    print(f"content_list.json 文件未找到: citekey={citekey}")
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
            print(f"TAB 图片未找到: {tab_id}")
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
            print(f"找到 {len(caption_images)} 个 CAPTION 文件 for {tab_id}")
        else:
            print(f"未找到 CAPTION 文件 for {tab_id}")

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
                      "citekey 文件夹或 content_list.json 未找到",
                      extra_info={"citekey": entry.get("citekey", "")})
            return {
                "id": paper_id,
                "label": label,
                "claim": [],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "content_list.json 文件未找到"
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
                "explanation": "content_list 为空"
            }

        citekey_items = [item for item in content_list if item["type"] in ["table"]]
        citekey_images = [os.path.join(content_list_dir, item["img_path"]) for item in citekey_items if "img_path" in item]

        claims_with_info = []
        for image_path in citekey_images:
            image_url = upload_image_to_oss(image_path, paper_id=paper_id)
            if not image_url:
                continue
            analysis = analyze_image(image_url, mode="citekey", paper_id=paper_id)
            claim = to_safe_string(analysis.get("description", ""), "citekey_description").strip()
            if not claim:
                claim = "模型未能生成有效描述，但图片已参与分析"
            claims_with_info.append({
                "claim": claim,
                "citekey_image_name": os.path.basename(image_path)
            })

        if not claims_with_info:
            claims_with_info = [{"claim": "未找到有效 citekey 图片", "citekey_image_name": "N/A"}]

        tab_groups = group_originalkey_images(originalkey_folder_path, charts, author)
        originalkey_analysis = []
        for group in tab_groups:
            tab_id = group["tab_id"]
            tab_image = group["tab_image"]
            caption_images = group["caption_images"]

            tab_desc = ""
            tab_author = ""
            tab_url = upload_image_to_oss(tab_image, paper_id=paper_id)
            if tab_url:
                tab_analysis = analyze_image(tab_url, author=author, mode="originalkey", paper_id=paper_id)
                tab_desc = to_safe_string(tab_analysis.get("description", ""), "tab_description").strip()
                tab_author = to_safe_string(tab_analysis.get("author_related_content", ""), "tab_author").strip()

            caption_annotations = ""
            for cap_img in caption_images:
                cap_url = upload_image_to_oss(cap_img, paper_id=paper_id)
                if not cap_url:
                    continue
                caption_prompt = f"""你正在查看一个表格的说明文字（caption）。请从中提取对表格内容的解释性信息，例如：
1.数据集缩写的全称（如 "ETD:End-to-End Turn-taking Dataset"）
2.指标缩写的含义（如 "SU:Speech Utterance","IoU: Intersection over Union"）
3.符号说明（如"↑表示性能提升","†表示使用额外数据"）
4.实验设置说明（如"所有模型在相同硬件上测试"）
不要提取作者结论、方法比较或主观评价（如"our method is better"）。
请将提取的信息整理为一句连贯的补充说明。
返回格式：
{{
  "table_annotations":"补充说明（单句）"
}}
"""
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPTS["caption_extractor"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": cap_url}},
                            {"type": "text", "text": caption_prompt}
                        ]
                    }
                ]
                try:
                    completion = client.chat.completions.create(
                        model="qwen3vl/qwen-vl-max",
                        messages=messages,
                        temperature=0.0
                    )
                    output_text = completion.choices[0].message.content.strip()
                    try:
                        if '"table_annotations"' in output_text:
                            start = output_text.find('"table_annotations"')
                            start_val = output_text.find(':', start)
                            end_val = output_text.find('}', start_val)
                            ann_str = output_text[start_val+1:end_val].strip().strip('"').strip()
                            if ann_str and ann_str.lower() not in ["", "null", "none", "无"]:
                                caption_annotations = ann_str
                                break
                    except Exception as e:
                        log_error(paper_id, "CaptionParseFailed", f"解析 CAPTION 注释失败: {str(e)}",
                                 traceback_msg=traceback.format_exc())
                except Exception as e:
                    log_error(paper_id, "CaptionLLMFailed", f"调用 LLM 分析 CAPTION 失败: {str(e)}",
                             traceback_msg=traceback.format_exc())
                if caption_annotations:
                    break

            final_description = tab_desc if tab_desc else "表格内容未能提取"
            if caption_annotations:
                final_description = f"{final_description} （注：{caption_annotations}）"

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
                "explanation": "未找到有效的 originalkey 图片进行分析"
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
            "explanation": "匹配完成，请查看 matching_results 获取详细信息"
        }

    except Exception as e:
        log_error(paper_id, "ProcessFailed", f"处理条目失败: {str(e)}",
                 traceback_msg=traceback.format_exc(),
                 extra_info={"entry_keys": list(entry.keys())})
        print(f"处理失败 (paper_id={paper_id}): {e}")
        return {
            "id": paper_id,
            "label": entry.get("label", ""),
            "claim": [],
            "author": entry.get("author", ""),
            "matching_results": [],
            "result": "false",
            "explanation": f"处理错误: {str(e)}"
        }



results = []
batch_size = 10

# 加载已有中间结果
last_counter = 0
batch_counter = 1
while True:
    batch_file = f"./qwen3vl/qwen-vl-max_batch_{batch_counter}.json"
    if os.path.exists(batch_file):
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
                results.extend(batch_results)
            last_counter = batch_counter * batch_size
            print(f"加载已有批次 {batch_counter}，已处理 {last_counter} 条")
            batch_counter += 1
        except Exception as e:
            print(f"加载批次文件失败: {batch_file}, 错误: {e}")
            break
    else:
        break

print(f"当前已处理条数: {last_counter}")


with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)


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
            print(f"originalkey文件夹未找到: {originalkey_folder_path}")
            result = {
                "id": paper_id,
                "label": entry.get("label", ""),
                "claim": [],
                "author": author,
                "matching_results": [],
                "result": "false",
                "explanation": "originalkey文件夹未找到"
            }
            log_error(paper_id, "OriginalKeyNotFound", "originalkey文件夹未找到",
                     extra_info={"originalkey": originalkey})
            results.append(result)
            counter += 1
            continue

        result = process_claim_stream(entry, originalkey_folder_path, charts, author)
        results.append(result)

        print(f"\n当前处理结果 (第 {i+1} 条):")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        counter += 1

        if counter % batch_size == 0:
            output_file_batch = f"./qwen3vl/qwen-vl-max_batch_{counter // batch_size}.json"
            with open(output_file_batch, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"\n中间结果已保存到 {output_file_batch}")

    except Exception as e:
        paper_id = entry.get("paper_id", "unknown")
        log_error(paper_id, "MainLoopException", f"主循环处理失败: {str(e)}",
                 traceback_msg=traceback.format_exc(),
                 extra_info={"index": i})
        print(f"主循环处理条目失败 (index={i}): {e}")


output_file = "./qwen3vl/qwen-vl-max.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\n最终结果已保存到 {output_file}")
print(f"错误日志已实时保存到: {ERROR_LOG_FILE}")
