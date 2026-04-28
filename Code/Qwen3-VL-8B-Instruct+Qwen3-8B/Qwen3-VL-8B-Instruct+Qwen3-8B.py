import json
import os
import time
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import logging
from peft import PeftModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置基础路径
CONFIG = {
    "base_path": "./TabRefError/citekeypdf",
    "json_file": "./TabRefError/fold_0_val.json",
    "originalkey_base_path": "./TabRefError/originalkey",
    "output_file": "./Qwen3-VL-Text_fold0_train.json",
    "batch_output_prefix": "./qwen3-vl-text_train_fold0_batch_",
    "vl_model_path": "./Qwen3-VL-8B-Instruct",
    "text_model_path": "./Qwen3-8B",
    "lora_citekey_path": "your_lora_citekey_path",
    "lora_original_path": "your_lora_original_path",
    "lora_semantic_path": "your_lora_semantic_path",
    "cache_file": "./image_analysis_cache_fold0_train.json"
}


# 加载 Qwen-VL 模型
try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    vl_processor = Qwen3VLProcessor.from_pretrained(CONFIG["vl_model_path"])
    vl_base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CONFIG["vl_model_path"], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    print("Qwen3-VL 基础模型加载完成！")
except ImportError:
    try:
        vl_processor = Qwen2VLProcessor.from_pretrained(CONFIG["vl_model_path"])
        vl_base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            CONFIG["vl_model_path"], 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("Qwen2-VL 基础模型加载完成！")
    except Exception as e:
        logger.error(f"VL 模型加载失败：{e}")
        raise e

# 加载文本模型
text_tokenizer = None
text_base_model = None
try:
    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model_path"])
    text_base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["text_model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("文本基础模型加载完成！")
except Exception as e:
    print(f"文本模型加载失败：{e}")

# 加载 Qwen-VL 的 LoRA adapters
print(f"  - citekey LoRA: {CONFIG['lora_citekey_path']}")
print(f"  - original LoRA: {CONFIG['lora_original_path']}")

vl_peft_model = PeftModel.from_pretrained(vl_base_model, CONFIG["lora_citekey_path"], adapter_name="citekey")
vl_peft_model.load_adapter(CONFIG["lora_original_path"], adapter_name="original")

# 加载文本模型 LoRA
text_model = None
if text_base_model is not None and os.path.exists(CONFIG["lora_semantic_path"]):
    print(f"正在加载文本模型的语义匹配 LoRA: {CONFIG['lora_semantic_path']}")
    try:
        text_model = PeftModel.from_pretrained(text_base_model, CONFIG["lora_semantic_path"])
        text_model.eval()
        print("文本模型语义匹配 LoRA 加载完成！")
    except Exception as e:
        print(f"文本模型 LoRA 加载失败：{e}")
        text_model = text_base_model
else:
    text_model = text_base_model

vl_current_mode = "base"
print("所有模型加载完成！")

def ensure_cache_directory():
    cache_dir = os.path.dirname(CONFIG["cache_file"])
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"创建缓存目录：{cache_dir}")

# 加载缓存
image_cache = {}
def load_cache():
    global image_cache
    try:
        ensure_cache_directory()
        if os.path.exists(CONFIG["cache_file"]):
            with open(CONFIG["cache_file"], 'r', encoding='utf-8') as f:
                image_cache = json.load(f)
            logger.info(f"加载缓存文件，包含 {len(image_cache)} 个图片分析结果")
        else:
            logger.info("缓存文件不存在，将创建新的缓存")
            image_cache = {}
    except Exception as e:
        logger.warning(f"缓存加载失败：{e}")
        image_cache = {}


def save_cache():
    try:
        ensure_cache_directory()
        # 只保存 citekey 的缓存，过滤掉 original的缓存
        filtered_cache = {}
        for key, value in image_cache.items():
            if "_citekey" in key or ("_" in key and key.split("_")[-1] != "original"):
                filtered_cache[key] = value
        
        with open(CONFIG["cache_file"], 'w', encoding='utf-8') as f:
            json.dump(filtered_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"缓存已保存到 {CONFIG['cache_file']}，过滤掉了 original 模式缓存")
    except Exception as e:
        logger.warning(f"缓存保存失败：{e}")

load_cache()

# 支持的图片格式
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

def remove_control_chars(s):
    return re.sub(r'[\x00-\x1f]+', ' ', s)

def validate_image_for_model(image_path):
    try:
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在：{image_path}")
            return False, "文件不存在"
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            logger.error(f"图片文件为空：{image_path}")
            return False, "文件为空"
        
        max_size = 50 * 1024 * 1024
        if file_size > max_size:
            logger.warning(f"图片文件过大：{image_path}, 大小：{file_size / (1024*1024):.2f}MB")
        
        with Image.open(image_path) as img:
            img.verify()
            img = Image.open(image_path)
            if img.format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                logger.error(f"不支持的图片格式：{image_path}, 格式：{img.format}")
                return False, f"格式不支持：{img.format}"
            
            width, height = img.size
            if width < 10 or height < 10:
                logger.warning(f"图片尺寸过小：{image_path}, 尺寸：{width}x{height}")
            
            img.convert("RGB")
            
        logger.info(f"图片验证通过：{image_path}, 尺寸：{width}x{height}, 格式：{img.format}")
        return True, "验证通过"
        
    except Exception as e:
        logger.error(f"图片验证失败：{image_path}, 错误：{str(e)}")
        return False, f"验证失败：{str(e)}"

def call_vl_model(prompt, image_path=None, mode="base"):
    """
    调用 Qwen-VL 模型进行推理
    """
    global vl_current_mode
    
    try:
        if mode == "citekey":
            current_model = vl_peft_model
            if vl_current_mode != "citekey":
                try:
                    vl_peft_model.enable_adapters()
                    vl_peft_model.set_adapter("citekey")
                except:
                    pass
                vl_current_mode = "citekey"
                logger.debug("切换至 citekey adapter")
        elif mode == "original":
            current_model = vl_peft_model
            if vl_current_mode != "original":
                try:
                    vl_peft_model.enable_adapters()
                    vl_peft_model.set_adapter("original")
                except:
                    pass
                vl_current_mode = "original"
                logger.debug("切换至 original adapter")
        else:  # base mode
            current_model = vl_base_model
            if vl_current_mode != "base":
                vl_current_mode = "base"
                logger.debug("切换至基础模型")

        if image_path and os.path.exists(image_path):
            is_valid, msg = validate_image_for_model(image_path)
            if not is_valid:
                logger.error(f"图片验证失败，无法传入模型：{image_path}, 原因：{msg}")
                return "图片验证失败，无法处理"
            
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
            text = vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vl_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(current_model.device)
            
            generated_ids = current_model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logger.debug(f"图片处理成功：{os.path.basename(image_path)}")
            return output_text.strip()
                
        else:
            # 纯文本模式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
            text = vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vl_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(current_model.device)
            
            generated_ids = current_model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text.strip()
    
    except Exception as e:
        logger.error(f"VL 模型推理失败：{e}")
        import traceback
        logger.error(f"详细错误：{traceback.format_exc()}")
        return "模型推理失败"

def call_text_model(prompt):
    """
    调用文本模型进行推理
    """
    if text_model is None or text_tokenizer is None:
        logger.error("文本模型未加载，使用 VL 模型替代")
        return call_vl_model(prompt, image_path=None, mode="base")
    
    try:
        messages = [{"role": "user", "content": prompt}]
        text = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(text_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = text_model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                temperature=0.1,
                pad_token_id=text_tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = text_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logger.error(f"文本模型推理失败：{e}")
        return "文本模型推理失败"

def get_cache_key(image_path, mode):
    abs_path = os.path.abspath(image_path)
    if mode == "citekey":
        return f"{abs_path}_citekey"
    else:
        return f"{abs_path}_{mode}"

def is_valid_cached_result(result):
    if not result or not isinstance(result, dict):
        return False
    
    description = result.get("description", "")
    if not description or description in ["", "模型推理失败", "图片验证失败，无法处理"]:
        return False
    
    if "JSON 解析失败" in description:
        return False
        
    return True

def analyze_image(image_path, author=None, mode="citekey"):
    cache_key = get_cache_key(image_path, mode)

    if mode == "citekey" and cache_key in image_cache:
        cached_result = image_cache[cache_key]
        if is_valid_cached_result(cached_result):
            logger.info(f"使用缓存结果：{os.path.basename(image_path)} ({mode})")
            return cached_result
        else:
            logger.info(f"缓存结果无效，重新分析：{os.path.basename(image_path)} ({mode})")
            if cache_key in image_cache:
                del image_cache[cache_key]
    
    # 执行实际分析
    is_valid, msg = validate_image_for_model(image_path)
    if not is_valid:
        logger.error(f"图片无法传入模型：{image_path}, 原因：{msg}")
        result = {"description": f"图片验证失败：{msg}", "author_related_content": ""} if mode != "citekey" else {"description": f"图片验证失败：{msg}"}
        if mode == "citekey":
            image_cache[cache_key] = result
            save_cache()
        return result

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if mode == "citekey":
                prompt = f"""请详细分析以下图片内容，特别注意以下几点：
- 图片中包含的**所有数值**（如准确率、F1 值、BLEU 分数等），明确区分是**小数（如 0.85）还是百分比（如 85%）**，不可省略或混淆。
- 涉及的**模型名称**、**数据集名称**、**评估指标**。
- 如果是表格，请逐行提取关键数据，尤其是模型对比结果。
- 不需要提取与特定作者相关的内容（因为这是当前论文的图表）。
- 描述应尽可能完整、准确，不要遗漏任何关键数据。

请严格按照 JSON 格式返回：
{{
  "description": "详细描述，包含所有关键数据、模型、指标、数据集"
}}
"""
            else:  # mode == "originalkey"
                prompt = f"""请分析以上图片内容，并按以下要求分别返回：
【任务 1：整体描述】
描述图片的整体内容，包括文字、图表、表格中的关键信息，特别注意数值是小数还是百分比，具体数值不可省略。
如果图片中包含表格，请尽量提取表格中的关键数据。
关注模型名称、数据集、评估指标等关键信息。
【任务 2：作者相关内容】
提取图片中与作者引用标识"{author}"所在单元格行或列相关的内容（根据该标识位置决定，位于行首则为该行所有信息，位于列首则是该列所有信息，无需返回具体是第几列或第几行）（作者引用标识包括顺序编码制 [1][2]、著者 - 出版年制 (Wang et al., 2023) 等形式的引用标识）。
特别注意提取作者引用标识"{author}"所在单元格的模型名称、方法或数据等内容，并提取所有图片中与其相关的信息。
如果图片中包含表格，请特别关注表格中与作者引用标识"{author}"所有相关的数据。
【关键定义：引用标识】
在作者相关内容中，可能会出现{author}这样的引用标识，类似 `(Author, Year)` 或 `[1]`，括号内容视为**"引用标识"**。该标识**仅作为定位信号**：定位到表格图片中引用标识**与其对应**的文本内容，返回内容时删除该引用标识。

请严格按照以下 JSON 格式返回，确保两个字段都有内容： 
{{ "description": "图片的整体描述", 
"author_related_content": "删除'{author}'引用标识后，标识所在的行或列相关的内容，重点是模型名称、实验条件、实验数据及数据评估指标" }}
"""

            result = call_vl_model(prompt, image_path=image_path, mode=mode)
            json_str = extract_json(result)

            try:
                response_json = robust_json_loads(json_str, context=f"analyze_image_{mode}")
                desc = response_json.get("description", "").strip()
                if mode == "citekey":
                    if not desc:
                        desc = result.strip() if result else "模型未能生成有效描述"
                    final_result = {"description": desc}
                else:
                    author_cont = response_json.get("author_related_content", "").strip()
                    if not desc:
                        desc = result.strip() if result else "模型未能生成描述"
                    final_result = {"description": desc, "author_related_content": author_cont}

                if mode == "citekey":
                    image_cache[cache_key] = final_result
                    save_cache()
                return final_result

            except Exception as je:
                logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}): {je}")
                if attempt == max_retries - 1:
                    clean_result = (result or "").strip()
                    if mode == "citekey":
                        final_result = {"description": clean_result if clean_result else "模型无输出"}
                    else:
                        final_result = {"description": clean_result if clean_result else "模型无输出", "author_related_content": ""}
                    
                    if mode == "citekey":
                        image_cache[cache_key] = final_result
                        save_cache()
                    return final_result

        except Exception as e:
            logger.warning(f"analyze_image 第 {attempt + 1} 次失败：{e}")
            if attempt == max_retries - 1:
                fallback = "图片分析失败（多次重试）"
                if mode == "citekey":
                    final_result = {"description": fallback}
                else:
                    final_result = {"description": fallback, "author_related_content": ""}
                
                if mode == "citekey":
                    image_cache[cache_key] = final_result
                    save_cache()
                return final_result
            time.sleep(1)
    
    final_result = {"description": "未知错误", "author_related_content": "未知错误"}
    if mode == "citekey":
        image_cache[cache_key] = final_result
        save_cache()
    return final_result

def extract_json(text):
    if not text:
        return '{"match": false, "score": 0.0, "explanation": "空文本"}'
    text = remove_control_chars(text.strip())
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    return text

def robust_json_loads(text, context=""):
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
        import ast
        # 预处理布尔值和 null
        py_text = cleaned_text.replace("true", "True").replace("false", "False").replace("null", "None")
        obj = ast.literal_eval(py_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    logger.warning(f"所有 JSON 解析方法失败，返回默认对象。上下文：{context}")
    return {
        "match": False,
        "score": 0.0,
        "explanation": "JSON 解析失败"
    }

def clean_author_from_text(text, author):

    if not text or not isinstance(text, str):
        return text

    cleaned = text.strip()

    cleaned = re.sub(r'\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]', '', cleaned)


    core_author = ""
    if author:
        name_part = re.sub(r'[()\[\],;]', ' ', str(author))
        name_part = re.sub(r'\b\d{4}\b', '', name_part)
        name_part = re.sub(r'\b(?:et|al|and|&)\b\.?', '', name_part, flags=re.IGNORECASE)
        name_part = re.sub(r'\s+', ' ', name_part).strip()
        tokens = [t.strip('.,;:') for t in name_part.split() if t.strip('.,;:')]
        if tokens:
            core_author = tokens[0]

    patterns = []
    if core_author:
        escaped = re.escape(core_author)
        patterns.append(r'\(' + escaped + r'\s*(?:[,，]?\s*(?:et\s+al\.?)?)?\s*(?:[,，]?\s*\d{4})?\s*\)')
        patterns.append(r'\b' + escaped + r'\s*(?:et\s+al\.?)?\s*(?:[,，]?\s*\d{4})?\b')
        patterns.append(r'\b' + escaped + r'\s*(?:et\s+al\.?)?\s*(?:\(|\[|\{)')

    patterns.append(r'\b(?:et\s+al\.?|etal\.?)\s*(?:[,，;；]?\s*\d{4})?\b')
    patterns.append(r'\(\s*(?:et\s+al\.?|etal\.?)\s*(?:[,，]?\s*\d{4})?\s*\)')

    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([,，;；.!！？:：\)\]])', r'\1', cleaned)
    cleaned = re.sub(r'([,，;；.!！？:：\(\[])\s*', r'\1 ', cleaned)
    cleaned = re.sub(r'\(\s*\)', '', cleaned)
    cleaned = re.sub(r'\[\s*\]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def semantic_comparison(claim, originalkey_item, author, caption_annotations="", paper_id=None):
    """
    使用文本模型
    """
    claim = claim.strip() if claim else ""
    if not isinstance(originalkey_item, dict):
        originalkey_item = {}

    author_content = originalkey_item.get("author_related_content", "").strip()

    if not author_content:
        return {
            "match": False,
            "score": 0.0,
            "explanation": f"原始文献图片中未找到与作者 '{author}' 相关的内容，无法验证引用。"
        }

    cleaned_author_content = clean_author_from_text(author_content, author)
    
    full_context = cleaned_author_content
    if caption_annotations:
        full_context = f"{cleaned_author_content}（表格注释信息：{caption_annotations}）"

    max_retries = 3

    safe_ground_truth = claim.replace("{", "{{").replace("}", "}}")
    safe_citation_sentence = full_context.replace("{", "{{").replace("}", "}}")

    for attempt in range(max_retries):
        try:
            prompt = f"""你是一名严谨的学术事实核查员（Academic Fact-Checker）。

【任务目标】
验证【待核查的引用句】中的事实陈述，是否准确反映了【原文事实】中的数据。

【输入数据】
<原文事实 (Ground Truth)>
{safe_ground_truth}
</原文事实 (Ground Truth)>

<待核查的引用句 (Citation Sentence)>
{safe_citation_sentence}  
</待核查的引用句 (Citation Sentence)>

【表格注释信息】
{caption_annotations if caption_annotations else "无表格注释信息"}

【关键定义：引用标识】
在"待核查的引用句"中，可能会出现{author}这样的引用标识，类似 `(Author, Year)` 或 `[1]` 括号内容视为**"引用标识"**。
1. **不作为核查对象**：禁止在"原文事实"中寻找完全一致的括号字符串（原文事实不包含引用标识）。
2. **作为定位信号**：表格图片中引用标识**与其对应**的文本内容应当源自"原文事实"，无需核查所在行数是否相同。
3. **核查重点**：完全删除引用标识本身，重点核查**标识所管辖的文本内容**（数值、结论、模型表现）是否与"原文事实"一致。

【核心核查原则】
1. **事实支持原则**：
   - "待核查的引用句"中被引用标记覆盖的陈述，必须在"原文事实"中找到确凿证据。
   - 允许引用句只提取原文的部分信息（如原文有 R-1/R-2/R-L，引用句只提 R-L），这属于**合理概括**，不扣分。
2. **严禁张冠李戴**：检查数值是否归属于正确的评估指标、模型和数据集。
3. **数值与逻辑一致性**：数值允许合理的四舍五入误差。
4. **表格注释参考**：结合注释信息理解表格内容。

【评分规则】
- **Score = 1.0 (Match=True)**: 引用句中的所有关键事实均准确无误，且归属正确。
- **0.5 <= Score < 1.0 (Match=True)**: 事实基本准确，存在非关键信息的模糊表述。
- **Score < 0.5 (Match=False)**: 存在数值错误、模型归属错误或捏造结论。

【输出格式】
请仅返回严格的 JSON 格式对象，不要包含 Markdown 标记。
{{
    "match": true/false, 
    "score": 0.0~1.0, 
    "explanation": "简短说明核查结果"
}}
"""

            result = call_text_model(prompt)
            json_str = extract_json(result)
            
            if not json_str:
                raise ValueError("模型未返回有效的 JSON 字符串")

            try:
                response_json = robust_json_loads(json_str, context="semantic_comparison")
                return {
                    "match": bool(response_json.get("match", False)),
                    "score": float(response_json.get("score", 0.0)),
                    "explanation": str(response_json.get("explanation", "无详细解释"))
                }
                
            except Exception as je:
                logger.warning(f"semantic_comparison JSON 解析失败 (尝试 {attempt + 1}): {je}. Raw: {json_str[:100]}...")
                if attempt == max_retries - 1:
                    return {
                        "match": False,
                        "score": 0.0,
                        "explanation": "语义比较成功但 JSON 解析失败，返回默认结果"
                    }
                time.sleep(1)

        except Exception as e:
            logger.warning(f"semantic_comparison 第 {attempt + 1} 次执行失败：{e}")
            if attempt == max_retries - 1:
                return {
                    "match": False,
                    "score": 0.0,
                    "explanation": f"语义匹配失败（多次重试）: {str(e)}"
                }
            time.sleep(1)
            
    return {"match": False, "score": 0.0, "explanation": "未知错误"}

def find_content_list_file(citekey):
    citekey_folder_path = os.path.join(CONFIG["base_path"], citekey)
    if not os.path.exists(citekey_folder_path):
        logger.error(f"citekey 文件夹未找到：{citekey_folder_path}")
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
                logger.info(f"找到 content_list.json 文件：{content_list_path}")
                return content_list_path

    logger.error(f"content_list.json 文件未找到：citekey={citekey}")
    return None

def is_valid_image(image_path):
    try:
        if not os.path.exists(image_path):
            return False
        SUPPORTED_FORMATS = ["JPEG", "JPG", "PNG", "WEBP"]
        with Image.open(image_path) as img:
            img.verify()
            if img.format not in SUPPORTED_FORMATS:
                logger.error(f"图片格式不支持：{image_path}, 格式：{img.format}")
                return False
        return True
    except (OSError, PermissionError, ValueError) as e:
        logger.error(f"图片无效或损坏或无权限：{image_path}, 错误：{e}")
        return False

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
            logger.warning(f"TAB 图片未找到：{tab_id}")
            continue

        tab_id_no_space = tab_id.replace(" ", "")
        caption_images = []

        base_caption_name = f"CAPTION {tab_id_no_space}"
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            path = os.path.join(originalkey_folder_path, base_caption_name + ext)
            if os.path.exists(path) and is_valid_image(path):
                caption_images.append(path)

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
            logger.info(f"找到 {len(caption_images)} 个 CAPTION 文件 for {tab_id}")
        else:
            logger.info(f" 未找到 CAPTION 文件 for {tab_id}")

        groups.append({
            "tab_id": tab_id,
            "tab_image": tab_image_path,
            "caption_images": caption_images
        })
    return groups

def process_claim_stream(entry, originalkey_folder_path, charts, author):
    try:
        paper_id = entry.get("paper_id", "")
        citekey = entry.get("citekey", "")
        label = entry.get("label", "")

        content_list_path = find_content_list_file(citekey)
        if not content_list_path:
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

        citekey_items = [item for item in content_list if item["type"] in ["table", "image"]]
        citekey_images = [os.path.join(content_list_dir, item["img_path"]) for item in citekey_items if "img_path" in item]

        claims_with_info = []
        for image_path in citekey_images:
            if not is_valid_image(image_path):
                logger.warning(f"跳过无效图片：{image_path}")
                continue
            analysis = analyze_image(image_path, mode="citekey")
            claim = analysis.get("description", "").strip()
            if not claim:
                claim = "模型未能生成有效描述"
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

            # 分析 TAB 图片
            tab_analysis = analyze_image(tab_image, author=author, mode="original")
            tab_desc = tab_analysis.get("description", "").strip()
            tab_author = tab_analysis.get("author_related_content", "").strip()

            caption_annotations_list = []
            for cap_img in caption_images:
                caption_prompt = f"""你正在查看一个表格的说明文字（caption）。请从中提取**对表格内容的解释性信息**。例如：
- 数据集缩写的全称
- 指标缩写的含义
- 符号说明
- 实验设置说明

**不要提取作者结论、方法比较或主观评价**。

请将提取的信息整理为一句连贯的补充说明。

返回 JSON 格式：
{{
  "table_annotations": "补充说明（单句）"
}}
"""
                cap_result = call_vl_model(caption_prompt, image_path=cap_img, mode="original")
                try:
                    cap_json = extract_json(cap_result)
                    cap_obj = robust_json_loads(cap_json, context="caption_analysis")
                    ann_str = cap_obj.get("table_annotations", "").strip()
                    if ann_str and ann_str.lower() not in ["", "null", "none", "无"]:
                        cleaned_ann = clean_author_from_text(ann_str, author).strip()
                        if cleaned_ann:
                            caption_annotations_list.append(cleaned_ann)
                            logger.info(f"成功提取并清洗 CAPTION 注释 ({os.path.basename(cap_img)})")
                except Exception as e:
                    logger.warning(f"CAPTION 解析失败：{e}")

            caption_annotations = "；".join(caption_annotations_list) if caption_annotations_list else ""

            final_description = tab_desc if tab_desc else "表格内容未能提取"
            if caption_annotations:
                final_description = f"{final_description}（注：{caption_annotations}）"

            final_author_content = clean_author_from_text(tab_author, author)

            if final_description or final_author_content:
                originalkey_analysis.append({
                    "tab_id": tab_id,
                    "analysis": {
                        "description": final_description,
                        "author_related_content": final_author_content,
                        "caption_annotations": caption_annotations
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
                    caption_annotations=originalkey_item["analysis"].get("caption_annotations", ""),
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
            "explanation": "多模态 - 文本混合语义匹配完成"
        }

    except Exception as e:
        logger.error(f"处理失败：{e}")
        import traceback
        logger.error(f"详细错误：{traceback.format_exc()}")
        paper_id = entry.get("paper_id", "")
        label = entry.get("label", "")
        return {
            "id": paper_id,
            "label": label,
            "claim": [],
            "author": entry.get("author", ""),
            "matching_results": [],
            "result": "false",
            "explanation": f"处理错误：{str(e)}"
        }

def get_all_existing_batch_files():
    batch_files = []
    batch_counter = 1
    while True:
        batch_file = f"{CONFIG['batch_output_prefix']}{batch_counter}.json"
        if os.path.exists(batch_file):
            batch_files.append(batch_file)
        else:
            break
        batch_counter += 1
    return batch_files

def merge_and_deduplicate_batches():
    
    batch_files = get_all_existing_batch_files()
    if not batch_files:
        logger.info("没有找到现有的批次文件")
        return []
    
    all_final_results = []
    processed_ids = set()

    for batch_file in batch_files:
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_results = json.load(f)
        
        unique_results = []
        for result in batch_results:
            result_id = result.get("id", "")
            if result_id not in processed_ids:
                unique_results.append(result)
                processed_ids.add(result_id)
        
        all_final_results.extend(unique_results)
        logger.info(f"处理批次文件 {batch_file}，添加 {len(unique_results)} 条唯一记录，总计 {len(all_final_results)} 条")
    
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(all_final_results, f, indent=4, ensure_ascii=False)
    logger.info(f"\n合并后的去重结果已保存到 {CONFIG['output_file']}，总共 {len(all_final_results)} 条")
    
    return all_final_results

# 主流程
if __name__ == "__main__":

    all_final_results = merge_and_deduplicate_batches()

    with open(CONFIG["json_file"], "r", encoding="utf-8") as f:
        all_data = json.load(f)

    already_processed_ids = set(result["id"] for result in all_final_results)
    logger.info(f"已处理 {len(already_processed_ids)} 条数据")

    unprocessed_data = [entry for entry in all_data if entry.get("paper_id", "") not in already_processed_ids]
    logger.info(f"发现 {len(unprocessed_data)} 条未处理的数据")

    if unprocessed_data:
        print(f"开始处理 {len(unprocessed_data)} 条未处理的数据...")
        
        results = []
        batch_size = 10
        total_processed = 0

        existing_batch_files = get_all_existing_batch_files()
        next_batch_num = len(existing_batch_files) + 1 if existing_batch_files else 1

        for entry in unprocessed_data:
            paper_id = entry.get("paper_id", "")
            originalkey = entry.get("originalkey", "")
            charts = entry.get("charts", [])
            author = entry.get("author", "")

            originalkey_folder_path = os.path.join(CONFIG["originalkey_base_path"], originalkey)
            if not os.path.exists(originalkey_folder_path):
                logger.error(f"originalkey 文件夹未找到：{originalkey_folder_path}")
                result = {
                    "id": paper_id,
                    "label": entry.get("label", ""),
                    "claim": [],
                    "author": author,
                    "matching_results": [],
                    "result": "false",
                    "explanation": "originalkey 文件夹未找到"
                }
                results.append(result)
                total_processed += 1
                continue

            result = process_claim_stream(entry, originalkey_folder_path, charts, author)
            results.append(result)

            print(f"\n当前处理结果 (第 {total_processed + 1} 条): ID={paper_id}, Match={result['result']}")
            print(json.dumps(result, ensure_ascii=False, indent=2)) # 调试时可打开

            total_processed += 1

            if len(results) >= batch_size:
                output_file_batch = f"{CONFIG['batch_output_prefix']}{next_batch_num}.json"
                with open(output_file_batch, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                logger.info(f"\n批次 {next_batch_num} 已保存，包含 {len(results)} 条记录")
                
                next_batch_num += 1
                results = []
                save_cache()

        if results:
            output_file_batch = f"{CONFIG['batch_output_prefix']}{next_batch_num}.json"
            with open(output_file_batch, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"\n最后一批次 {next_batch_num} 已保存，包含 {len(results)} 条记录")

        all_final_results = merge_and_deduplicate_batches()
    else:
        print("所有数据均已处理完成")

    save_cache()
    logger.info(f"图片分析缓存已保存到 {CONFIG['cache_file']}")
    print("程序执行完成")
