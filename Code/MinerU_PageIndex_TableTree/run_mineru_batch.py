import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE / "manifest.json"


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mineru_doc_dir(output_root, stem):
    return output_root / stem / "hybrid_auto"


def content_list_path(output_root, stem):
    return mineru_doc_dir(output_root, stem) / f"{stem}_content_list.json"


def find_content_list(output_root, stem):
    for mode in ("hybrid_auto", "auto", "ocr", "txt"):
        candidate = output_root / stem / mode / f"{stem}_content_list.json"
        if candidate.exists():
            return candidate
    matches = list((output_root / stem).glob(f"**/{stem}_content_list.json"))
    if matches:
        return matches[0]
    return content_list_path(output_root, stem)


def stage_pdf(doc, manifest):
    staged_dir = Path(manifest["output_root"]) / "staged_pdfs"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_pdf = staged_dir / f"{doc['slug']}.pdf"
    source_pdf = Path(doc["pdf_path"])
    if not staged_pdf.exists() or staged_pdf.stat().st_size != source_pdf.stat().st_size:
        shutil.copy2(source_pdf, staged_pdf)
    return staged_pdf


def merge_chunk_outputs(chunk_infos, merged_content_list):
    merged = []
    image_dir = merged_content_list.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunk_infos:
        chunk_content = Path(chunk["content_list"])
        if not chunk_content.exists():
            return False
        items = json.loads(chunk_content.read_text(encoding="utf-8"))
        start_page = chunk["start"]
        chunk_stem = chunk["stem"]
        for item in items:
            if item.get("page_idx") is not None:
                item["page_idx"] = int(item["page_idx"]) + start_page
            if item.get("img_path"):
                chunk_mode = Path(chunk["content_list"]).parent.name
                source_image = chunk_content.parent / item["img_path"]
                image_name = f"{chunk_stem}_{Path(item['img_path']).name}"
                target_image = image_dir / image_name
                if source_image.exists() and not target_image.exists():
                    shutil.copy2(source_image, target_image)
                item["img_path"] = f"images/{image_name}"
            item["mineru_chunk"] = {
                "stem": chunk_stem,
                "start_page": chunk["start"],
                "end_page": chunk["end"],
            }
        merged.extend(items)
    merged_content_list.parent.mkdir(parents=True, exist_ok=True)
    merged_content_list.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def build_attempts(doc, manifest, output_root, pdf_path):
    attempts = doc.get("mineru_attempts")
    if attempts:
        return attempts
    return [
        {
            "name": "hybrid_auto",
            "args": ["-p", str(pdf_path), "-o", str(output_root)],
        },
        {
            "name": "hybrid_no_image_analysis",
            "args": ["-p", str(pdf_path), "-o", str(output_root), "--image-analysis", "false"],
        },
        {
            "name": "pipeline_auto",
            "args": ["-p", str(pdf_path), "-o", str(output_root), "-b", "pipeline", "-m", "auto"],
        },
    ]


def api_args(manifest):
    api_url = manifest.get("api_url")
    return ["--api-url", api_url] if api_url else []


def run_chunked(doc, manifest, staged_pdf, env, log):
    page_count = int(doc.get("page_count") or 0)
    if page_count <= 0:
        return {"name": "chunked_pipeline", "returncode": 1, "content_list_exists": False}

    chunk_size = int(doc.get("mineru_chunk_size") or 3)
    chunks_root = Path(manifest["output_root"]) / "mineru_output" / "chunks"
    merged_root = Path(manifest["output_root"]) / "mineru_output"
    merged_content = content_list_path(merged_root, doc["slug"])
    chunk_infos = []
    returncodes = []

    for start in range(0, page_count, chunk_size):
        end = min(page_count - 1, start + chunk_size - 1)
        chunk_stem = f"{doc['slug']}_p{start + 1:03d}_{end + 1:03d}"
        chunk_pdf = staged_pdf.with_name(f"{chunk_stem}.pdf")
        if not chunk_pdf.exists() or chunk_pdf.stat().st_size != staged_pdf.stat().st_size:
            shutil.copy2(staged_pdf, chunk_pdf)
        chunk_output = chunks_root
        chunk_content = find_content_list(chunk_output, chunk_stem)
        if chunk_content.exists() and not doc.get("rerun_existing_chunks"):
            log.write(f"\n=== CHUNK {chunk_stem} pages {start + 1}-{end + 1} skipped_existing ===\n")
            chunk_infos.append({
                "stem": chunk_stem,
                "start": start,
                "end": end,
                "returncode": 0,
                "content_list": str(chunk_content),
                "skipped_existing": True,
            })
            returncodes.append(0)
            continue
        proc_returncode = 1
        chunk_method = None
        chunk_attempts = [
            ("pipeline_auto", ["-b", "pipeline", "-m", "auto"]),
            ("pipeline_txt", ["-b", "pipeline", "-m", "txt", "--table", "false", "--image-analysis", "false"]),
        ]
        for chunk_method, method_args in chunk_attempts:
            cmd = [
                manifest["mineru_exe"],
                *api_args(manifest),
                "-p",
                str(chunk_pdf),
                "-o",
                str(chunk_output),
                *method_args,
                "-s",
                str(start),
                "-e",
                str(end),
            ]
            log.write(f"\n=== CHUNK {chunk_stem} pages {start + 1}-{end + 1} {chunk_method} ===\n")
            log.write(f"COMMAND: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            proc_returncode = proc.returncode
            chunk_content = find_content_list(chunk_output, chunk_stem)
            if proc.returncode == 0 and chunk_content.exists():
                break
        returncodes.append(proc.returncode)
        chunk_infos.append({
            "stem": chunk_stem,
            "start": start,
            "end": end,
            "returncode": proc_returncode,
            "method": chunk_method,
            "content_list": str(chunk_content),
        })
        if proc_returncode != 0 or not chunk_content.exists():
            return {
                "name": "chunked_pipeline",
                "returncode": proc_returncode,
                "chunks": chunk_infos,
                "content_list_exists": False,
            }
        time.sleep(float(doc.get("mineru_chunk_cooldown_seconds") or 3))

    ok = merge_chunk_outputs(chunk_infos, merged_content)
    return {
        "name": "chunked_pipeline",
        "returncode": 0 if ok and all(code == 0 for code in returncodes) else 1,
        "chunks": chunk_infos,
        "content_list": str(merged_content),
        "content_list_exists": merged_content.exists(),
    }


def run_doc(doc, manifest, force=False):
    output_root = Path(manifest["output_root"]) / "mineru_output"
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = stage_pdf(doc, manifest)
    stem = doc["slug"]
    content_list = find_content_list(output_root, stem)
    log_path = Path(manifest["output_root"]) / "logs" / f"{doc['slug']}_mineru.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if content_list.exists() and not force:
        return {
            "slug": doc["slug"],
            "status": "skipped_existing",
            "content_list": str(content_list),
            "log": str(log_path),
        }

    env = os.environ.copy()
    env.setdefault("MINERU_MODEL_SOURCE", "local")
    env.setdefault("MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS", "900")
    for key, value in (manifest.get("environment") or {}).items():
        env.setdefault(str(key), str(value))

    attempts = []
    final_returncode = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        if not doc.get("mineru_prefer_chunked"):
            for attempt in build_attempts(doc, manifest, output_root, pdf_path):
                cmd = [manifest["mineru_exe"], *api_args(manifest), *attempt["args"]]
                log.write(f"\n=== ATTEMPT {attempt['name']} ===\n")
                log.write(f"COMMAND: {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                content_list = find_content_list(output_root, stem)
                final_returncode = proc.returncode
                attempts.append({"name": attempt["name"], "returncode": proc.returncode})
                if proc.returncode == 0 and content_list.exists():
                    break
        if not content_list.exists() or (force and doc.get("mineru_prefer_chunked")):
            chunk_attempt = run_chunked(doc, manifest, pdf_path, env, log)
            final_returncode = chunk_attempt.get("returncode")
            attempts.append(chunk_attempt)

    completed = content_list.exists()
    return {
        "slug": doc["slug"],
        "status": "completed" if completed else "failed",
        "returncode": final_returncode,
        "attempts": attempts,
        "content_list": str(content_list),
        "content_list_exists": completed,
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run local MinerU for documents listed in the batch manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Run only selected slug(s). Can be repeated.")
    parser.add_argument("--force", action="store_true", help="Re-run MinerU even when content_list already exists.")
    parser.add_argument("--api-url", help="Use an already running mineru-api server instead of starting one per command.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.api_url:
        manifest["api_url"] = args.api_url
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [run_doc(doc, manifest, force=args.force) for doc in docs]
    status_path = Path(manifest["output_root"]) / "mineru_batch_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
