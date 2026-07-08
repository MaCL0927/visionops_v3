"""v3 服务端内置标注器服务。

该服务把 v2 标注器的核心能力迁移到 v3 的 batch 目录结构：
server_data/batches/<batch_id>/raw/{all_images|images,labels,labels_auto}。
"""

from __future__ import annotations

import json
import os
import random
import shutil
import shlex
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ingest_service import BatchService
from .label_io import list_images, parse_yolo_label, save_yolo_label

try:  # Pillow 是标注器读取尺寸/裁剪 ROI 的唯一额外依赖。
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

QUICK_TRAIN_MIN_IMAGES = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_IMAGES", "5"))
QUICK_TRAIN_EPOCHS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_EPOCHS", "50"))
QUICK_TRAIN_IMGSZ = int(os.environ.get("VISIONOPS_QUICK_TRAIN_IMGSZ", "640"))
QUICK_TRAIN_BATCH = int(os.environ.get("VISIONOPS_QUICK_TRAIN_BATCH", "2"))
QUICK_DET_MODEL = os.environ.get("VISIONOPS_QUICK_DET_MODEL", "models/pretrained/yolov8n.pt")
QUICK_OBB_MODEL = os.environ.get("VISIONOPS_QUICK_OBB_MODEL", "models/pretrained/yolov8n-obb.pt")
QUICK_SEG_MODEL = os.environ.get("VISIONOPS_QUICK_SEG_MODEL", "models/pretrained/yolov8n-seg.pt")
QUICK_YOLO_CMD = os.environ.get("VISIONOPS_QUICK_YOLO_CMD", "yolo")
AUTO_LABEL_CONF = float(os.environ.get("VISIONOPS_QUICK_AUTO_CONF", "0.25"))
QUICK_TRAIN_MIN_PER_CLASS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_PER_CLASS", "3"))
ROI_CLS_DEFAULT_CONF = float(os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_CONF", "0.35"))
ROI_CLS_DEFAULT_PADDING = float(os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_PADDING", "0.05"))
ROI_CLS_DEFAULT_DET_MODEL = os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_DET_MODEL", "models/checkpoints_detection/best.pt")
ROI_CLS_MAX_CANDIDATES = int(os.environ.get("VISIONOPS_ROI_CLS_MAX_CANDIDATES", "0"))


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    text = str(value or "").strip().replace("\\", "_").replace("/", "_").replace("..", "_")
    text = "_".join(text.split())
    if not text:
        raise ValueError("名称不能为空")
    return text


def _normalize_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg"}:
        return "segmentation"
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    if task in {"classification", "cls"}:
        return "classification"
    return "detection"


def _task_to_yolo(task_type: str) -> str:
    task = _normalize_task(task_type)
    if task == "segmentation":
        return "segment"
    if task == "obb":
        return "obb"
    return "detect"


def _quick_model_for_task(task_type: str) -> str:
    task = _normalize_task(task_type)
    if task == "segmentation":
        return QUICK_SEG_MODEL
    if task == "obb":
        return QUICK_OBB_MODEL
    return QUICK_DET_MODEL


def _non_empty_label(path: Path) -> bool:
    return path.exists() and bool(path.read_text(encoding="utf-8", errors="ignore").strip())


class AnnotationJobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start(self, job_root: Path, name: str, target: Callable[..., dict[str, Any]], *args: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = job_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "run.log"
        status_path = job_dir / "status.json"
        status = {
            "job_id": job_id,
            "name": name,
            "status": "running",
            "progress": 5,
            "started_at": _now_text(),
            "finished_at": "",
            "log_path": str(log_path),
            "message": "任务已启动",
        }
        _write_json(status_path, status)
        with self.lock:
            self.jobs[job_id] = status

        def update(progress: int, message: str) -> None:
            status["progress"] = max(0, min(100, int(progress)))
            status["message"] = message
            _write_json(status_path, status)

        def runner() -> None:
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    log_file.write(f"[INFO] job_id={job_id}\n[INFO] name={name}\n")
                    log_file.flush()
                    result = target(log_file, update, *args)
                status["status"] = "success"
                status["progress"] = 100
                if isinstance(result, dict):
                    status.update(result)
                    status["progress"] = 100
                status["message"] = status.get("message") or "完成"
            except Exception as exc:
                status["status"] = "failed"
                status["progress"] = 100
                status["message"] = str(exc)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n[ERROR] {exc}\n")
            status["finished_at"] = _now_text()
            _write_json(status_path, status)
            with self.lock:
                self.jobs[job_id] = status

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    def get(self, job_root: Path, job_id: str) -> dict[str, Any]:
        status = _read_json(job_root / "jobs" / _safe_name(job_id) / "status.json", None)
        if not status:
            raise FileNotFoundError(f"未知任务: {job_id}")
        return status

    def logs(self, job_root: Path, job_id: str, tail: int = 20000) -> str:
        status = self.get(job_root, job_id)
        log_path = Path(str(status.get("log_path") or ""))
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        return text[-tail:]


class AnnotationService:
    def __init__(self, batch_service: BatchService, data_root: Path) -> None:
        self.batch_service = batch_service
        self.data_root = Path(data_root)
        self.jobs = AnnotationJobManager()

    # ----------------------------- batch paths -----------------------------
    def batch_dir(self, batch_id: str) -> Path:
        meta = self.batch_service.get_batch(batch_id)
        return Path(str(meta.get("batch_path") or self.batch_service.batches_root / batch_id))

    def raw_dir(self, batch_id: str) -> Path:
        meta = self.batch_service.get_batch(batch_id)
        raw = Path(str(meta.get("raw_path") or self.batch_dir(batch_id) / "raw"))
        if not raw.exists():
            raise FileNotFoundError(f"batch raw 目录不存在: {raw}")
        return raw

    def images_dir(self, batch_id: str) -> Path:
        raw = self.raw_dir(batch_id)
        for name in ("all_images", "images", "positive", "negative"):
            p = raw / name
            if p.is_dir():
                return p
        # 如果上传包直接把图片放在 raw 根目录，也允许读取。
        if list_images(raw):
            return raw
        raise FileNotFoundError(f"未找到图片目录: {raw}/all_images 或 {raw}/images")

    def labels_dir(self, batch_id: str) -> Path:
        p = self.raw_dir(batch_id) / "labels"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def labels_auto_dir(self, batch_id: str) -> Path:
        p = self.raw_dir(batch_id) / "labels_auto"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def classes_path(self, batch_id: str) -> Path:
        return self.raw_dir(batch_id) / "annotation_classes.json"

    def task_path(self, batch_id: str) -> Path:
        return self.raw_dir(batch_id) / "annotation_task.json"

    def quick_root(self, batch_id: str) -> Path:
        p = self.batch_dir(batch_id) / "quick_train"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def roi_raw_dir(self) -> Path:
        p = self.data_root / "raw_classification"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def roi_sessions_dir(self, batch_id: str) -> Path:
        p = self.batch_dir(batch_id) / "roi_classification_sessions"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.data_root.parent))
        except Exception:
            return str(path)

    # ----------------------------- shared state -----------------------------
    def image_size(self, path: Path) -> tuple[int, int]:
        if Image is None:
            raise RuntimeError("Pillow 未安装，无法读取图片尺寸。请安装 pillow。")
        with Image.open(path) as im:
            return im.size

    def load_classes(self, batch_id: str) -> list[str]:
        data = _read_json(self.classes_path(batch_id), {}) or {}
        names = data.get("names")
        return [str(x) for x in names] if isinstance(names, list) else []

    def save_classes(self, batch_id: str, names: list[str]) -> list[str]:
        cleaned: list[str] = []
        for name in names:
            name = str(name).strip()
            if name and name not in cleaned:
                cleaned.append(name)
        _write_json(self.classes_path(batch_id), {"names": cleaned, "num_classes": len(cleaned), "updated_at": _now_text()})
        return cleaned

    def load_task(self, batch_id: str) -> str:
        data = _read_json(self.task_path(batch_id), {}) or {}
        task = _normalize_task(data.get("task_type") or self.batch_service.get_batch(batch_id).get("task_type") or "detection")
        if task == "classification":
            task = "detection"
        return task

    def save_task(self, batch_id: str, task_type: str) -> str:
        task = _normalize_task(task_type)
        if task == "classification":
            task = "detection"
        _write_json(self.task_path(batch_id), {"task_type": task, "updated_at": _now_text()})
        return task

    def count_manual(self, batch_id: str) -> int:
        return sum(1 for p in self.labels_dir(batch_id).glob("*.txt") if _non_empty_label(p))

    def count_auto(self, batch_id: str) -> int:
        return sum(1 for p in self.labels_auto_dir(batch_id).glob("*.txt") if _non_empty_label(p))

    def session_info(self, batch_id: str) -> dict[str, Any]:
        meta = self.batch_service.get_batch(batch_id)
        images = list_images(self.images_dir(batch_id))
        if not images:
            raise FileNotFoundError("当前 batch 中没有图片")
        return {
            "batch_id": batch_id,
            "device_id": meta.get("device_id", ""),
            "customer_id": meta.get("customer_id", ""),
            "images_dir": str(self.images_dir(batch_id)),
            "labels_dir": str(self.labels_dir(batch_id)),
            "labels_auto_dir": str(self.labels_auto_dir(batch_id)),
            "classes_path": str(self.classes_path(batch_id)),
            "classes": self.load_classes(batch_id),
            "images": [p.name for p in images],
            "total": len(images),
            "manual_label_count": self.count_manual(batch_id),
            "auto_label_count": self.count_auto(batch_id),
            "quick_train": _read_json(self.quick_root(batch_id) / "quick_state.json", {}) or {},
            "last_auto_label": (_read_json(self.quick_root(batch_id) / "quick_state.json", {}) or {}).get("last_auto_label", {}),
            "default_task_type": self.load_task(batch_id),
        }

    def image_meta(self, batch_id: str, index: int) -> dict[str, Any]:
        images = list_images(self.images_dir(batch_id))
        if index < 0 or index >= len(images):
            raise FileNotFoundError(f"图片索引越界: {index}")
        img = images[index]
        w, h = self.image_size(img)
        manual = self.labels_dir(batch_id) / f"{img.stem}.txt"
        auto = self.labels_auto_dir(batch_id) / f"{img.stem}.txt"
        if manual.exists():
            source, active = "manual", manual
        elif _non_empty_label(auto):
            source, active = "auto", auto
        else:
            source, active = "none", manual
        task = self.load_task(batch_id)
        return {
            "index": index,
            "filename": img.name,
            "image_url": f"/api/annotator/file/{index}?batch_id={batch_id}",
            "image_w": w,
            "image_h": h,
            "manual_label_path": str(manual),
            "auto_label_path": str(auto),
            "label_path": str(active),
            "label_source": source,
            "needs_confirm": source == "auto",
            "task_type": task,
            "annotations": parse_yolo_label(active, image_w=w, image_h=h, task_type=task),
        }

    def image_file(self, batch_id: str, index: int) -> Path:
        images = list_images(self.images_dir(batch_id))
        if index < 0 or index >= len(images):
            raise FileNotFoundError(f"图片索引越界: {index}")
        return images[index]

    def save_annotation(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        filename = Path(str(payload.get("filename") or "")).name
        if not filename:
            raise ValueError("filename 不能为空")
        image_path = self.images_dir(batch_id) / filename
        if not image_path.exists():
            # 支持 raw 根目录作为图片目录时的情况。
            image_path = self.images_dir(batch_id) / filename
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {filename}")
        w, h = self.image_size(image_path)
        task = self.save_task(batch_id, str(payload.get("task_type") or "detection"))
        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("annotations 必须是列表")
        classes = payload.get("classes", [])
        if isinstance(classes, list):
            self.save_classes(batch_id, [str(x) for x in classes])
        label_path = self.labels_dir(batch_id) / f"{image_path.stem}.txt"
        save_yolo_label(label_path=label_path, annotations=annotations, image_w=w, image_h=h, task_type=task)
        return {"message": "已保存人工标注", "label_path": str(label_path), "count": len(annotations), "task_type": task}

    def confirm_auto(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        filename = Path(str(payload.get("filename") or "")).name
        if not filename:
            raise ValueError("filename 不能为空")
        image_path = self.images_dir(batch_id) / filename
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {filename}")
        src = self.labels_auto_dir(batch_id) / f"{image_path.stem}.txt"
        dst = self.labels_dir(batch_id) / f"{image_path.stem}.txt"
        if not src.exists():
            raise FileNotFoundError(f"没有找到自动标注文件: {src}")
        shutil.copy2(src, dst)
        return {"message": "已确认自动标注并复制到 labels", "label_path": str(dst)}

    # ----------------------------- quick train / auto-label -----------------------------
    def class_image_counts(self, batch_id: str, num_classes: int) -> dict[int, int]:
        counts = {i: 0 for i in range(num_classes)}
        for label_path in self.labels_dir(batch_id).glob("*.txt"):
            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            found: set[int] = set()
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                if 0 <= cid < num_classes:
                    found.add(cid)
            for cid in found:
                counts[cid] += 1
        return counts

    def _build_quick_dataset(self, batch_id: str, classes: list[str], task_type: str) -> tuple[Path, int]:
        images = list_images(self.images_dir(batch_id))
        labels_dir = self.labels_dir(batch_id)
        usable = [img for img in images if _non_empty_label(labels_dir / f"{img.stem}.txt")]
        if len(usable) < QUICK_TRAIN_MIN_IMAGES:
            raise RuntimeError(f"labels 下非空人工标注只有 {len(usable)} 张，至少需要 {QUICK_TRAIN_MIN_IMAGES} 张。")
        if not classes:
            raise RuntimeError("还没有类别信息，请先标注至少一个框并选择/新建类别。")
        counts = self.class_image_counts(batch_id, len(classes))
        insufficient = [f"{i}:{classes[i]}={counts.get(i, 0)}张" for i in range(len(classes)) if counts.get(i, 0) < QUICK_TRAIN_MIN_PER_CLASS]
        if insufficient:
            raise RuntimeError(f"快速学习前类别覆盖不足。请先为每个已创建类别至少人工确认 {QUICK_TRAIN_MIN_PER_CLASS} 张。当前不足: " + "，".join(insufficient))
        dataset_dir = self.quick_root(batch_id) / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (dataset_dir / sub).mkdir(parents=True, exist_ok=True)
        rng = random.Random(42)
        rng.shuffle(usable)
        val_count = max(1, int(len(usable) * 0.2)) if len(usable) >= 6 else 1
        val_names = {p.name for p in usable[:val_count]}
        for img in usable:
            split = "val" if img.name in val_names else "train"
            shutil.copy2(img, dataset_dir / "images" / split / img.name)
            shutil.copy2(labels_dir / f"{img.stem}.txt", dataset_dir / "labels" / split / f"{img.stem}.txt")
        names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
        data_yaml = dataset_dir / "data.yaml"
        task_line = "task: segment\n" if _normalize_task(task_type) == "segmentation" else ""
        data_yaml.write_text(
            f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\n{task_line}\nnc: {len(classes)}\nnames:\n{names_lines}\n",
            encoding="utf-8",
        )
        return data_yaml, len(usable)

    def _run_shell(self, cmd: str, log_file: Any, cwd: Path) -> None:
        log_file.write(f"[CMD] {cmd}\n")
        log_file.flush()
        proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"命令执行失败，returncode={code}")

    def _find_best_pt(self, root: Path) -> Path | None:
        candidates = sorted(root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def start_quick_train(self, batch_id: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
        task = self.save_task(batch_id, str(payload.get("task_type") or "detection"))
        classes = payload.get("classes")
        if isinstance(classes, list):
            classes = self.save_classes(batch_id, [str(x) for x in classes])
        else:
            classes = self.load_classes(batch_id)
        job_id = self.jobs.start(self.quick_root(batch_id), "quick-train", self._quick_train_worker, batch_id, task, classes, project_root)
        return {"job_id": job_id, "message": "快速学习已开始"}

    def _quick_train_worker(self, log_file: Any, update: Callable[[int, str], None], batch_id: str, task: str, classes: list[str], project_root: Path) -> dict[str, Any]:
        update(10, "正在扫描 labels 下的人工标注")
        data_yaml, count = self._build_quick_dataset(batch_id, classes, task)
        runs_dir = self.quick_root(batch_id) / "runs"
        yolo_task = _task_to_yolo(task)
        model = _quick_model_for_task(task)
        run_name = f"{yolo_task}_quick"
        run_dir = runs_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        update(25, f"开始快速学习，训练图片 {count} 张")
        cmd = (
            f"{shlex.quote(QUICK_YOLO_CMD)} {yolo_task} train "
            f"model={shlex.quote(model)} "
            f"data={shlex.quote(str(data_yaml))} "
            f"epochs={QUICK_TRAIN_EPOCHS} imgsz={QUICK_TRAIN_IMGSZ} batch={QUICK_TRAIN_BATCH} "
            f"mosaic=0.0 mixup=0.0 copy_paste=0.0 degrees=0.0 perspective=0.0 "
            f"translate=0.02 scale=0.5 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 "
            f"patience=20 amp=False "
            f"project={shlex.quote(str(runs_dir))} name={shlex.quote(run_name)} exist_ok=True"
        )
        self._run_shell(cmd, log_file, project_root)
        update(90, "正在整理快速学习模型")
        best = self._find_best_pt(run_dir) or self._find_best_pt(runs_dir)
        if not best:
            raise RuntimeError("快速学习完成，但没有找到 best.pt")
        state_path = self.quick_root(batch_id) / "quick_state.json"
        state = _read_json(state_path, {}) or {}
        state["quick_train"] = {"task_type": task, "model_path": str(best), "train_images": count, "updated_at": _now_text()}
        _write_json(state_path, state)
        return {"message": "快速学习完成", "model_path": str(best), "train_images": count}

    def start_auto_label(self, batch_id: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
        task = self.save_task(batch_id, str(payload.get("task_type") or "detection"))
        job_id = self.jobs.start(self.quick_root(batch_id), "auto-label-remaining", self._auto_label_worker, batch_id, task, project_root)
        return {"job_id": job_id, "message": "预标注剩余图片已开始"}

    def _auto_label_worker(self, log_file: Any, update: Callable[[int, str], None], batch_id: str, task: str, project_root: Path) -> dict[str, Any]:
        state = _read_json(self.quick_root(batch_id) / "quick_state.json", {}) or {}
        model_path = state.get("quick_train", {}).get("model_path")
        if not model_path:
            raise RuntimeError("还没有快速学习模型，请先点击“快速学习”。")
        model_abs = Path(model_path)
        if not model_abs.is_absolute():
            model_abs = project_root / model_abs
        if not model_abs.exists():
            raise RuntimeError(f"快速学习模型不存在: {model_abs}")
        images = list_images(self.images_dir(batch_id))
        labels_dir = self.labels_dir(batch_id)
        remaining = [p for p in images if not (labels_dir / f"{p.stem}.txt").exists()]
        if not remaining:
            return {"message": "没有剩余未确认图片", "auto_labeled_count": 0}
        labels_auto_dir = self.labels_auto_dir(batch_id)
        if labels_auto_dir.exists():
            shutil.rmtree(labels_auto_dir)
        labels_auto_dir.mkdir(parents=True, exist_ok=True)
        source_dir = self.quick_root(batch_id) / "predict_source"
        pred_root = self.quick_root(batch_id) / "predict"
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(pred_root, ignore_errors=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        for img in remaining:
            shutil.copy2(img, source_dir / img.name)
        yolo_task = _task_to_yolo(task)
        run_name = f"{yolo_task}_predict"
        update(30, "正在用快速学习模型预标注剩余图片")
        cmd = (
            f"{shlex.quote(QUICK_YOLO_CMD)} {yolo_task} predict model={shlex.quote(str(model_abs))} source={shlex.quote(str(source_dir))} "
            f"imgsz={QUICK_TRAIN_IMGSZ} conf={AUTO_LABEL_CONF} save_txt=True save_conf=False "
            f"project={shlex.quote(str(pred_root))} name={shlex.quote(run_name)} exist_ok=True"
        )
        self._run_shell(cmd, log_file, project_root)
        pred_labels = pred_root / run_name / "labels"
        count = 0
        non_empty = 0
        for img in remaining:
            src = pred_labels / f"{img.stem}.txt"
            dst = labels_auto_dir / f"{img.stem}.txt"
            if src.exists() and _non_empty_label(src):
                shutil.copy2(src, dst)
                non_empty += 1
            count += 1
        state["last_auto_label"] = {"task_type": task, "count": count, "non_empty_count": non_empty, "updated_at": _now_text()}
        _write_json(self.quick_root(batch_id) / "quick_state.json", state)
        return {"message": "预标注剩余图片完成", "auto_labeled_count": count, "auto_non_empty_count": non_empty}

    # ----------------------------- review complete -----------------------------
    def accept_reviewed(self, batch_id: str, task_type: str) -> dict[str, Any]:
        task = _normalize_task(task_type)
        if task == "classification":
            task = "detection"
        # 标注器、服务端 dataset、模型包和边缘端 Runtime 统一使用 obb/segmentation/detection。
        batch = self.batch_service.set_status(batch_id, "accepted", "annotator_review_completed", task_type=task)
        batch["manual_label_count"] = self.count_manual(batch_id)
        return {"message": "审核完成，已返回服务端控制台", "batch": batch, "redirect": "/"}

    # ----------------------------- ROI classification (v2-compatible subset) -----------------------------
    def roi_classes(self) -> list[dict[str, Any]]:
        root = self.roi_raw_dir()
        result: list[dict[str, Any]] = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            count = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            result.append({"name": d.name, "count": count})
        return result

    def add_roi_class(self, name: str) -> dict[str, Any]:
        label = _safe_name(name)
        d = self.roi_raw_dir() / label
        d.mkdir(parents=True, exist_ok=True)
        return {"name": label, "path": str(d), "count": len([p for p in d.iterdir() if p.is_file()])}

    def roi_detectors(self, project_root: Path) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        default = Path(ROI_CLS_DEFAULT_DET_MODEL)
        candidates.append(default if default.is_absolute() else project_root / default)
        ckpt = project_root / "models" / "checkpoints_detection"
        if ckpt.exists():
            candidates.extend(ckpt.rglob("*.pt"))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in candidates:
            try:
                p = p.resolve()
            except Exception:
                continue
            if str(p) in seen or not p.exists() or not p.is_file():
                continue
            seen.add(str(p))
            out.append({"name": p.name, "path": str(p), "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
        out.sort(key=lambda x: x.get("mtime", ""), reverse=True)
        return out

    def roi_sessions(self, batch_id: str) -> list[dict[str, Any]]:
        root = self.roi_sessions_dir(batch_id)
        items: list[dict[str, Any]] = []
        for manifest_path in root.glob("*/manifest.json"):
            data = _read_json(manifest_path, {}) or {}
            session_id = data.get("session_id") or manifest_path.parent.name
            candidates = data.get("items", []) if isinstance(data.get("items"), list) else []
            labeled = sum(1 for x in candidates if x.get("status") == "labeled")
            items.append({"session_id": session_id, "batch_id": batch_id, "created_at": data.get("created_at", ""), "updated_at": data.get("updated_at", ""), "total": len(candidates), "labeled": labeled, "path": str(manifest_path.parent)})
        items.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
        return items

    def roi_session_info(self, batch_id: str, project_root: Path) -> dict[str, Any]:
        images = list_images(self.images_dir(batch_id))
        return {
            "batch_id": batch_id,
            "images_dir": str(self.images_dir(batch_id)),
            "image_count": len(images),
            "raw_classification_dir": str(self.roi_raw_dir()),
            "sessions_dir": str(self.roi_sessions_dir(batch_id)),
            "detectors": self.roi_detectors(project_root),
            "classes": self.roi_classes(),
            "sessions": self.roi_sessions(batch_id),
            "defaults": {"detector_model": ROI_CLS_DEFAULT_DET_MODEL, "conf_threshold": ROI_CLS_DEFAULT_CONF, "padding_ratio": ROI_CLS_DEFAULT_PADDING, "select_policy": "conf_area"},
        }

    def start_roi_candidates(self, batch_id: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
        job_id = self.jobs.start(self.roi_sessions_dir(batch_id), "roi-cls-build-candidates", self._roi_candidates_worker, batch_id, payload, project_root)
        return {"job_id": job_id, "message": "ROI 分类候选生成任务已开始"}

    def _roi_candidates_worker(self, log_file: Any, update: Callable[[int, str], None], batch_id: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"无法导入 ultralytics.YOLO: {exc}") from exc
        if Image is None:
            raise RuntimeError("Pillow 未安装，无法裁剪 ROI")
        detector = Path(str(payload.get("detector_model") or ROI_CLS_DEFAULT_DET_MODEL))
        if not detector.is_absolute():
            detector = project_root / detector
        if not detector.exists():
            raise FileNotFoundError(f"检测模型不存在: {detector}")
        conf = float(payload.get("conf_threshold") or ROI_CLS_DEFAULT_CONF)
        padding = float(payload.get("padding_ratio") or ROI_CLS_DEFAULT_PADDING)
        target_class_id_raw = payload.get("target_class_id")
        target_class_id = None if target_class_id_raw in {None, "", "null"} else int(target_class_id_raw)
        images = list_images(self.images_dir(batch_id))
        if not images:
            raise RuntimeError("当前 batch 没有图片")
        # v2 习惯每次重建 current session。
        session_id = "current"
        session_dir = self.roi_sessions_dir(batch_id) / session_id
        shutil.rmtree(session_dir, ignore_errors=True)
        candidates_dir = session_dir / "candidates"
        previews_dir = session_dir / "previews"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)
        update(10, "正在加载检测模型")
        model = YOLO(str(detector))
        model_names = getattr(model, "names", {}) or {}
        items: list[dict[str, Any]] = []
        miss_count = 0
        max_candidates = ROI_CLS_MAX_CANDIDATES if ROI_CLS_MAX_CANDIDATES > 0 else len(images)
        for idx, image_path in enumerate(images):
            if len(items) >= max_candidates:
                break
            if idx % 5 == 0:
                update(15 + int((idx + 1) / max(1, len(images)) * 75), f"正在处理 {idx + 1}/{len(images)}")
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                result = model.predict(str(image_path), conf=conf, verbose=False)[0]
                if result.boxes is None or len(result.boxes) == 0:
                    miss_count += 1
                    continue
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                best_i = None
                best_score = -1.0
                for i in range(len(xyxy)):
                    cls_id = int(clss[i])
                    if target_class_id is not None and cls_id != target_class_id:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    score = float(confs[i]) * 0.7 + min(area / max(1, w * h), 1.0) * 0.3
                    if score > best_score:
                        best_score, best_i = score, i
                if best_i is None:
                    miss_count += 1
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[best_i]]
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                rx1 = max(0, int(round(x1 - bw * padding)))
                ry1 = max(0, int(round(y1 - bh * padding)))
                rx2 = min(w, int(round(x2 + bw * padding)))
                ry2 = min(h, int(round(y2 + bh * padding)))
                if rx2 <= rx1 or ry2 <= ry1:
                    miss_count += 1
                    continue
                item_id = f"crop_{len(items) + 1:06d}"
                crop_path = candidates_dir / f"{item_id}.jpg"
                preview_path = previews_dir / f"{item_id}.jpg"
                im.crop((rx1, ry1, rx2, ry2)).save(crop_path, quality=95)
                preview = im.copy()
                if ImageDraw is not None:
                    draw = ImageDraw.Draw(preview)
                    draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline=(0, 255, 0), width=4)
                    draw.rectangle((rx1, ry1, rx2, ry2), outline=(255, 0, 0), width=3)
                preview.thumbnail((960, 720))
                preview.save(preview_path, quality=90)
                cls_id = int(clss[best_i])
                items.append({
                    "id": item_id,
                    "status": "pending",
                    "source_image": str(image_path),
                    "source_filename": image_path.name,
                    "crop_path": str(crop_path),
                    "preview_path": str(preview_path),
                    "bbox": [round(float(v), 2) for v in [x1, y1, x2, y2]],
                    "roi_bbox": [rx1, ry1, rx2, ry2],
                    "roi_size": [rx2 - rx1, ry2 - ry1],
                    "image_size": [w, h],
                    "det_conf": float(confs[best_i]),
                    "det_class_id": cls_id,
                    "det_class_name": str(model_names.get(cls_id, cls_id) if isinstance(model_names, dict) else cls_id),
                    "assigned_label": "",
                    "exported_path": "",
                })
        manifest = {"session_id": session_id, "task_type": "roi_classification_data", "batch_id": batch_id, "detector_model": str(detector), "conf_threshold": conf, "padding_ratio": padding, "created_at": _now_text(), "updated_at": _now_text(), "total_images": len(images), "candidate_count": len(items), "miss_count": miss_count, "items": items, "roi_policy": {"by_detector_class": {}}}
        _write_json(session_dir / "manifest.json", manifest)
        return {"message": f"ROI 候选生成完成：{len(items)} 个，未检测到目标 {miss_count} 张", "session_id": session_id, "candidate_count": len(items), "miss_count": miss_count, "manifest_path": str(session_dir / "manifest.json")}

    def get_roi_session(self, batch_id: str, session_id: str) -> dict[str, Any]:
        data = _read_json(self.roi_sessions_dir(batch_id) / _safe_name(session_id) / "manifest.json", None)
        if not data:
            raise FileNotFoundError(f"未找到 ROI session: {session_id}")
        return data

    def roi_file(self, batch_id: str, session_id: str, kind: str, filename: str) -> Path:
        if kind not in {"candidates", "previews"}:
            raise ValueError(f"不支持的文件类型: {kind}")
        path = self.roi_sessions_dir(batch_id) / _safe_name(session_id) / kind / Path(filename).name
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        return path

    def label_roi(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = _safe_name(str(payload.get("session_id") or "current"))
        item_id = str(payload.get("item_id") or "")
        label = _safe_name(str(payload.get("label") or ""))
        self.add_roi_class(label)
        data = self.get_roi_session(batch_id, session_id)
        item = next((x for x in data.get("items", []) if x.get("id") == item_id), None)
        if not item:
            raise FileNotFoundError(f"未找到候选项: {item_id}")
        crop_path = Path(str(item.get("crop_path") or ""))
        if not crop_path.exists():
            raise FileNotFoundError(f"ROI 图片不存在: {crop_path}")
        dst_dir = self.roi_raw_dir() / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{session_id}_{item_id}_{Path(str(item.get('source_filename') or item_id)).stem}{crop_path.suffix.lower()}"
        old = str(item.get("exported_path") or "")
        if old and Path(old).exists() and Path(old) != dst_path:
            try:
                Path(old).unlink()
            except Exception:
                pass
        shutil.copy2(crop_path, dst_path)
        item.update({"status": "labeled", "assigned_label": label, "exported_path": str(dst_path), "labeled_at": _now_text()})
        data["updated_at"] = _now_text()
        _write_json(self.roi_sessions_dir(batch_id) / session_id / "manifest.json", data)
        return {"message": f"已保存为分类样本: {label}", "item": item, "classes": self.roi_classes(), "manifest": data}

    def skip_roi(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = _safe_name(str(payload.get("session_id") or "current"))
        item_id = str(payload.get("item_id") or "")
        data = self.get_roi_session(batch_id, session_id)
        item = next((x for x in data.get("items", []) if x.get("id") == item_id), None)
        if not item:
            raise FileNotFoundError(f"未找到候选项: {item_id}")
        item.update({"status": "skipped", "assigned_label": "", "exported_path": "", "skipped_at": _now_text()})
        data["updated_at"] = _now_text()
        _write_json(self.roi_sessions_dir(batch_id) / session_id / "manifest.json", data)
        return {"message": "已跳过", "item": item, "manifest": data, "classes": self.roi_classes()}

    def save_roi_policy(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 精细 ROI 规则先做兼容保存，后续如果需要可扩展为重新裁剪已标样本。
        session_id = _safe_name(str(payload.get("session_id") or "current"))
        item_id = str(payload.get("item_id") or "")
        data = self.get_roi_session(batch_id, session_id)
        item = next((x for x in data.get("items", []) if x.get("id") == item_id), None)
        if not item:
            raise FileNotFoundError(f"未找到候选项: {item_id}")
        key = f"{item.get('det_class_id')}:{item.get('det_class_name')}"
        policy = data.setdefault("roi_policy", {})
        by_class = policy.setdefault("by_detector_class", {})
        by_class[key] = {"enabled": bool(payload.get("enabled", True)), "relative_box": payload.get("relative_box") or {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, "updated_at": _now_text()}
        data["updated_at"] = _now_text()
        _write_json(self.roi_sessions_dir(batch_id) / session_id / "manifest.json", data)
        return {"message": f"已保存检测类别 {key} 的精细 ROI 规则", "class_key": key, "policy": by_class[key], "rebuilt_count": 0, "manifest": data, "classes": self.roi_classes()}
