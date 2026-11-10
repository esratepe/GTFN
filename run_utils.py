"""Deney kaydı ve test sızıntısız seçim için yardımcı fonksiyonlar.

Bu modül orijinal GTFN koduna eklenmiştir. Orijinal model, veri ve eğitim
fonksiyonlarına dokunmaz; yalnızca:
  * eğitim dışı piksellerden sınıf başına küçük bir doğrulama kümesi ayırır,
  * koşu bilgilerini (parametreler, git commit, ortam, metrikler) diske yazar,
  * isteğe bağlı olarak Weights & Biases'e kayıt atar.
"""

import json
import os
import platform
import subprocess
import time

import numpy as np


# ---------------------------------------------------------------------------
# Doğrulama kümesi
# ---------------------------------------------------------------------------
def carve_validation(test_pos, y_test, num_classes, max_per_class=10):
    """Test havuzundan sınıf başına doğrulama pikseli ayırır.

    Kural (eşit protokolle aynı): n_val = min(max_per_class, n_kalan // 2).
    Hiç rastgelelik kullanmaz: orijinal kod test indekslerini zaten karıştırdığı
    için her sınıfın ilk n_val pikseli alınır. Böylece eğitim kümesi ve
    rastgele sayı üreticisinin durumu orijinal koşuyla birebir aynı kalır.

    Dönüş: (val_pos, y_val, test_pos_yeni, y_test_yeni, val_per_class)
    """
    y_test = np.asarray(y_test)
    val_mask = np.zeros(len(y_test), dtype=bool)
    val_per_class = []
    for c in range(num_classes):
        idx = np.flatnonzero(y_test == c)
        n_val = min(max_per_class, len(idx) // 2)
        val_mask[idx[:n_val]] = True
        val_per_class.append(int(n_val))
    return (test_pos[val_mask], y_test[val_mask],
            test_pos[~val_mask], y_test[~val_mask], val_per_class)


# ---------------------------------------------------------------------------
# Ortam ve kod sürümü bilgisi
# ---------------------------------------------------------------------------
def git_info(repo_dir):
    """Kodun hangi commit'ten çalıştığını ve commit'lenmemiş değişiklik olup
    olmadığını döndürür."""
    def _run(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL,
                text=True).strip()
        except Exception:
            return None
    commit = _run("rev-parse", "--short", "HEAD")
    status = _run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "remote": _run("config", "--get", "remote.origin.url"),
    }


def env_info():
    import torch
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


# ---------------------------------------------------------------------------
# Diske kayıt
# ---------------------------------------------------------------------------
def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "item") and callable(obj.item):  # torch skaler
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # yarım dosya kalmasın


def save_history(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k))
                             for k in keys) + "\n")


# ---------------------------------------------------------------------------
# Weights & Biases (isteğe bağlı)
# ---------------------------------------------------------------------------
class WandbLogger:
    """wandb kurulu değilse ya da proje adı verilmemişse hiçbir şey yapmaz."""

    def __init__(self, project, config, name=None, group=None, tags=None,
                 run_dir=None):
        self.run = None
        if not project:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] paket yok, W&B kaydı atlanıyor.")
            return
        try:
            self.run = wandb.init(project=project, config=_to_jsonable(config),
                                  name=name, group=group, tags=tags,
                                  dir=run_dir)
        except Exception as e:  # giriş yapılmamışsa koşu yine de sürsün
            print(f"[wandb] başlatılamadı, W&B kaydı atlanıyor: {e}")
            self.run = None

    def log(self, data, step=None):
        if self.run is not None:
            self.run.log(_to_jsonable(data), step=step)

    def summary(self, data):
        if self.run is not None:
            for k, v in _to_jsonable(data).items():
                self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            self.run.finish()


def now_str():
    """İstanbul saatiyle 'YYYY-AA-GG SS:DD' (Colab sunucusu UTC'dedir)."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
