"""GTFN'yi birden fazla veri seti ve seed için çalıştırır.

Her koşu kendi klasörüne yazar:
    <out_root>/<veri>/seed<kk>_<commit>/
        config.json   parametreler, seed, commit, ortam, bölme boyutları
        metrics.json  sonuçlar (koşu bittiğinde yazılır)
        history.csv   epoch bazında kayıp/doğruluk
        confusion_test.npy
        run.log       GTFN.py'nin bütün ekran çıktısı

Aynı commit ile tamamlanmış bir koşu (metrics.json var) yeniden çalıştırılmaz,
böylece Colab bağlantısı koparsa kaldığın yerden devam edebilirsin. Kodu
değiştirip yeni commit attığında koşular yeni klasörlere yazılır; eski
sonuçlar yanlışlıkla yeniden kullanılmaz.

Örnek:
    python scripts/run_seeds.py --out_root /content/drive/MyDrive/hsi-gnn-PHD/results/GTFN_noleak \
        --datasets Indian PaviaU Salinas --seeds 0-9 --wandb_project hsi-gtfn
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import run_utils as ru  # noqa: E402

# Orijinal makale/README ayarları
PARAMS = {
    "Indian":  {"patches": 9, "n_gcn": 21, "pca_band": 70, "epoches": 200},
    "PaviaU":  {"patches": 9, "n_gcn": 15, "pca_band": 50, "epoches": 200},
    "Salinas": {"patches": 9, "n_gcn": 21, "pca_band": 70, "epoches": 200},
}


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def summarize(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", required=True)
    p.add_argument("--datasets", nargs="+", default=["Indian", "PaviaU", "Salinas"])
    p.add_argument("--seeds", default="0-9", help="ör. 0-9 ya da 0,3,5")
    p.add_argument("--epoches", type=int, default=None, help="hızlı deneme için (ör. 20)")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--allow_dirty", action="store_true",
                   help="commit'lenmemiş değişiklikle çalıştırmaya izin ver")
    p.add_argument("--exp_id", default="", help="deney günlüğündeki kimlik, ör. E002")
    p.add_argument("--note", default="", help="bu deneyin amacı (deney kaydına yazılır)")
    args = p.parse_args()
    started = ru.now_str()

    git = ru.git_info(REPO)
    if git["dirty"] and not args.allow_dirty:
        sys.exit("Repoda commit'lenmemiş değişiklik var. Önce commit + push yap "
                 "(ya da bilerek --allow_dirty ekle).")
    commit = git["commit"] or "nocommit"
    tag = commit + ("-dirty" if git["dirty"] else "") + ("" if args.epoches is None else "-ep{}".format(args.epoches))
    seeds = parse_seeds(args.seeds)
    print("commit={}  seeds={}  out_root={}".format(tag, seeds, args.out_root))

    all_rows = {}
    for ds in args.datasets:
        prm = dict(PARAMS[ds])
        if args.epoches is not None:
            prm["epoches"] = args.epoches
        all_rows[ds] = []
        print("\n" + "=" * 60 + "\n" + ds + "  " + json.dumps(prm) + "\n" + "=" * 60)
        for seed in seeds:
            run_dir = os.path.join(args.out_root, ds, "seed{:02d}_{}".format(seed, tag))
            mfile = os.path.join(run_dir, "metrics.json")
            if os.path.exists(mfile):
                with open(mfile) as f:
                    m = json.load(f)
                print("  seed={:2d}  ATLANDI (tamamlanmış)  OA={:.2f}".format(seed, m["selected"]["OA"]))
                all_rows[ds].append(m)
                continue
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, os.path.join(REPO, "GTFN.py"),
                   "--dataset", ds, "--seed", str(seed),
                   "--epoches", str(prm["epoches"]), "--patches", str(prm["patches"]),
                   "--n_gcn", str(prm["n_gcn"]), "--pca_band", str(prm["pca_band"]),
                   "--out_dir", run_dir]
            if args.wandb_project:
                cmd += ["--wandb_project", args.wandb_project,
                        "--wandb_group", "GTFN_noleak_{}_{}".format(ds, tag)]
            print("  seed={:2d}  çalışıyor...".format(seed), end="", flush=True)
            t0 = time.time()
            with open(os.path.join(run_dir, "run.log"), "w") as log:
                proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
            dt = time.time() - t0
            if proc.returncode != 0 or not os.path.exists(mfile):
                print(" HATA ({:.0f} sn). Son satırlar:".format(dt))
                with open(os.path.join(run_dir, "run.log")) as f:
                    print("".join(f.readlines()[-15:]))
                continue
            with open(mfile) as f:
                m = json.load(f)
            all_rows[ds].append(m)
            print(" OA={:.2f}  (son epoch {:.2f}, sızıntılı {:.2f})  {:.0f} sn".format(
                m["selected"]["OA"], m["last_epoch"]["OA"], m["test_oracle_leaky"]["OA"], dt))

    # Özet tablo
    print("\n" + "=" * 60 + "\nÖZET (ortalama ± std, test kümesi)\n" + "=" * 60)
    summary = {}
    for ds, ms in all_rows.items():
        if not ms:
            continue
        sel = [m["selected"] for m in ms]
        last = [m["last_epoch"] for m in ms]
        leak = [m["test_oracle_leaky"] for m in ms]
        s = {"n_runs": len(ms), "commit": tag}
        for name, rows in (("selected", sel), ("last_epoch", last), ("leaky", leak)):
            for k in ("OA", "AA", "Kappa"):
                s["{}_{}_mean".format(name, k)], s["{}_{}_std".format(name, k)] = summarize(rows, k)
        summary[ds] = s
        print("{:8s} n={:2d}  OA={:.2f}±{:.2f}  AA={:.2f}±{:.2f}  κ={:.2f}±{:.2f}   | son epoch OA={:.2f}  sızıntılı OA={:.2f}".format(
            ds, len(ms), s["selected_OA_mean"], s["selected_OA_std"], s["selected_AA_mean"], s["selected_AA_std"],
            s["selected_Kappa_mean"], s["selected_Kappa_std"], s["last_epoch_OA_mean"], s["leaky_OA_mean"]))
    os.makedirs(args.out_root, exist_ok=True)
    ru.save_json(os.path.join(args.out_root, "summary_{}.json".format(tag)), summary)
    print("\nÖzet kaydedildi: summary_{}.json".format(tag))

    # Deney kaydı: her çalıştırma DENEY_KAYDI.md dosyasının sonuna eklenir
    expected = len(args.datasets) * len(seeds)
    done = sum(len(v) for v in all_rows.values())
    lines = [
        "",
        "## {} {} — GTFN".format(args.exp_id or "(kimliksiz)", started),
        "",
        "- **Amaç:** {}".format(args.note or "_(not girilmedi, --note ile ekle)_"),
        "- **Bitiş:** {}  ·  **Durum:** {}/{} koşu tamamlandı".format(ru.now_str(), done, expected),
        "- **Kod:** `{}` ({})".format(tag, git.get("remote") or "-"),
        "- **Veri setleri:** {}  ·  **Seed'ler:** {}".format(", ".join(args.datasets), args.seeds),
        "- **Epoch:** {}".format(args.epoches if args.epoches is not None else "orijinal (200)"),
        "- **Seçim:** doğrulama kümesi (sınıf başına ≤10 piksel, eğitim dışından)",
        "- **Klasör:** `{}`".format(args.out_root),
        "- **W&B:** {}".format(args.wandb_project or "kullanılmadı"),
        "",
        "| Veri | n | OA (val seçili) | AA | κ | OA son epoch | OA sızıntılı | Sızıntı farkı |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds, s in summary.items():
        lines.append("| {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} | {:+.2f} |".format(
            ds, s["n_runs"], s["selected_OA_mean"], s["selected_OA_std"],
            s["selected_AA_mean"], s["selected_AA_std"],
            s["selected_Kappa_mean"], s["selected_Kappa_std"],
            s["last_epoch_OA_mean"], s["leaky_OA_mean"],
            s["leaky_OA_mean"] - s["selected_OA_mean"]))
    lines.append("")
    log_path = os.path.join(args.out_root, "DENEY_KAYDI.md")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a") as f:
        if new_file:
            f.write("# Otomatik deney kaydı\n\nBu dosyayı `scripts/run_seeds.py` doldurur; elle düzenleme gerekmez.\n")
        f.write("\n".join(lines) + "\n")
    print("Deney kaydına eklendi: " + log_path)


if __name__ == "__main__":
    main()
