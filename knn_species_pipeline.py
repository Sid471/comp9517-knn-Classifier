"""
COMP9517 26T2 Group Project - Siddhant's part
=============================================
Method: K-Nearest-Neighbours (KNN) classifier on combined handcrafted features
        (HSV colour histogram + HOG + LBP) with PCA dimensionality reduction.

This is the "traditional pipeline" counterpart to the DL/CNN method. It pairs a
classical classifier (KNN) with handcrafted descriptors, so its results sit next
to Hunter's SVM and Louie/Abdul's Random Forest for a clean classifier comparison.

Designed to run on Google Colab (free T4 is fine - this pipeline is CPU-only).

------------------------------------------------------------------------------
WHAT YOU NEED TO DO BEFORE RUNNING (read this):
------------------------------------------------------------------------------
1. Put this file in Colab, OR paste each numbered "# ==== SECTION ====" block
   into its own Colab cell. Running it top-to-bottom also works.

2. Set the paths + column names in the CONFIG block below to match the shared
   Drive folder (CV_Proj_Dataset). The ONLY thing you must verify by hand is the
   two CSV column names (image path column + label column) - open one of the
   *_500_*.csv files and check the header row, then set LABEL_CSV_COLS.

3. First run extracts only the ~25k images your 500-class subset needs out of
   train_mini.tar.gz (NOT all 500k), and caches features to Drive, so later runs
   are fast.

Everything is controlled from CONFIG - you should not need to touch the logic.
"""

# ==== SECTION 0: INSTALL / IMPORTS ==========================================
# On Colab these are already installed except maybe nothing extra is needed.
# If a package is missing run:  !pip install scikit-image scikit-learn opencv-python-headless
import os
import io
import time
import json
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path

import cv2
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")           # safe on headless; on Colab drop this line to see inline
import matplotlib.pyplot as plt


# ==== SECTION 1: CONFIG =====================================================
class CONFIG:
    # --- Where the data lives (Google Drive, mounted at /content/drive) -----
    # After drive.mount, the shared folder usually appears under "Shared with me".
    # Easiest: add a shortcut to the folder in your own Drive, then point here.
    DRIVE_ROOT   = "/content/drive/MyDrive/CV_Proj_Dataset"          # SHARED folder (read-only for you): CSVs + archives
    RESULTS_ROOT = "/content/drive/MyDrive/COMP9517 GROUP PROJECT"   # YOUR private folder: caches + outputs written here

    # Two archives are needed (verified from the CSV file_name prefixes):
    #   train + val images  -> train_mini.tar.gz  (file_name starts "train_mini/")
    #   test images         -> val.tar.gz         (file_name starts "val/")
    # val.tar.gz (8.4 GB) was NOT in the shared Drive folder - get it from your
    # teammate or download it from the official iNat2021 URL, and put it here.
    TRAIN_TAR    = "train_mini.tar.gz"          # 41.57 GB, holds train_mini/ images
    VAL_TAR      = "val.tar.gz"                  # 8.4 GB,  holds val/ images (= test set)
    # If you downloaded val.tar.gz to Colab's LOCAL disk instead of Drive
    # (e.g. !wget -O /content/val.tar.gz ...), set this to that path. Leave as
    # None to look for val.tar.gz inside DRIVE_ROOT.
    VAL_TAR_PATH = None                         # val.tar.gz is now in DRIVE_ROOT; set to
                                                # "/content/val.tar.gz" only if using a local copy
    TRAIN_CSV    = "train_label_500_species.csv"  # 500-class subset, train split (20000 rows)
    VAL_CSV      = "val_label_500_species.csv"     # 500-class subset, val  split (5000 rows)
    TEST_CSV     = "test_label_500_species.csv"    # 500-class subset, test split (5000 rows)

    # --- Local (fast) working dirs on the Colab VM -------------------------
    WORK_DIR     = "/content/work"              # extracted images go here
    CACHE_DIR    = os.path.join(RESULTS_ROOT, "knn_cache")  # feature cache (persists on Drive)

    # --- CSV schema:  (image_path_column, label_column) --------------------
    # !!! VERIFY THESE against the CSV header row. Common iNat column names below.
    # If the CSV stores a path like "train_mini/00123_Genus_species/abcd.jpg",
    # set IMG_PATH_COL to that column. The loader also tolerates a bare filename.
    IMG_PATH_COL = "file_name"      # <-- CHANGE if your CSV calls it "image", "path", ...
    LABEL_COL    = "category_id"    # <-- CHANGE if your CSV calls it "species", "label", "class_id"

    # --- Feature extraction (v2: spatial colour + per-block L2 norm) --------
    IMG_SIZE     = 128              # images resized to IMG_SIZE x IMG_SIZE
    HSV_BINS     = (8, 8, 8)        # colour histogram bins per channel
    COLOR_GRID   = 2               # split image into COLOR_GRID x COLOR_GRID cells
                                    # and take one HSV histogram per cell (captures
                                    # WHERE colours are, not just how much)
    HOG_PPC      = (16, 16)         # HOG pixels-per-cell (bigger = fewer dims)
    HOG_CPB      = (2, 2)           # HOG cells-per-block
    HOG_ORIENT   = 9
    LBP_P        = 24               # LBP neighbours
    LBP_R        = 3                # LBP radius
    LBP_BINS     = LBP_P + 2        # uniform LBP histogram size
    # Each descriptor block (colour / HOG / LBP) is L2-normalised, then weighted,
    # so no block dominates the distance just because it has more dimensions.
    W_COLOR      = 1.0
    W_HOG        = 1.0
    W_LBP        = 1.0
    FEATURE_VER  = "v2"            # bump this string to invalidate the feature cache

    # --- Dimensionality reduction + classifier -----------------------------
    PCA_DIMS     = 300              # PCA output dims (fit on TRAIN only)
    K_GRID       = [1, 3, 5, 7, 9, 15, 21, 31, 41, 51]   # k values tuned on VAL
    WEIGHTS_GRID = ["uniform", "distance"]
    METRIC_GRID  = ["euclidean", "cosine"]       # cosine works well on hist features

    N_JOBS       = -1               # parallelism for feature extraction
    RANDOM_SEED  = 42
    OUT_DIR      = os.path.join(RESULTS_ROOT, "knn_outputs")  # plots + results saved here


os.makedirs(CONFIG.WORK_DIR, exist_ok=True)
os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
np.random.seed(CONFIG.RANDOM_SEED)


# ==== SECTION 2: MOUNT DRIVE (Colab only) ===================================
def mount_drive():
    """Mount Google Drive. No-op if already mounted or not on Colab."""
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        print("Drive mounted.")
    except ImportError:
        print("Not on Colab - assuming DRIVE_ROOT paths already point to local data.")


# ==== SECTION 3: LOAD LABEL CSVs ============================================
def _norm_name(p):
    """Normalise a path/filename so CSV entries and tar members can be matched."""
    return os.path.basename(str(p).strip())

def load_split(csv_name):
    """Return a DataFrame with two clean columns: 'img' (basename) and 'label'."""
    path = os.path.join(CONFIG.DRIVE_ROOT, csv_name)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # tolerate a few likely column names
    img_col = CONFIG.IMG_PATH_COL if CONFIG.IMG_PATH_COL in df.columns else \
              next((cols[c] for c in ("file_name", "filename", "image", "path", "img") if c in cols), None)
    lab_col = CONFIG.LABEL_COL if CONFIG.LABEL_COL in df.columns else \
              next((cols[c] for c in ("category_id", "label", "class_id", "species", "species_id") if c in cols), None)
    if img_col is None or lab_col is None:
        raise ValueError(
            f"Could not find image/label columns in {csv_name}. "
            f"Columns present: {list(df.columns)}. "
            f"Set CONFIG.IMG_PATH_COL and CONFIG.LABEL_COL manually."
        )
    out = pd.DataFrame({
        "img_full": df[img_col].astype(str),
        "img":      df[img_col].map(_norm_name),
        "label":    df[lab_col],
    })
    print(f"{csv_name}: {len(out)} rows, {out['label'].nunique()} classes "
          f"(img col='{img_col}', label col='{lab_col}')")
    return out


# ==== SECTION 4: SELECTIVE EXTRACTION FROM THE 41 GB TAR ====================
def _archive_path(name):
    """Resolve an archive name to a full path, honouring VAL_TAR_PATH override."""
    if name == CONFIG.VAL_TAR and CONFIG.VAL_TAR_PATH:
        return CONFIG.VAL_TAR_PATH
    return os.path.join(CONFIG.DRIVE_ROOT, name)


def _extract_from_archive(tar_path, todo):
    """Stream one .tar.gz once, extract members whose basename is in `todo`."""
    if not todo:
        return
    if not os.path.exists(tar_path):
        print(f"  WARNING: archive not found: {tar_path} "
              f"({len(todo)} images from it will be missing).")
        return
    print(f"  scanning {os.path.basename(tar_path)} for {len(todo)} images...")
    t0 = time.time(); n = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in todo:
                f = tar.extractfile(member)
                if f is None:
                    continue
                with open(os.path.join(CONFIG.WORK_DIR, base), "wb") as out:
                    out.write(f.read())
                n += 1
                todo.discard(base)
                if n % 1000 == 0:
                    print(f"    extracted {n} ... ({time.time()-t0:.0f}s)")
                if not todo:
                    break
    print(f"  done: {n} images in {time.time()-t0:.0f}s, {len(todo)} still missing.")


def extract_needed_images(splits):
    """
    Extract ONLY the images our 500-class CSVs reference into WORK_DIR (flat,
    keyed by basename - basenames are unique UUIDs). Routes each image to the
    right archive based on its file_name prefix: 'train_mini/' -> TRAIN_TAR,
    'val/' -> VAL_TAR. Skips already-extracted images so re-runs are cheap.
    """
    have = set(os.listdir(CONFIG.WORK_DIR))
    todo_by_archive = {CONFIG.TRAIN_TAR: set(), CONFIG.VAL_TAR: set()}
    for s in splits:
        for full, base in zip(s["img_full"], s["img"]):
            if base in have:
                continue
            prefix = str(full).split("/", 1)[0]
            archive = CONFIG.VAL_TAR if prefix == "val" else CONFIG.TRAIN_TAR
            todo_by_archive[archive].add(base)

    total = sum(len(v) for v in todo_by_archive.values())
    if total == 0:
        print("All images already extracted.")
        return
    print(f"Need to extract {total} images across archives.")
    for archive, todo in todo_by_archive.items():
        _extract_from_archive(_archive_path(archive), todo)


# ==== SECTION 5: FEATURE EXTRACTION =========================================
def _l2(v):
    """L2-normalise a vector (unit length), safe against all-zero vectors."""
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-7)


def extract_features_one(img_path):
    """
    Combined descriptor for a single image:
        [ w_color * spatial-HSV-hist | w_hog * HOG | w_lbp * LBP-hist ]
    Each block is L2-normalised before weighting so no block dominates the
    distance purely because it has more dimensions.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (CONFIG.IMG_SIZE, CONFIG.IMG_SIZE))
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Spatial HSV colour histogram: one histogram per grid cell ---
    G = CONFIG.COLOR_GRID
    S = CONFIG.IMG_SIZE // G
    cell_hists = []
    for r in range(G):
        for c in range(G):
            cell = hsv[r * S:(r + 1) * S, c * S:(c + 1) * S]
            h = cv2.calcHist([cell], [0, 1, 2], None, CONFIG.HSV_BINS,
                             [0, 180, 0, 256, 0, 256]).flatten()
            cell_hists.append(_l2(h))          # normalise each cell
    color = _l2(np.concatenate(cell_hists))    # then normalise the whole block

    # --- 2. HOG (shape / gradient structure) ---
    hog_vec = _l2(hog(gray, orientations=CONFIG.HOG_ORIENT,
                      pixels_per_cell=CONFIG.HOG_PPC, cells_per_block=CONFIG.HOG_CPB,
                      block_norm="L2-Hys", feature_vector=True))

    # --- 3. LBP (micro-texture) ---
    lbp = local_binary_pattern(gray, CONFIG.LBP_P, CONFIG.LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, CONFIG.LBP_BINS + 1),
                               range=(0, CONFIG.LBP_BINS))
    lbp_hist = _l2(lbp_hist)

    return np.concatenate([
        CONFIG.W_COLOR * color,
        CONFIG.W_HOG   * hog_vec,
        CONFIG.W_LBP   * lbp_hist,
    ]).astype("float32")


def build_matrix(df, split_name):
    """Extract (and cache) the feature matrix X and label vector y for a split."""
    cache_x = os.path.join(CONFIG.CACHE_DIR, f"X_{split_name}_{CONFIG.FEATURE_VER}.npy")
    cache_y = os.path.join(CONFIG.CACHE_DIR, f"y_{split_name}_{CONFIG.FEATURE_VER}.npy")
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"[{split_name}] loading cached features.")
        return np.load(cache_x), np.load(cache_y)

    paths  = [os.path.join(CONFIG.WORK_DIR, b) for b in df["img"]]
    labels = df["label"].to_numpy()

    print(f"[{split_name}] extracting features for {len(paths)} images...")
    t0 = time.time()
    feats = Parallel(n_jobs=CONFIG.N_JOBS, verbose=5)(
        delayed(extract_features_one)(p) for p in paths
    )
    keep = [i for i, f in enumerate(feats) if f is not None]
    if not keep:
        raise RuntimeError(
            f"[{split_name}] 0 of {len(paths)} images were readable. The images "
            f"for this split are not in WORK_DIR - usually because the source "
            f"archive is missing. For the TEST split this means val.tar.gz was "
            f"not found. Download it (e.g. !wget -O /content/val.tar.gz <url>) and "
            f"make sure CONFIG.VAL_TAR_PATH points to it, then re-run."
        )
    X = np.vstack([feats[i] for i in keep])
    y = labels[keep]
    print(f"[{split_name}] X shape {X.shape} in {time.time()-t0:.0f}s "
          f"({len(paths)-len(keep)} unreadable images skipped).")

    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


# ==== SECTION 6: TUNE KNN ON THE VALIDATION SET =============================
def tune_knn(Xtr, ytr, Xva, yva):
    """Grid-search k / weights / metric on the validation set. Returns best cfg."""
    print("\nTuning KNN on validation set...")
    results = []
    best = {"acc": -1}
    for metric in CONFIG.METRIC_GRID:
        for weights in CONFIG.WEIGHTS_GRID:
            for k in CONFIG.K_GRID:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights,
                                           metric=metric, n_jobs=CONFIG.N_JOBS)
                knn.fit(Xtr, ytr)
                acc = accuracy_score(yva, knn.predict(Xva))
                results.append((metric, weights, k, acc))
                if acc > best["acc"]:
                    best = {"acc": acc, "k": k, "weights": weights, "metric": metric}
                print(f"  metric={metric:9s} weights={weights:8s} k={k:2d} "
                      f"-> val acc={acc:.4f}")
    print(f"BEST on val: {best}")
    _plot_k_sweep(results)
    return best


def _plot_k_sweep(results):
    plt.figure(figsize=(7, 5))
    for metric in CONFIG.METRIC_GRID:
        for weights in CONFIG.WEIGHTS_GRID:
            pts = [(k, a) for (m, w, k, a) in results if m == metric and w == weights]
            pts.sort()
            ks = [p[0] for p in pts]; accs = [p[1] for p in pts]
            plt.plot(ks, accs, marker="o", label=f"{metric}/{weights}")
    plt.xlabel("k (neighbours)"); plt.ylabel("validation accuracy")
    plt.title("KNN hyperparameter sweep"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    p = os.path.join(CONFIG.OUT_DIR, "knn_k_sweep.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {p}")


# ==== SECTION 7: FINAL EVALUATION ON TEST ===================================
def evaluate(best, Xtr, ytr, Xte, yte):
    """Fit best KNN on train, evaluate on held-out test with the full metric suite."""
    print("\nFinal evaluation on TEST set...")
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights=best["weights"],
                               metric=best["metric"], n_jobs=CONFIG.N_JOBS)

    t0 = time.time(); knn.fit(Xtr, ytr); train_time = time.time() - t0
    t0 = time.time(); y_pred = knn.predict(Xte); test_time = time.time() - t0
    proba = knn.predict_proba(Xte)          # for top-5

    classes = knn.classes_
    top1 = accuracy_score(yte, y_pred)
    top5 = top_k_accuracy_score(yte, proba, k=5, labels=classes)
    bal  = balanced_accuracy_score(yte, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro",
                                                  zero_division=0)

    print("\n================ RESULTS (KNN) ================")
    print(f"Config           : k={best['k']}, weights={best['weights']}, "
          f"metric={best['metric']}, PCA={CONFIG.PCA_DIMS}")
    print(f"Top-1 accuracy   : {top1:.4f}")
    print(f"Top-5 accuracy   : {top5:.4f}")
    print(f"Balanced acc     : {bal:.4f}")
    print(f"Macro precision  : {p:.4f}")
    print(f"Macro recall     : {r:.4f}")
    print(f"Macro F1         : {f1:.4f}")
    print(f"Train time (fit) : {train_time:.2f}s")
    print(f"Test time (pred) : {test_time:.2f}s  ({test_time/len(yte)*1000:.2f} ms/img)")
    print("===============================================")

    # confusion matrix (500x500 -> save as image, no cell text)
    cm = confusion_matrix(yte, y_pred, labels=classes)
    plt.figure(figsize=(9, 8))
    plt.imshow(np.log1p(cm), cmap="viridis")
    plt.title("KNN confusion matrix (log-scaled)")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.colorbar()
    cm_path = os.path.join(CONFIG.OUT_DIR, "knn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {cm_path}")

    # most-confused species pairs (useful for the report's error analysis)
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm_off[i, j] > 0:
                pairs.append((cm_off[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    print("\nTop-15 most-confused (true -> predicted, count):")
    for cnt, t, pr in pairs[:15]:
        print(f"  {t} -> {pr}: {cnt}")

    # save metrics + full per-class report
    metrics = {"top1": top1, "top5": top5, "balanced_acc": bal,
               "macro_precision": p, "macro_recall": r, "macro_f1": f1,
               "train_time_s": train_time, "test_time_s": test_time,
               "config": best, "pca_dims": CONFIG.PCA_DIMS}
    with open(os.path.join(CONFIG.OUT_DIR, "knn_metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2, default=str)
    with open(os.path.join(CONFIG.OUT_DIR, "knn_classification_report.txt"), "w") as fp:
        fp.write(classification_report(yte, y_pred, zero_division=0))
    print(f"Saved metrics + per-class report to {CONFIG.OUT_DIR}")
    return metrics


# ==== SECTION 8: MAIN =======================================================
def preflight_check():
    """Verify the Drive path, required files, and archives before heavy work."""
    print("\n===== PRE-FLIGHT CHECK =====")
    root = CONFIG.DRIVE_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"DRIVE_ROOT does not exist: {root}\n"
            f"Fix CONFIG.DRIVE_ROOT to match where your shortcut landed. "
            f"Run  !ls /content/drive/MyDrive  to see your folders."
        )
    print(f"OK  folder found: {root}")

    required_csv = [CONFIG.TRAIN_CSV, CONFIG.VAL_CSV, CONFIG.TEST_CSV]
    for name in required_csv:
        p = os.path.join(root, name)
        print(f"{'OK ' if os.path.exists(p) else 'MISSING'} csv: {name}")

    for name in [CONFIG.TRAIN_TAR, CONFIG.VAL_TAR]:
        p = _archive_path(name)
        if os.path.exists(p):
            gb = os.path.getsize(p) / 1e9
            print(f"OK  archive: {name} ({gb:.1f} GB)")
        else:
            note = "  <-- needed for the TEST set!" if name == CONFIG.VAL_TAR else ""
            print(f"MISSING archive: {name}{note}")
    print("============================\n")


def main():
    mount_drive()
    preflight_check()

    train_df = load_split(CONFIG.TRAIN_CSV)
    val_df   = load_split(CONFIG.VAL_CSV)
    test_df  = load_split(CONFIG.TEST_CSV)

    # only train/val images are inside train_mini.tar.gz; see note in Section 4
    # about val.tar.gz if your test images live in a separate archive.
    extract_needed_images([train_df, val_df, test_df])

    Xtr_raw, ytr = build_matrix(train_df, "train")
    Xva_raw, yva = build_matrix(val_df,   "val")
    Xte_raw, yte = build_matrix(test_df,  "test")

    # scale + PCA: FIT ON TRAIN ONLY, then transform val/test (no data leakage)
    print("\nScaling + PCA (fit on train only)...")
    scaler = StandardScaler().fit(Xtr_raw)
    pca    = PCA(n_components=min(CONFIG.PCA_DIMS, Xtr_raw.shape[1]),
                 random_state=CONFIG.RANDOM_SEED).fit(scaler.transform(Xtr_raw))
    tf = lambda X: pca.transform(scaler.transform(X))
    Xtr, Xva, Xte = tf(Xtr_raw), tf(Xva_raw), tf(Xte_raw)
    print(f"PCA kept {pca.n_components_} dims, "
          f"explained var = {pca.explained_variance_ratio_.sum():.3f}")

    best = tune_knn(Xtr, ytr, Xva, yva)
    evaluate(best, Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    main()
