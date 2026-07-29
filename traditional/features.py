"""Handcrafted image descriptors used by the traditional baselines."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import SIFT, hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans


def load_rgb_image(path, image_size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def extract_color_histogram(image, bins):
    hsv = rgb2hsv(image)
    channels = [
        np.histogram(hsv[:, :, index], bins=bins, range=(0.0, 1.0))[0]
        for index in range(3)
    ]
    feature = np.concatenate(channels).astype(np.float32)
    feature /= max(float(feature.sum()), 1.0)
    return feature


def extract_lbp_histogram(image, points, radius):
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)
    codes = local_binary_pattern(gray, points, radius, method="uniform")
    feature = np.bincount(
        codes.astype(np.int32).ravel(),
        minlength=points + 2,
    ).astype(np.float32)
    feature /= max(float(feature.sum()), 1.0)
    return feature


def extract_hog_feature(image, orientations, pixels_per_cell, cells_per_block):
    gray = rgb2gray(image)
    return hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_sift_descriptors(image, max_descriptors):
    gray = rgb2gray(image).astype(np.float32)
    detector = SIFT(upsampling=1)
    try:
        detector.detect_and_extract(gray)
    except RuntimeError:
        return np.empty((0, 128), dtype=np.float32)
    descriptors = detector.descriptors.astype(np.float32)
    if len(descriptors) <= max_descriptors:
        return descriptors
    scales = getattr(detector, "scales", None)
    if scales is None:
        return descriptors[:max_descriptors]
    selected = np.argsort(scales)[-max_descriptors:]
    return descriptors[selected]


def rows_fingerprint(rows):
    # Bind each cache file to the exact image order and labels.
    digest = hashlib.sha1()
    for row in rows:
        digest.update(row["file_name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["class_index"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def feature_cache_name(split_name, feature_name, args, rows):
    parts = [
        split_name,
        feature_name,
        f"size{args.image_size}",
        f"n{len(rows)}",
        rows_fingerprint(rows),
    ]
    if feature_name == "color":
        parts.append(f"bins{args.color_bins}")
    elif feature_name == "lbp":
        parts.extend([f"p{args.lbp_points}", f"r{args.lbp_radius:g}"])
    elif feature_name == "hog":
        parts.extend(
            [
                f"ori{args.hog_orientations}",
                f"ppc{args.hog_pixels_per_cell}",
                f"cpb{args.hog_cells_per_block}",
            ]
        )
    elif feature_name == "sift-bovw":
        parts.extend(
            [
                f"vocab{args.sift_vocabulary_size}",
                f"desc{args.sift_max_descriptors}",
            ]
        )
    return "_".join(parts) + ".joblib"


def extract_single_feature(path, feature_name, args):
    image = load_rgb_image(path, args.image_size)
    if feature_name == "color":
        return extract_color_histogram(image, args.color_bins)
    if feature_name == "lbp":
        return extract_lbp_histogram(image, args.lbp_points, args.lbp_radius)
    if feature_name == "hog":
        return extract_hog_feature(
            image,
            args.hog_orientations,
            args.hog_pixels_per_cell,
            args.hog_cells_per_block,
        )
    if feature_name == "sift-bovw":
        descriptors = extract_sift_descriptors(image, args.sift_max_descriptors)
        if not len(descriptors):
            return np.zeros(args.sift_vocabulary_size, dtype=np.float32)
        words = args.sift_vocabulary.predict(descriptors)
        histogram = np.bincount(
            words,
            minlength=args.sift_vocabulary_size,
        ).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        return histogram
    raise ValueError(f"Unsupported feature: {feature_name}")


def _parallel_map(function, values, workers):
    if workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values))


def load_or_extract_features(
    rows,
    split_name,
    feature_name,
    data_root,
    cache_dir,
    args,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / feature_cache_name(
        split_name,
        feature_name,
        args,
        rows,
    )
    if cache_path.exists() and not args.no_cache:
        payload = joblib.load(cache_path)
        payload["cache_hit"] = True
        payload["cache_path"] = str(cache_path)
        print(f"Loading cached {split_name} {feature_name} features: {cache_path}")
        return payload

    paths = [data_root / row["file_name"] for row in rows]
    missing = next((path for path in paths if not path.exists()), None)
    if missing is not None:
        raise FileNotFoundError(f"Missing image: {missing}")

    started = time.perf_counter()
    completed = 0

    def extract(path):
        nonlocal completed
        result = extract_single_feature(path, feature_name, args)
        completed += 1
        if completed % args.progress_every == 0 or completed == len(paths):
            elapsed = time.perf_counter() - started
            print(
                f"  {split_name} {feature_name}: "
                f"extracted {completed}/{len(paths)} in {elapsed:.1f}s"
            )
        return result

    features = _parallel_map(extract, paths, args.workers)
    extraction_seconds = time.perf_counter() - started
    payload = {
        "X": np.vstack(features).astype(np.float32),
        "y": np.asarray([int(row["class_index"]) for row in rows], dtype=np.int64),
        "paths": [row["file_name"] for row in rows],
        "extraction_seconds": float(extraction_seconds),
        "cache_hit": False,
        "cache_path": str(cache_path),
    }
    if not args.no_cache:
        joblib.dump(payload, cache_path, compress=3)
        print(f"Saved {split_name} feature cache: {cache_path}")
    return payload


def fit_or_load_sift_vocabulary(train_rows, data_root, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = cache_dir / (
        f"sift_vocab_size{args.image_size}_k{args.sift_vocabulary_size}"
        f"_images{args.sift_vocabulary_images}"
        f"_desc{args.sift_vocabulary_descriptors}"
        f"_{rows_fingerprint(train_rows)}.joblib"
    )
    if vocabulary_path.exists() and not args.no_cache:
        payload = joblib.load(vocabulary_path)
        payload["cache_hit"] = True
        print(f"Loading cached SIFT vocabulary: {vocabulary_path}")
        return payload

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sift_vocabulary_images, len(train_rows))
    selected = rng.choice(len(train_rows), size=sample_count, replace=False)
    selected_paths = [data_root / train_rows[index]["file_name"] for index in selected]
    per_image_limit = max(
        1,
        int(np.ceil(args.sift_vocabulary_descriptors / sample_count)),
    )
    started = time.perf_counter()

    def extract(path):
        image = load_rgb_image(path, args.image_size)
        return extract_sift_descriptors(image, per_image_limit)

    descriptors = _parallel_map(extract, selected_paths, args.workers)
    nonempty = [item for item in descriptors if len(item)]
    if not nonempty:
        raise RuntimeError("SIFT found no descriptors in the vocabulary image sample.")
    descriptor_matrix = np.vstack(nonempty)
    if len(descriptor_matrix) > args.sift_vocabulary_descriptors:
        # Limit KMeans input so vocabulary fitting stays within memory.
        sample = rng.choice(
            len(descriptor_matrix),
            size=args.sift_vocabulary_descriptors,
            replace=False,
        )
        descriptor_matrix = descriptor_matrix[sample]
    extraction_seconds = time.perf_counter() - started

    print(
        f"Fitting {args.sift_vocabulary_size}-word SIFT vocabulary "
        f"from {len(descriptor_matrix)} descriptors"
    )
    fit_started = time.perf_counter()
    vocabulary = MiniBatchKMeans(
        n_clusters=args.sift_vocabulary_size,
        random_state=args.seed,
        batch_size=4096,
        n_init=3,
        max_iter=100,
    )
    vocabulary.fit(descriptor_matrix)
    fit_seconds = time.perf_counter() - fit_started
    payload = {
        "model": vocabulary,
        "sampled_images": int(sample_count),
        "sampled_descriptors": int(len(descriptor_matrix)),
        "descriptor_extraction_seconds": float(extraction_seconds),
        "fit_seconds": float(fit_seconds),
        "cache_hit": False,
        "path": str(vocabulary_path),
    }
    if not args.no_cache:
        joblib.dump(payload, vocabulary_path, compress=3)
        print(f"Saved SIFT vocabulary: {vocabulary_path}")
    return payload
