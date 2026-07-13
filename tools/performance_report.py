import argparse
import json
import math
import os
import statistics

import numpy as np
from PIL import Image


def load_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as source:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def workload_key(record):
    return json.dumps(record['workload'], ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def summarize(records):
    completed = [record for record in records if record.get('status') == 'complete']
    times = [record['total_seconds'] for record in completed]
    peaks = [record['peak_vram_mb'] for record in completed if 'peak_vram_mb' in record]
    return {
        'runs': len(completed),
        'median_seconds': statistics.median(times) if times else None,
        'median_peak_vram_mb': statistics.median(peaks) if peaks else None,
    }


def compare_images(reference_path, candidate_path):
    if not os.path.isfile(reference_path) or not os.path.isfile(candidate_path):
        return None
    reference = np.asarray(Image.open(reference_path).convert('RGB'), dtype=np.float32)
    candidate = np.asarray(Image.open(candidate_path).convert('RGB'), dtype=np.float32)
    if reference.shape != candidate.shape:
        return None
    mse = float(np.mean((reference - candidate) ** 2))
    return {
        'mae': float(np.mean(np.abs(reference - candidate))),
        'psnr': float('inf') if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse)),
    }


def compare(records, baseline_label, candidate_label):
    baseline = [record for record in records if record.get('label') == baseline_label]
    candidate = [record for record in records if record.get('label') == candidate_label]
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)

    speedup = None
    if baseline_summary['median_seconds'] and candidate_summary['median_seconds']:
        speedup = baseline_summary['median_seconds'] / candidate_summary['median_seconds']

    by_workload = {workload_key(record): record for record in baseline if record.get('status') == 'complete'}
    image_scores = []
    for record in candidate:
        reference = by_workload.get(workload_key(record))
        if reference is None:
            continue
        for reference_image, candidate_image in zip(reference.get('images', []), record.get('images', [])):
            score = compare_images(reference_image['path'], candidate_image['path'])
            if score is not None:
                image_scores.append(score)

    return baseline_summary, candidate_summary, speedup, image_scores


def main():
    parser = argparse.ArgumentParser(description='Compare Fooocus performance benchmark runs.')
    parser.add_argument('log', help='JSONL file written by --performance-log.')
    parser.add_argument('--baseline', required=True, help='Baseline run label.')
    parser.add_argument('--candidate', required=True, help='Candidate run label.')
    args = parser.parse_args()

    baseline, candidate, speedup, image_scores = compare(
        load_records(args.log), args.baseline, args.candidate)
    print(f'Baseline:  {baseline}')
    print(f'Candidate: {candidate}')
    print(f'Speedup:   {speedup:.3f}x' if speedup is not None else 'Speedup:   unavailable')
    if image_scores:
        print(f'Image MAE: {statistics.median(score["mae"] for score in image_scores):.4f}')
        print(f'Image PSNR: {statistics.median(score["psnr"] for score in image_scores):.2f} dB')


if __name__ == '__main__':
    main()
