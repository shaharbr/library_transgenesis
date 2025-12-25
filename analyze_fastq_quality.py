#!/usr/bin/env python3
"""
Analyze quality metrics from raw FASTQ files.

Calculates:
- Total sequenced reads
- Mean quality score per read
- Percentage of bases >= Q20
- Percentage of bases >= Q30

Usage:
    python analyze_fastq_quality.py <fastq_file1> [<fastq_file2> ...]

Example:
    python analyze_fastq_quality.py 7462.fastq.gz 6978.fastq.gz
"""

import sys
import gzip
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def parse_fastq(file_path):
    """
    Parse FASTQ file and yield records.

    Yields:
        Tuple of (sequence, quality_string)
    """
    open_func = gzip.open if str(file_path).endswith('.gz') else open

    with open_func(file_path, 'rt') as f:
        while True:
            # Read 4 lines per record
            header = f.readline().strip()
            if not header:
                break

            sequence = f.readline().strip()
            plus = f.readline().strip()
            quality = f.readline().strip()

            if header and sequence and quality:
                yield (sequence, quality)


def phred_to_quality(phred_char):
    """
    Convert Phred+33 character to quality score.

    Args:
        phred_char: Single character from quality string

    Returns:
        Integer quality score
    """
    return ord(phred_char) - 33


def calculate_fastq_metrics(fastq_file):
    """
    Calculate quality metrics for a FASTQ file.

    Args:
        fastq_file: Path to FASTQ file (.fastq or .fastq.gz)

    Returns:
        Dictionary with quality metrics
    """
    print(f"\nAnalyzing: {fastq_file}")

    total_reads = 0
    total_bases = 0
    bases_q20 = 0
    bases_q30 = 0
    sum_mean_quality = 0.0

    for sequence, quality in parse_fastq(fastq_file):
        total_reads += 1

        # Convert quality string to scores
        quality_scores = [phred_to_quality(q) for q in quality]

        # Count bases
        n_bases = len(quality_scores)
        total_bases += n_bases

        # Count high-quality bases
        bases_q20 += sum(1 for q in quality_scores if q >= 20)
        bases_q30 += sum(1 for q in quality_scores if q >= 30)

        # Sum of mean quality for this read
        mean_quality = np.mean(quality_scores)
        sum_mean_quality += mean_quality

        # Progress indicator
        if total_reads % 100000 == 0:
            print(f"  Processed {total_reads:,} reads...")

    # Calculate percentages and averages
    percent_q20 = 100.0 * bases_q20 / total_bases if total_bases > 0 else 0
    percent_q30 = 100.0 * bases_q30 / total_bases if total_bases > 0 else 0
    mean_quality = sum_mean_quality / total_reads if total_reads > 0 else 0

    print(f"  Complete: {total_reads:,} reads analyzed")

    return {
        'file': Path(fastq_file).name,
        'sequenced_reads': total_reads,
        'total_bases': total_bases,
        'mean_quality_score': mean_quality,
        'bases_gte_q20': bases_q20,
        'bases_gte_q30': bases_q30,
        'percent_gte_q20': percent_q20,
        'percent_gte_q30': percent_q30,
    }


def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("ERROR: Missing required arguments")
        print()
        print("Usage:")
        print("  python analyze_fastq_quality.py <fastq_file1> [<fastq_file2> ...]")
        print()
        print("Description:")
        print("  Analyzes quality metrics from raw FASTQ files.")
        print()
        print("Metrics calculated:")
        print("  - Total sequenced reads")
        print("  - Total bases")
        print("  - Mean quality score (Phred)")
        print("  - Number and percentage of bases >= Q20")
        print("  - Number and percentage of bases >= Q30")
        print()
        print("Example:")
        print("  python analyze_fastq_quality.py 7462.fastq.gz 6978.fastq.gz")
        print()
        print("Accepts both compressed (.fastq.gz) and uncompressed (.fastq) files.")
        return

    fastq_files = sys.argv[1:]

    print("="*80)
    print("FASTQ QUALITY ANALYSIS")
    print("="*80)
    print(f"\nFiles to analyze: {len(fastq_files)}")
    for f in fastq_files:
        print(f"  - {f}")

    # Analyze each file
    results = []
    for fastq_file in fastq_files:
        if not Path(fastq_file).exists():
            print(f"\nERROR: File not found: {fastq_file}")
            continue

        metrics = calculate_fastq_metrics(fastq_file)
        results.append(metrics)

    if not results:
        print("\nERROR: No files successfully analyzed")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f'fastq_quality_metrics_{timestamp}.csv'
    df.to_csv(output_csv, index=False)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print()

    # Print summary table
    for _, row in df.iterrows():
        print(f"File: {row['file']}")
        print(f"  Sequenced reads:     {row['sequenced_reads']:>15,}")
        print(f"  Total bases:         {row['total_bases']:>15,}")
        print(f"  Mean quality score:  {row['mean_quality_score']:>15.2f}")
        print(f"  Bases >= Q20:        {row['bases_gte_q20']:>15,}  ({row['percent_gte_q20']:>6.2f}%)")
        print(f"  Bases >= Q30:        {row['bases_gte_q30']:>15,}  ({row['percent_gte_q30']:>6.2f}%)")
        print()

    print(f"Results saved to: {output_csv}")
    print()

    # Print CSV-friendly format
    print("="*80)
    print("CSV FORMAT (for easy copying)")
    print("="*80)
    print()
    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
