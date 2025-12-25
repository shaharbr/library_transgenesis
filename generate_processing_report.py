#!/usr/bin/env python3
"""
Generate comprehensive processing report from barcode analysis pipeline.

Extracts statistics from all intermediate .pkl files to show:
- Number of reads and barcodes at each processing stage
- Per-sample and per-fish breakdown
- Filtering and collapsing statistics

Usage:
    python generate_processing_report.py
"""

import pickle
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob


def get_fish_display_name(fish_id):
    """
    Map fish barcode IDs to display names (Fish 1, Fish 2, etc.) in specified order.
    """
    fish_order = [
        'ACCTT', 'AATCG', 'AGAGA', 'CGGAG', 'CTACT', 'GAAAC',
        'GCGCA', 'CCTGC', 'TAGGT', 'GTCGG', 'TTTAA', 'TGCCC'
    ]

    if fish_id in fish_order:
        return f"Fish {fish_order.index(fish_id) + 1}"
    else:
        # If not in the predefined order, just return the ID
        return fish_id


def load_pickle_safe(filepath):
    """Load pickle file safely, return None if error."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def extract_raw_fastq_stats(processed_file):
    """Extract statistics from raw FASTQ processing (demultiplexing stage)."""
    print(f"Processing: {processed_file}")
    data = load_pickle_safe(processed_file)
    if data is None:
        return []

    stats = []

    sample_type = data.get('sample_type', 'unknown')

    # Extract overall stats from the 'stats' dict
    if 'stats' in data:
        file_stats = data['stats']

        if sample_type == '6978':
            # 6978 has demultiplexing stats
            total_reads = file_stats.get('total_reads', 0)
            demultiplexed_reads = file_stats.get('demultiplexed_reads', 0)
            failed_demux = file_stats.get('failed_demux', 0)
            ambiguous = file_stats.get('ambiguous_barcodes', 0)

            # Add overall 6978 stats
            stats.append({
                'stage': '0_raw_fastq',
                'sample': '6978',
                'fish': 'all',
                'fish_display': '6978',
                'n_reads': total_reads,
                'n_barcodes': 'N/A',
                'notes': f'Total reads in FASTQ file'
            })

            stats.append({
                'stage': '1_demultiplexed',
                'sample': '6978',
                'fish': 'all',
                'fish_display': '6978',
                'n_reads': demultiplexed_reads,
                'n_barcodes': 'N/A',
                'notes': f'Successfully demultiplexed ({failed_demux} failed, {ambiguous} ambiguous)'
            })

        elif sample_type == '7462':
            # 7462 has no demultiplexing
            total_reads = file_stats.get('total_reads', 0)

            stats.append({
                'stage': '0_raw_fastq',
                'sample': '7462',
                'fish': 'all',
                'fish_display': '7462',
                'n_reads': total_reads,
                'n_barcodes': 'N/A',
                'notes': f'Total reads in FASTQ file (no demultiplexing needed)'
            })

    # Extract per-sample/per-fish demultiplexed reads
    if 'demultiplexed_reads' in data:
        demux_reads = data['demultiplexed_reads']

        # For 6978, also collect per-barcode statistics
        if sample_type == '6978' and 'stats' in data:
            per_sample_stats = data['stats'].get('per_sample_reads', {})
        else:
            per_sample_stats = {}

        for sample_name, reads in demux_reads.items():
            n_reads = len(reads)

            # Determine if this is 7462 or a 6978 fish
            if sample_name == '7462':
                fish_id = 'all'
                sample = '7462'
                fish_display = sample_name
                notes = f'Demultiplexed reads for {sample_name}'
            else:
                # 6978_F1, 6978_F2, etc - extract the barcode
                fish_id = sample_name.replace('6978_', '')
                sample = '6978'
                fish_name = get_fish_display_name(fish_id)
                fish_display = f"{fish_id} ({fish_name})"

                # Get the sample barcode if available
                if fish_id in per_sample_stats:
                    notes = f'Sample barcode: {fish_id}, {fish_name}'
                else:
                    notes = f'Sample barcode: {fish_id}, {fish_name}'

            stats.append({
                'stage': '1_demultiplexed',
                'sample': sample,
                'fish': fish_id,
                'fish_display': fish_display,
                'n_reads': n_reads,
                'n_barcodes': 'N/A',
                'notes': notes
            })

    return stats


def extract_extracted_barcodes_stats(extracted_file):
    """Extract statistics from extracted barcodes."""
    print(f"Processing: {extracted_file}")
    data = load_pickle_safe(extracted_file)
    if data is None:
        return []

    stats = []

    sample_type = data.get('sample_type', 'unknown')

    # Extract per-sample stats (don't add overall summary to avoid duplication)
    if 'sample_stats' in data:
        sample_stats = data['sample_stats']

        for sample_name, sample_stat in sample_stats.items():
            n_reads_processed = sample_stat.get('reads_processed', 0)
            n_successful = sample_stat.get('successful', 0)
            n_failed = sample_stat.get('failed', 0)
            n_unique = sample_stat.get('unique_barcodes', 0)
            success_rate = sample_stat.get('success_rate', 0)

            # Determine sample and fish
            if sample_name == '7462':
                sample = '7462'
                fish_id = 'all'
                fish_display = sample_name
            else:
                sample = '6978'
                fish_id = sample_name.replace('6978_', '')
                fish_display = f"{fish_id} ({get_fish_display_name(fish_id)})"

            stats.append({
                'stage': '2_extracted',
                'sample': sample,
                'fish': fish_id,
                'fish_display': fish_display,
                'n_reads': n_successful,
                'n_barcodes': n_unique,
                'notes': f'{success_rate:.1f}% success ({n_failed} failed)'
            })

    return stats


def extract_collapsed_stats(collapsed_file):
    """Extract statistics from collapsed barcodes."""
    print(f"Processing: {collapsed_file}")
    data = load_pickle_safe(collapsed_file)
    if data is None:
        return []

    stats = []

    if 'collapsed_barcodes' not in data:
        return stats

    collapsed_barcodes = data['collapsed_barcodes']

    # Count parent barcodes
    n_parents = len(collapsed_barcodes)

    # Count total barcodes (parents + children)
    n_total_barcodes = n_parents
    for parent_info in collapsed_barcodes.values():
        n_total_barcodes += len(parent_info.get('children', []))

    # Count reads and unique barcodes per sample/fish
    reads_7462 = 0
    barcodes_7462 = 0
    reads_6978 = {}
    barcodes_6978 = {}

    for parent, info in collapsed_barcodes.items():
        # The collapse script already merged all children counts into the parent
        # So we just use the parent's counts directly (no need to add children again)
        parent_7462 = info.get('7462', 0)
        parent_6978 = info.get('6978', {})

        # Count for 7462
        if parent_7462 > 0:
            reads_7462 += parent_7462
            barcodes_7462 += 1

        # Count for each fish in 6978
        for fish, count in parent_6978.items():
            if count > 0:
                reads_6978[fish] = reads_6978.get(fish, 0) + count
                barcodes_6978[fish] = barcodes_6978.get(fish, 0) + 1

    # Add 7462 stats
    stats.append({
        'stage': '3_collapsed',
        'sample': '7462',
        'fish': 'all',
        'fish_display': '7462',
        'n_reads': reads_7462,
        'n_barcodes': barcodes_7462,
        'notes': f'{barcodes_7462} unique barcodes (collapsed from {n_total_barcodes} total)'
    })

    # Add per-fish stats for 6978
    for fish in sorted(reads_6978.keys()):
        stats.append({
            'stage': '3_collapsed',
            'sample': '6978',
            'fish': fish,
            'fish_display': f"{fish} ({get_fish_display_name(fish)})",
            'n_reads': reads_6978[fish],
            'n_barcodes': barcodes_6978[fish],
            'notes': f'{barcodes_6978[fish]} unique barcodes present in this fish'
        })

    return stats


def extract_filtered_stats(filtered_file):
    """Extract statistics from filtered high-confidence barcodes."""
    print(f"Processing: {filtered_file}")
    data = load_pickle_safe(filtered_file)
    if data is None:
        return []

    stats = []

    if 'barcodes' not in data:
        return stats

    barcodes = data['barcodes']
    sample_type = data.get('sample_type', 'unknown')
    filter_criteria = data.get('filter_criteria', {})

    n_barcodes = len(barcodes)

    if sample_type == '7462':
        # Count total reads
        n_reads = sum(info.get('count', info.get('7462', 0)) for info in barcodes.values())

        # Get filtering stats
        file_stats = data.get('stats', {})
        n_removed_similar = file_stats.get('n_removed_similar', 0)

        notes = f"High-confidence: {filter_criteria.get('description', 'filtered')}"
        if n_removed_similar > 0:
            notes += f" ({n_removed_similar} removed as similar to conserved)"

        stats.append({
            'stage': '4_filtered',
            'sample': '7462',
            'fish': 'all',
            'fish_display': '7462',
            'n_reads': n_reads,
            'n_barcodes': n_barcodes,
            'notes': notes
        })

    elif sample_type == '6978':
        # Per-fish stats
        reads_per_fish = {}
        barcodes_per_fish = {}

        for barcode, info in barcodes.items():
            fish_counts = info.get('6978', {})
            for fish, count in fish_counts.items():
                reads_per_fish[fish] = reads_per_fish.get(fish, 0) + count
                barcodes_per_fish[fish] = barcodes_per_fish.get(fish, 0) + 1

        # Get filtering stats
        file_stats = data.get('stats', {})
        n_removed_similar = file_stats.get('n_removed_similar', 0)

        notes = f"High-confidence: {filter_criteria.get('description', 'filtered')}"
        if n_removed_similar > 0:
            notes += f" ({n_removed_similar} removed as similar to conserved)"

        for fish in sorted(reads_per_fish.keys()):
            stats.append({
                'stage': '4_filtered',
                'sample': '6978',
                'fish': fish,
                'fish_display': f"{fish} ({get_fish_display_name(fish)})",
                'n_reads': reads_per_fish[fish],
                'n_barcodes': barcodes_per_fish[fish],
                'notes': notes
            })

    return stats


def find_latest_files():
    """Find the most recent files for each processing stage."""
    files = {
        'processed_7462': None,
        'demultiplexed_6978': None,
        'extracted_7462': None,
        'extracted_6978': None,
        'collapsed': None,
        'filtered_7462': None,
        'filtered_6978': None
    }

    # Find processed 7462 (raw FASTQ)
    processed_7462_files = glob.glob('processed_7462_*.pkl')
    if processed_7462_files:
        files['processed_7462'] = max(processed_7462_files, key=lambda x: Path(x).stat().st_mtime)

    # Find demultiplexed 6978 (raw FASTQ + demux)
    demux_6978_files = glob.glob('demultiplexed_6978_*.pkl')
    if demux_6978_files:
        files['demultiplexed_6978'] = max(demux_6978_files, key=lambda x: Path(x).stat().st_mtime)

    # Find extracted 7462
    extracted_7462_files = glob.glob('extracted_barcodes_7462_*.pkl')
    if extracted_7462_files:
        files['extracted_7462'] = max(extracted_7462_files, key=lambda x: Path(x).stat().st_mtime)

    # Find extracted 6978
    extracted_6978_files = glob.glob('extracted_barcodes_6978_*.pkl')
    if extracted_6978_files:
        files['extracted_6978'] = max(extracted_6978_files, key=lambda x: Path(x).stat().st_mtime)

    # Find collapsed
    collapsed_files = glob.glob('collapsed_barcodes_collapsed_*.pkl')
    if collapsed_files:
        files['collapsed'] = max(collapsed_files, key=lambda x: Path(x).stat().st_mtime)

    # Find filtered 7462
    filtered_7462_files = glob.glob('high_confidence_barcodes_7462*.pkl')
    if filtered_7462_files:
        files['filtered_7462'] = max(filtered_7462_files, key=lambda x: Path(x).stat().st_mtime)

    # Find filtered 6978
    filtered_6978_files = glob.glob('high_confidence_barcodes_6978*.pkl')
    if filtered_6978_files:
        files['filtered_6978'] = max(filtered_6978_files, key=lambda x: Path(x).stat().st_mtime)

    return files


def generate_report():
    """Generate comprehensive processing report."""
    print("="*80)
    print("GENERATING BARCODE PROCESSING REPORT")
    print("="*80)
    print()

    # Find latest files
    print("Finding latest processing files...")
    files = find_latest_files()

    print("\nFiles found:")
    for stage, filepath in files.items():
        if filepath:
            print(f"  {stage}: {filepath}")
        else:
            print(f"  {stage}: NOT FOUND")
    print()

    # Extract statistics from each stage
    all_stats = []

    if files['processed_7462']:
        all_stats.extend(extract_raw_fastq_stats(files['processed_7462']))

    if files['demultiplexed_6978']:
        all_stats.extend(extract_raw_fastq_stats(files['demultiplexed_6978']))

    if files['extracted_7462']:
        all_stats.extend(extract_extracted_barcodes_stats(files['extracted_7462']))

    if files['extracted_6978']:
        all_stats.extend(extract_extracted_barcodes_stats(files['extracted_6978']))

    if files['collapsed']:
        all_stats.extend(extract_collapsed_stats(files['collapsed']))

    if files['filtered_7462']:
        all_stats.extend(extract_filtered_stats(files['filtered_7462']))

    if files['filtered_6978']:
        all_stats.extend(extract_filtered_stats(files['filtered_6978']))

    if not all_stats:
        print("ERROR: No statistics extracted. Check that .pkl files exist.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_stats)

    # Sort by stage, sample, fish
    stage_order = ['0_raw_fastq', '1_demultiplexed', '2_extracted', '3_collapsed', '4_filtered']
    df['stage_order'] = df['stage'].map({s: i for i, s in enumerate(stage_order)})
    df = df.sort_values(['stage_order', 'sample', 'fish']).drop('stage_order', axis=1)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'barcode_processing_report_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print("REPORT GENERATED")
    print(f"{'='*80}")
    print(f"\nSaved: {output_file}")
    print(f"\nTotal rows: {len(df)}")
    print(f"\nPreview:")
    print(df.to_string(index=False))

    # Generate summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Group by stage and sample
    summary = df.groupby(['stage', 'sample']).agg({
        'n_reads': 'sum',
        'n_barcodes': lambda x: x.iloc[0] if x.iloc[0] != 'N/A' else 'N/A'
    }).reset_index()

    print("\nReads and barcodes by stage and sample:")
    print(summary.to_string(index=False))

    # Calculate retention rates
    print(f"\n{'='*80}")
    print("RETENTION RATES")
    print(f"{'='*80}")

    for sample in ['7462', '6978']:
        print(f"\n{sample}:")

        sample_data = df[df['sample'] == sample]

        # Get reads at each stage
        # For stages with 'all' fish summary row, use that to avoid double-counting
        # For stages without 'all' row, sum across individual fish
        stages = {}

        for stage in ['0_raw_fastq', '1_demultiplexed', '2_extracted', '3_collapsed', '4_filtered']:
            stage_data = sample_data[sample_data['stage'] == stage]

            if not stage_data.empty:
                # Check if there's an 'all' summary row for this stage
                all_row = stage_data[stage_data['fish'] == 'all']

                if not all_row.empty:
                    # Use the 'all' summary to avoid double-counting
                    stages[stage] = all_row['n_reads'].iloc[0]
                else:
                    # No 'all' summary, sum across individual fish
                    fish_rows = stage_data[stage_data['fish'] != 'all']
                    if not fish_rows.empty:
                        stages[stage] = fish_rows['n_reads'].sum()

        # Print retention
        if '0_raw_fastq' in stages:
            baseline = stages['0_raw_fastq']
            for stage in ['1_demultiplexed', '2_extracted', '3_collapsed', '4_filtered']:
                if stage in stages:
                    retention = 100 * stages[stage] / baseline if baseline > 0 else 0
                    print(f"  {stage}: {stages[stage]:,} reads ({retention:.1f}% of raw FASTQ)")

    return output_file


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage:")
        print("  python generate_processing_report.py")
        print()
        print("This script automatically finds the most recent .pkl files from each")
        print("processing stage and generates a comprehensive CSV report showing:")
        print("  - Number of reads at each stage")
        print("  - Number of barcodes at each stage")
        print("  - Per-sample and per-fish breakdown")
        print("  - Retention rates")
        print()
        print("Required files (most recent will be used):")
        print("  - processed_7462_*.pkl (from unified_demultiplex.py)")
        print("  - demultiplexed_6978_*.pkl (from unified_demultiplex.py)")
        print("  - extracted_barcodes_7462_*.pkl (from unified_extract_barcodes.py)")
        print("  - extracted_barcodes_6978_*.pkl (from unified_extract_barcodes.py)")
        print("  - collapsed_barcodes_collapsed_*.pkl (from collapse_barcodes.py)")
        print("  - high_confidence_barcodes_7462*.pkl (from filter_high_confidence_barcodes_7462.py)")
        print("  - high_confidence_barcodes_6978*.pkl (from filter_high_confidence_barcodes_6978.py)")
        return

    generate_report()


if __name__ == "__main__":
    main()
