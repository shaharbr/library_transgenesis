#!/usr/bin/env python3
"""
Filter for high-confidence barcodes in 7462 library data.

Filters barcodes that appear >= 2 times in the 7462 library.
Uses only parent barcodes with children counts merged into parents.
This identifies barcodes that were present in the injected library.

Usage:
    python filter_high_confidence_barcodes_7462.py <collapsed_barcodes.pkl>
"""

import pickle
import sys
import re
from Levenshtein import distance as levenshtein_distance


# Template sequence for 7462 with N's marking the random barcode position
TEMPLATE_7462 = 'acgggGACAGCCCCCTCCCAAAGCCCCCAGGGANNNNNNNNNNNNNNNCACGCTAGCTGTAATTACGTCCCTCCCCCGCTAGGGGGCAGCAGCGAGCCGCCCGGGGCTCCGCTCCGGTCCGGCGCTCCCCCCGCATCCCCGAGCCGGCAGCGTGCGGGGACAGCCCGGGCACGGGGAAGGTGGCACGGGATCGCTTTCCTCTGAACGCTTCTCGCTGCTCTTTGAGCCTGCAGACACCTGGGGGGATACGGGGAAAAAGCTTTAGGCTG'


def extract_constant_regions(template):
    """Extract all constant (non-N) regions from template."""
    constant_regions = []
    for match in re.finditer(r'[^Nn]+', template):
        region = match.group()
        if len(region) >= 14:  # Only consider regions long enough
            constant_regions.append(region.upper())
    return constant_regions


def generate_sliding_windows(sequence, window_sizes=[14, 15, 16]):
    """Generate all sliding windows of given sizes from a sequence."""
    windows = set()
    for window_size in window_sizes:
        if len(sequence) < window_size:
            continue
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            windows.add(window)
    return windows


def is_barcode_similar_to_constant(barcode, constant_windows, max_distance=2):
    """Check if barcode is too similar to any constant region window."""
    barcode_upper = barcode.upper()
    for window in constant_windows:
        if len(window) == len(barcode_upper):
            dist = levenshtein_distance(barcode_upper, window)
            if dist <= max_distance:
                return True, window, dist
    return False, None, None


def filter_high_confidence_barcodes_7462(collapsed_file, min_count_7462=2, filter_similar=True, max_distance=2):
    """
    Filter for high-confidence barcodes based on 7462 counts.

    Args:
        collapsed_file: Path to collapsed barcode pickle file
        min_count_7462: Minimum total count in 7462 data (default: 2, meaning >= 2)
        filter_similar: Whether to filter barcodes similar to conserved regions (default: True)
        max_distance: Maximum Levenshtein distance for similarity filtering (default: 2)

    Returns:
        Dictionary of high-confidence barcodes with same structure as input
    """
    print("="*80)
    print("FILTERING HIGH-CONFIDENCE BARCODES (7462 LIBRARY)")
    print("="*80)
    print()
    print(f"Filters:")
    print(f"  - Minimum count: >= {min_count_7462}")
    print(f"  - Filter similar to conserved: {filter_similar}")
    if filter_similar:
        print(f"  - Max Levenshtein distance: {max_distance}")
    print()

    print(f"Loading: {collapsed_file}")
    with open(collapsed_file, 'rb') as f:
        data = pickle.load(f)

    collapsed_barcodes = data['collapsed_barcodes']

    print(f"\nOriginal number of parent barcodes: {len(collapsed_barcodes):,}")

    # Collect parent barcodes with merged counts (parent + all children)
    all_barcodes = {}

    for parent, info in collapsed_barcodes.items():
        # The collapse script already merged all children counts into the parent
        # So we just use the parent's counts directly (no need to add children again)
        parent_7462 = info['7462']

        # For 7462 filter, only keep 7462 counts (remove 6978 data)
        all_barcodes[parent] = {
            'count': parent_7462,  # Single count field for consistency
            '7462': parent_7462,   # Also keep in '7462' for backwards compatibility
            'n_children': len(info['children'])
        }

    print(f"Total parent barcodes (with merged children counts): {len(all_barcodes):,}")

    # Step 1: Filter for high confidence by count (>= min_count_7462 reads in 7462)
    print(f"\nStep 1: Filtering by count (>= {min_count_7462})...")
    high_confidence = {
        bc: info for bc, info in all_barcodes.items()
        if info['7462'] >= min_count_7462
    }

    print(f"  After count filter: {len(high_confidence):,} barcodes")
    print(f"  Filtered out: {len(all_barcodes) - len(high_confidence):,}")

    # Step 2: Filter barcodes similar to conserved regions
    removed_similar = {}
    if filter_similar:
        print(f"\nStep 2: Filtering barcodes similar to conserved regions...")
        print(f"  Extracting constant regions from template...")

        constant_regions = extract_constant_regions(TEMPLATE_7462)
        print(f"  Constant regions found: {len(constant_regions)}")

        print(f"  Generating sliding windows (14bp, 15bp, 16bp)...")
        constant_windows = set()
        for region in constant_regions:
            windows = generate_sliding_windows(region)
            constant_windows.update(windows)
        print(f"  Total constant windows: {len(constant_windows):,}")

        print(f"  Checking similarity (Levenshtein distance <= {max_distance})...")
        filtered_after_similarity = {}

        for barcode, info in high_confidence.items():
            is_similar, similar_window, dist = is_barcode_similar_to_constant(
                barcode, constant_windows, max_distance
            )

            if is_similar:
                removed_similar[barcode] = {
                    'info': info,
                    'similar_to': similar_window,
                    'distance': dist
                }
            else:
                filtered_after_similarity[barcode] = info

        high_confidence = filtered_after_similarity

        print(f"  After similarity filter: {len(high_confidence):,} barcodes")
        print(f"  Removed as similar to conserved: {len(removed_similar):,}")

        if removed_similar:
            sorted_removed = sorted(removed_similar.items(),
                                   key=lambda x: x[1]['info']['7462'],
                                   reverse=True)
            print(f"\n  Top 10 removed barcodes (by count):")
            print(f"    {'Barcode':<20} {'Count':>10} {'Distance':>10} {'Similar to':<20}")
            print(f"    {'-'*70}")
            for barcode, data in sorted_removed[:10]:
                print(f"    {barcode:<20} {data['info']['7462']:>10,} "
                      f"{data['distance']:>10} {data['similar_to'][:20]:<20}")

    # Calculate statistics
    total_7462_counts_all = sum(info['7462'] for info in all_barcodes.values())
    total_7462_counts_filtered = sum(info['7462'] for info in high_confidence.values())

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal 7462 reads:")
    print(f"  All barcodes: {total_7462_counts_all:,}")
    print(f"  High-confidence: {total_7462_counts_filtered:,}")
    print(f"  Retained: {100 * total_7462_counts_filtered / total_7462_counts_all:.1f}%")

    # Save filtered barcodes
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'high_confidence_barcodes_7462gte{min_count_7462}_{timestamp}.pkl'

    output_data = {
        'barcodes': high_confidence,
        'sample_type': '7462',  # Explicitly mark this as 7462 data
        'filter_criteria': {
            'min_count_7462': min_count_7462,
            'filter_similar': filter_similar,
            'max_distance': max_distance if filter_similar else None,
            'description': f'7462_count >= {min_count_7462}' + (f', no similarity to conserved (dist<={max_distance})' if filter_similar else '')
        },
        'stats': {
            'n_total': len(all_barcodes),
            'n_filtered': len(high_confidence),
            'n_removed': len(all_barcodes) - len(high_confidence),
            'n_removed_similar': len(removed_similar) if filter_similar else 0,
            'total_reads_all': total_7462_counts_all,
            'total_reads_filtered': total_7462_counts_filtered
        },
        'removed_similar': removed_similar if filter_similar else {}
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nSaved: {output_file}")

    return high_confidence, output_file


def main():
    if len(sys.argv) < 2:
        print("ERROR: Missing required argument")
        print()
        print("Usage:")
        print("  python filter_high_confidence_barcodes_7462.py <collapsed_barcodes.pkl> [OPTIONS]")
        print()
        print("Arguments:")
        print("  collapsed_barcodes.pkl    Collapsed barcode file")
        print()
        print("Options:")
        print("  --min-count N            Minimum total count in 7462 (default: 2)")
        print("  --no-filter-similar      Disable filtering of barcodes similar to conserved regions")
        print("  --max-distance N         Maximum Levenshtein distance for similarity (default: 2)")
        print()
        print("Examples:")
        print("  python filter_high_confidence_barcodes_7462.py collapsed_barcodes_collapsed_*.pkl")
        print("  python filter_high_confidence_barcodes_7462.py collapsed_barcodes_collapsed_*.pkl --min-count 5")
        print("  python filter_high_confidence_barcodes_7462.py collapsed_barcodes_collapsed_*.pkl --no-filter-similar")
        return

    collapsed_file = sys.argv[1]
    min_count = 2
    filter_similar = True
    max_distance = 2

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--min-count' and i + 1 < len(sys.argv):
            min_count = int(sys.argv[i + 1])
        elif arg == '--no-filter-similar':
            filter_similar = False
        elif arg == '--max-distance' and i + 1 < len(sys.argv):
            max_distance = int(sys.argv[i + 1])

    filter_high_confidence_barcodes_7462(collapsed_file,
                                         min_count_7462=min_count,
                                         filter_similar=filter_similar,
                                         max_distance=max_distance)

    print("\n" + "="*80)
    print("Filtering complete!")
    print("="*80)


if __name__ == "__main__":
    main()
