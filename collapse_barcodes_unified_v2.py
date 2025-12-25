#!/usr/bin/env python3
"""
Collapse barcodes from both 6978 and 7462 samples using Levenshtein distance ≤ 1.

For each barcode, find all neighbors within Levenshtein distance 1 and collapse
less abundant barcodes into the most abundant one (parent).

Tracks:
- Parent-child relationships
- Abundance in 7462 library
- Abundance in each of the 12 fish from 6978
- Collapsing statistics and visualizations

Uses optimized functions from combine_6978_and_7462.py

Usage:
    python collapse_barcodes_unified_v2.py <7462_barcodes.pkl> <6978_barcodes.pkl>
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime

sns.set_style("whitegrid")


# ============================================================================
# OPTIMIZED DISTANCE AND NEIGHBOR FUNCTIONS (from combine_6978_and_7462.py)
# ============================================================================

def hamming_distance(s1, s2, max_dist=1):
    """Calculate Hamming distance with early termination."""
    if len(s1) != len(s2):
        return max_dist + 1

    dist = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            dist += 1
            if dist > max_dist:
                return max_dist + 1
    return dist


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein (edit) distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def generate_neighbors_levenshtein_1(barcode):
    """
    Generate all possible barcodes at Levenshtein distance = 1.
    Optimized version with pre-allocation and minimal string operations.
    """
    bases = ('A', 'C', 'G', 'T')
    bc_len = len(barcode)

    neighbors = []
    neighbors_append = neighbors.append

    # Convert to list for faster indexing
    bc_list = list(barcode)

    # 1. Substitutions
    for pos in range(bc_len):
        original = bc_list[pos]
        for new_base in bases:
            if new_base != original:
                bc_list[pos] = new_base
                neighbors_append(''.join(bc_list))
        bc_list[pos] = original  # Restore

    # 2. Deletions
    for pos in range(bc_len):
        neighbors_append(barcode[:pos] + barcode[pos+1:])

    # 3. Insertions
    for pos in range(bc_len + 1):
        prefix = barcode[:pos]
        suffix = barcode[pos:]
        for new_base in bases:
            neighbors_append(prefix + new_base + suffix)

    return neighbors


def validate_merges(merge_map, max_distance=1):
    """Validate that all merges are within the specified Levenshtein distance"""
    print("\nValidating merges...")

    distance_violations = []
    distance_counts = defaultdict(int)

    for child, parent in merge_map.items():
        dist = levenshtein_distance(child, parent)
        distance_counts[dist] += 1

        if dist > max_distance:
            distance_violations.append((child, parent, dist))

    print(f"  Total merges checked: {len(merge_map):,}")
    print(f"  Distance distribution:")

    for dist in sorted(distance_counts.keys()):
        count = distance_counts[dist]
        pct = count / len(merge_map) * 100 if merge_map else 0
        print(f"    Distance {dist}: {count:,} ({pct:.2f}%)")

    if distance_violations:
        print(f"\n  WARNING: {len(distance_violations)} merges with distance > {max_distance}")
        print(f"  First 5 violations:")
        for child, parent, dist in distance_violations[:5]:
            print(f"    {child} -> {parent} (distance={dist})")
        return False, distance_violations
    else:
        print(f"  [OK] All merges are <= distance={max_distance}")
        return True, []


# ============================================================================
# LOADING AND COLLAPSING FUNCTIONS
# ============================================================================

def load_barcode_data(barcode_file, sample_type):
    """Load barcode data and extract counts per sample."""
    print(f"Loading {sample_type}: {barcode_file}")
    with open(barcode_file, 'rb') as f:
        data = pickle.load(f)

    # Extract barcode counts by sample
    if 'sample_barcodes' in data:
        sample_barcodes = data['sample_barcodes']
    else:
        # If no sample breakdown, use overall counts
        if 'overall_barcode_counts' in data:
            sample_barcodes = {sample_type: data['overall_barcode_counts']}
        else:
            print(f"ERROR: Could not find barcode counts in {barcode_file}")
            return None

    print(f"  Samples: {list(sample_barcodes.keys())}")
    for sample, barcodes in sample_barcodes.items():
        print(f"    {sample}: {len(barcodes):,} unique barcodes, {sum(barcodes.values()):,} reads")

    return sample_barcodes


def collapse_barcodes(barcode_7462, barcode_6978):
    """
    Collapse barcodes using Levenshtein distance ≤ 1.
    Uses optimized neighbor generation approach.
    """
    print("\n" + "="*80)
    print("COLLAPSING BARCODES (Levenshtein distance ≤ 1)")
    print("="*80)
    print()

    # Combine all barcodes with their sample-specific counts
    all_barcodes = defaultdict(lambda: {'7462': 0, '6978': defaultdict(int), 'total': 0})

    # Add 7462 barcodes
    for sample, barcodes in barcode_7462.items():
        for bc, count in barcodes.items():
            all_barcodes[bc]['7462'] += count
            all_barcodes[bc]['total'] += count

    # Add 6978 barcodes
    for sample, barcodes in barcode_6978.items():
        for bc, count in barcodes.items():
            all_barcodes[bc]['6978'][sample] += count
            all_barcodes[bc]['total'] += count

    print(f"Total unique barcodes: {len(all_barcodes):,}")
    total_reads = sum(bc['total'] for bc in all_barcodes.values())
    print(f"Total reads: {total_reads:,}")

    # Sort barcodes by abundance (for parent selection)
    sorted_barcodes = sorted(all_barcodes.items(), key=lambda x: x[1]['total'], reverse=True)

    # Show length distribution
    length_counts = defaultdict(int)
    for bc, _ in sorted_barcodes:
        length_counts[len(bc)] += 1

    print(f"\nBarcode length distribution:")
    for length in sorted(length_counts.keys()):
        print(f"  {length}bp: {length_counts[length]:,} barcodes")

    print(f"\nStarting collapse process...")
    print(f"  Method: Neighbor generation (Levenshtein distance = 1)")
    print(f"  Generates ~68 neighbors per barcode (3*L substitutions + L deletions + 4*(L+1) insertions)")

    # Data structures for collapsing
    merge_map = {}  # child -> parent
    merged_into = set()
    barcode_set = set(all_barcodes.keys())
    barcode_lookup = {bc: info['total'] for bc, info in all_barcodes.items()}

    n_merged = 0
    n_processed = 0
    n_skipped = 0
    last_progress_pct = -1
    last_merge_count = 0

    # Process each barcode (most to least abundant)
    for i, (current_bc, current_info) in enumerate(sorted_barcodes):
        if current_bc in merged_into:
            n_skipped += 1
            continue

        n_processed += 1

        # Show progress every 100k checked barcodes
        if (i + 1) % 100000 == 0:
            current_pct = (i + 1) / len(sorted_barcodes) * 100
            merges_since_last = n_merged - last_merge_count
            remaining = len(sorted_barcodes) - n_processed - n_merged
            print(f"  Checked {i+1:,}/{len(sorted_barcodes):,} ({current_pct:.1f}%) | "
                  f"Processed: {n_processed:,} | Merged: {n_merged:,} (+{merges_since_last:,}) | "
                  f"Skipped: {n_skipped:,} | Remaining: {remaining:,}", flush=True)
            last_merge_count = n_merged

        # Generate neighbors at distance 1
        neighbors = generate_neighbors_levenshtein_1(current_bc)

        # Filter to valid neighbors that exist and haven't been merged
        valid_neighbors = (set(neighbors) & barcode_set) - merged_into

        # Merge less abundant neighbors
        for neighbor in valid_neighbors:
            neighbor_count = barcode_lookup[neighbor]

            # Merge if current is more abundant (or equal, with tie-breaking by lexicographic order)
            if current_info['total'] > neighbor_count or \
               (current_info['total'] == neighbor_count and current_bc < neighbor):
                merge_map[neighbor] = current_bc
                merged_into.add(neighbor)
                barcode_set.discard(neighbor)
                n_merged += 1

    print(f"\n  Final stats:")
    print(f"    Total checked: {len(sorted_barcodes):,}")
    print(f"    Processed (parents): {n_processed:,}")
    print(f"    Merged (children): {n_merged:,}")
    print(f"    Skipped (already merged): {n_skipped:,}")
    print(f"    Remaining (unique parents): {len(barcode_set):,}")
    print(f"    Merge rate: {n_merged/len(sorted_barcodes)*100:.1f}%")

    # Calculate cross-length merges
    cross_length = sum(1 for child, parent in merge_map.items() if len(child) != len(parent))
    print(f"    Cross-length merges (indels): {cross_length:,} ({cross_length/max(n_merged,1)*100:.1f}%)")

    # Validate merges
    validate_merges(merge_map, max_distance=1)

    # Build collapsed barcode structure
    print(f"\nBuilding collapsed barcode structure...")

    collapsed_barcodes = defaultdict(lambda: {
        '7462': 0,
        '6978': defaultdict(int),
        'total': 0,
        'children': [],
        'children_counts': {}
    })

    for barcode, info in all_barcodes.items():
        # Find parent
        parent = merge_map.get(barcode, barcode)

        # Add counts to parent
        collapsed_barcodes[parent]['7462'] += info['7462']

        for sample, count in info['6978'].items():
            collapsed_barcodes[parent]['6978'][sample] += count

        collapsed_barcodes[parent]['total'] += info['total']

        # Track children
        if barcode != parent:
            collapsed_barcodes[parent]['children'].append(barcode)
            collapsed_barcodes[parent]['children_counts'][barcode] = info

    # Statistics
    print(f"\n{'='*80}")
    print("COLLAPSING STATISTICS")
    print(f"{'='*80}")

    n_parents = len(collapsed_barcodes)
    n_children = len(merge_map)

    print(f"\nOriginal unique barcodes: {len(all_barcodes):,}")
    print(f"Parent barcodes (retained): {n_parents:,}")
    print(f"Child barcodes (collapsed): {n_children:,}")
    print(f"Reduction: {len(all_barcodes) - n_parents:,} barcodes ({100*(len(all_barcodes) - n_parents)/len(all_barcodes):.2f}%)")

    # Children per parent stats
    children_per_parent = [len(info['children']) for info in collapsed_barcodes.values()]

    parents_with_children = sum(1 for c in children_per_parent if c > 0)
    parents_without_children = n_parents - parents_with_children

    print(f"\nParent barcode breakdown:")
    print(f"  Parents with children: {parents_with_children:,}")
    print(f"  Parents without children: {parents_without_children:,}")

    if children_per_parent:
        print(f"\nChildren per parent statistics:")
        print(f"  Mean: {np.mean(children_per_parent):.2f}")
        print(f"  Median: {np.median(children_per_parent):.0f}")
        print(f"  Max: {np.max(children_per_parent)}")
        print(f"  Total children: {sum(children_per_parent):,}")

    # Top parents by number of children
    top_parents = sorted(collapsed_barcodes.items(),
                        key=lambda x: len(x[1]['children']), reverse=True)[:10]

    print(f"\nTop 10 parents by number of children:")
    print(f"  {'Parent':<20} {'Children':>10} {'Total Count':>15}")
    print(f"  {'-'*20} {'-'*10} {'-'*15}")

    for parent, info in top_parents:
        n_children_bc = len(info['children'])
        if n_children_bc > 0:
            print(f"  {parent:<20} {n_children_bc:>10} {info['total']:>15,}")

    # Read statistics
    total_reads_collapsed = sum(bc['total'] for bc in collapsed_barcodes.values())
    print(f"\nRead counts:")
    print(f"  Original: {total_reads:,}")
    print(f"  After collapsing: {total_reads_collapsed:,}")
    print(f"  Verification: {'PASS' if total_reads == total_reads_collapsed else 'FAIL'}")

    return collapsed_barcodes, all_barcodes, merge_map


def save_results(collapsed_barcodes, all_barcodes, merge_map,
                barcode_7462, barcode_6978, output_prefix):
    """Save collapsed barcode results."""

    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save collapsed barcodes with full information
    output_file = f"{output_prefix}_collapsed_{timestamp}.pkl"

    output_data = {
        'collapsed_barcodes': dict(collapsed_barcodes),
        'parent_map': merge_map,
        'original_barcodes': dict(all_barcodes),
        'stats': {
            'n_original': len(all_barcodes),
            'n_collapsed': len(collapsed_barcodes),
            'n_children': len(merge_map),
            'reduction_pct': 100 * (len(all_barcodes) - len(collapsed_barcodes)) / len(all_barcodes),
            'levenshtein_distance': 1
        },
        'timestamp': timestamp,
        'source_7462_samples': list(barcode_7462.keys()),
        'source_6978_samples': list(barcode_6978.keys())
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n  Saved: {output_file}")

    # 2. Save a CSV with collapsed barcode information
    csv_file = f"{output_prefix}_collapsed_{timestamp}.csv"

    with open(csv_file, 'w') as f:
        # Header
        fish_samples = sorted(set(sample for bc in collapsed_barcodes.values()
                                  for sample in bc['6978'].keys()))

        header = ['parent_barcode', 'total_count', '7462_count', 'n_children']
        header.extend([f'6978_{sample}' for sample in fish_samples])
        header.append('children_list')

        f.write(','.join(header) + '\n')

        # Data rows
        for parent, info in sorted(collapsed_barcodes.items(),
                                   key=lambda x: x[1]['total'], reverse=True):
            row = [
                parent,
                str(info['total']),
                str(info['7462']),
                str(len(info['children']))
            ]

            for sample in fish_samples:
                row.append(str(info['6978'].get(sample, 0)))

            row.append('|'.join(info['children']) if info['children'] else 'none')

            f.write(','.join(row) + '\n')

    print(f"  Saved: {csv_file}")

    return output_file, csv_file


def plot_collapsing_results(collapsed_barcodes, all_barcodes, output_prefix):
    """Generate plots for collapsing analysis."""

    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Distribution of children per parent
    ax = axes[0, 0]
    children_counts = [len(info['children']) for info in collapsed_barcodes.values()]
    children_counts_nonzero = [c for c in children_counts if c > 0]

    if children_counts_nonzero:
        bins = np.logspace(0, np.log10(max(children_counts_nonzero) + 1), 30)
        ax.hist(children_counts_nonzero, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Number of Children', fontsize=11)
        ax.set_ylabel('Number of Parent Barcodes', fontsize=11)
        ax.set_title('Children per Parent Distribution\n(Parents with ≥1 child)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 2: Barcode count before vs after
    ax = axes[0, 1]
    categories = ['Original', 'After Collapsing']
    counts = [len(all_barcodes), len(collapsed_barcodes)]
    colors = ['red', 'green']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Unique Barcodes', fontsize=11)
    ax.set_title('Barcode Count Reduction', fontsize=12, fontweight='bold')

    # Add counts on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count:,}', ha='center', va='bottom', fontsize=10)

    # Add reduction percentage
    reduction = 100 * (counts[0] - counts[1]) / counts[0]
    ax.text(0.5, 0.95, f'Reduction: {reduction:.1f}%',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Parent abundance distribution
    ax = axes[0, 2]
    parent_abundances = [info['total'] for info in collapsed_barcodes.values()]

    bins = np.logspace(0, np.log10(max(parent_abundances) + 1), 50)
    ax.hist(parent_abundances, bins=bins, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.set_xscale('log')
    ax.set_xlabel('Total Count', fontsize=11)
    ax.set_ylabel('Number of Parent Barcodes', fontsize=11)
    ax.set_title('Parent Barcode Abundance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative reads in parents
    ax = axes[1, 0]
    sorted_abundances = sorted(parent_abundances, reverse=True)
    cumsum = np.cumsum(sorted_abundances)
    cumsum_pct = 100 * cumsum / cumsum[-1]

    ax.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, color='navy', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Top Parent Barcodes', fontsize=11)
    ax.set_ylabel('Cumulative % of Total Reads', fontsize=11)
    ax.set_title('Read Concentration in Top Parents', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    for pct in [50, 90, 99]:
        idx = np.searchsorted(cumsum_pct, pct)
        if idx < len(cumsum_pct):
            ax.axhline(pct, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(len(cumsum_pct) * 0.7, pct + 2,
                   f'{pct}% at {idx:,} parents', fontsize=9)

    # Plot 5: Parents with vs without children
    ax = axes[1, 1]
    parents_with = sum(1 for c in children_counts if c > 0)
    parents_without = len(children_counts) - parents_with

    categories = ['With Children', 'Without Children']
    counts_pie = [parents_with, parents_without]
    colors_pie = ['orange', 'lightblue']

    wedges, texts, autotexts = ax.pie(counts_pie, labels=categories, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90)
    ax.set_title('Parent Barcodes by Children Status', fontsize=12, fontweight='bold')

    # Plot 6: Top 20 parents by children count
    ax = axes[1, 2]
    top_parents = sorted(collapsed_barcodes.items(),
                        key=lambda x: len(x[1]['children']), reverse=True)[:20]

    if top_parents:
        y_pos = np.arange(len(top_parents))
        children_counts_top = [len(info['children']) for _, info in top_parents]

        ax.barh(y_pos, children_counts_top, color='purple', alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{bc[:8]}..." for bc, _ in top_parents], fontsize=8)
        ax.set_xlabel('Number of Children', fontsize=11)
        ax.set_title('Top 20 Parents by Children Count', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_file = f'{output_prefix}_collapsing_analysis_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("ERROR: Missing required arguments")
        print()
        print("Usage:")
        print("  python collapse_barcodes_unified_v2.py <7462_barcodes.pkl> <6978_barcodes.pkl> [--output PREFIX]")
        print()
        print("Arguments:")
        print("  7462_barcodes.pkl    Extracted barcodes from 7462 library")
        print("  6978_barcodes.pkl    Extracted barcodes from 6978 fish samples")
        print("  --output PREFIX      Output file prefix (default: collapsed_barcodes)")
        print()
        print("Example:")
        print("  python collapse_barcodes_unified_v2.py extracted_barcodes_7462_*.pkl extracted_barcodes_6978_*.pkl")
        return

    file_7462 = sys.argv[1]
    file_6978 = sys.argv[2]

    output_prefix = "collapsed_barcodes"
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == '--output' and i + 1 < len(sys.argv):
            output_prefix = sys.argv[i + 1]

    print("="*80)
    print("UNIFIED BARCODE COLLAPSING (Levenshtein distance ≤ 1)")
    print("="*80)
    print()

    # Load data
    barcode_7462 = load_barcode_data(file_7462, '7462')
    barcode_6978 = load_barcode_data(file_6978, '6978')

    if barcode_7462 is None or barcode_6978 is None:
        print("ERROR: Failed to load barcode data")
        return

    # Collapse barcodes
    collapsed_barcodes, all_barcodes, merge_map = collapse_barcodes(
        barcode_7462, barcode_6978
    )

    # Save results
    save_results(collapsed_barcodes, all_barcodes, merge_map,
                barcode_7462, barcode_6978, output_prefix)

    # Generate plots
    plot_collapsing_results(collapsed_barcodes, all_barcodes, output_prefix)

    print("\nDone!")


if __name__ == "__main__":
    main()
