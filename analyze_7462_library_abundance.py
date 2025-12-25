#!/usr/bin/env python3
"""
Analyze 7462 library barcode abundance distribution.

Shows:
- Most overrepresented barcodes (after collapsing)
- Abundance percentages
- Distribution statistics (mean, median per quantile)
- Quantile breakdowns

Usage:
    python analyze_7462_library_abundance.py <collapsed_barcodes.pkl>
"""

import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("whitegrid")


def analyze_7462_abundance(collapsed_file, top_n=50):
    """
    Analyze 7462 library abundance distribution.

    Args:
        collapsed_file: Path to collapsed barcode pickle file
        top_n: Number of top barcodes to display
    """
    print("="*80)
    print("ANALYZING 7462 LIBRARY ABUNDANCE")
    print("="*80)
    print()

    print(f"Loading: {collapsed_file}")
    with open(collapsed_file, 'rb') as f:
        data = pickle.load(f)

    collapsed_barcodes = data['collapsed_barcodes']
    print(f"Total parent barcodes: {len(collapsed_barcodes):,}")
    print()

    # Merge children counts into parents
    barcode_counts = {}

    for parent, info in collapsed_barcodes.items():
        # Start with parent count in 7462
        merged_7462 = info.get('7462', 0)

        # Add children counts
        for child in info.get('children', []):
            child_info = info.get('children_counts', {}).get(child, {})
            merged_7462 += child_info.get('7462', 0)

        if merged_7462 > 0:
            barcode_counts[parent] = merged_7462

    print(f"Barcodes with reads in 7462: {len(barcode_counts):,}")

    # Calculate total reads
    total_reads = sum(barcode_counts.values())
    print(f"Total reads in 7462: {total_reads:,}")
    print()

    # Calculate abundance percentages
    barcode_abundance = {
        bc: {'count': count, 'percent': 100 * count / total_reads}
        for bc, count in barcode_counts.items()
    }

    # Sort by count
    sorted_barcodes = sorted(barcode_abundance.items(), key=lambda x: x[1]['count'], reverse=True)

    # Display top N barcodes
    print("="*80)
    print(f"TOP {top_n} MOST ABUNDANT BARCODES")
    print("="*80)
    print()
    print(f"{'Rank':<6} {'Barcode':<25} {'Count':>15} {'% of Library':>15} {'Cumulative %':>15}")
    print("-"*80)

    cumulative_percent = 0
    for rank, (barcode, stats) in enumerate(sorted_barcodes[:top_n], 1):
        cumulative_percent += stats['percent']
        print(f"{rank:<6} {barcode:<25} {stats['count']:>15,} {stats['percent']:>14.3f}% {cumulative_percent:>14.2f}%")

    # Calculate quantile statistics
    print()
    print("="*80)
    print("ABUNDANCE DISTRIBUTION STATISTICS")
    print("="*80)
    print()

    counts = np.array(list(barcode_counts.values()))
    percentages = 100 * counts / total_reads

    print(f"Overall statistics:")
    print(f"  Mean count: {np.mean(counts):,.2f}")
    print(f"  Median count: {np.median(counts):,.2f}")
    print(f"  Std dev: {np.std(counts):,.2f}")
    print(f"  Min count: {np.min(counts):,}")
    print(f"  Max count: {np.max(counts):,}")
    print()
    print(f"  Mean abundance: {np.mean(percentages):.6f}%")
    print(f"  Median abundance: {np.median(percentages):.6f}%")
    print()

    # Quantile analysis
    quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

    print("Quantile breakdown:")
    print(f"  {'Quantile':<12} {'Count':>15} {'% of Library':>15}")
    print("  " + "-"*42)

    for q in quantiles:
        q_count = np.quantile(counts, q)
        q_percent = 100 * q_count / total_reads
        print(f"  {f'{int(q*100)}%':<12} {q_count:>15,.0f} {q_percent:>14.6f}%")

    # Calculate statistics per quantile range
    print()
    print("Statistics per quantile range:")
    print(f"  {'Range':<20} {'N Barcodes':>12} {'Mean':>12} {'Median':>12} {'Total Reads':>15} {'% of Total':>12}")
    print("  " + "-"*85)

    quantile_ranges = [
        ("Bottom 10%", 0.0, 0.1),
        ("10-25%", 0.1, 0.25),
        ("25-50%", 0.25, 0.5),
        ("50-75%", 0.5, 0.75),
        ("75-90%", 0.75, 0.9),
        ("90-95%", 0.9, 0.95),
        ("95-99%", 0.95, 0.99),
        ("Top 1%", 0.99, 1.0),
    ]

    for label, q_low, q_high in quantile_ranges:
        low_val = np.quantile(counts, q_low)
        high_val = np.quantile(counts, q_high)

        # Get barcodes in this range
        range_counts = counts[(counts >= low_val) & (counts <= high_val)]

        if len(range_counts) > 0:
            range_mean = np.mean(range_counts)
            range_median = np.median(range_counts)
            range_total = np.sum(range_counts)
            range_percent = 100 * range_total / total_reads

            print(f"  {label:<20} {len(range_counts):>12,} {range_mean:>12,.1f} "
                  f"{range_median:>12,.1f} {range_total:>15,} {range_percent:>11.2f}%")

    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Top N barcodes
    fig, ax = plt.subplots(figsize=(14, 8))

    top_data = sorted_barcodes[:min(top_n, len(sorted_barcodes))]
    barcodes_plot = [bc[:15] + '...' if len(bc) > 15 else bc for bc, _ in top_data]
    counts_plot = [stats['count'] for _, stats in top_data]

    x = np.arange(len(barcodes_plot))
    bars = ax.bar(x, counts_plot, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color code: top 10 in red, rest in blue
    for i, bar in enumerate(bars):
        if i < 10:
            bar.set_color('red')
        else:
            bar.set_color('steelblue')

    ax.set_xlabel('Barcode', fontsize=12)
    ax.set_ylabel('Read Count', fontsize=12)
    ax.set_title(f'Top {len(top_data)} Most Abundant Barcodes in 7462 Library\n'
                 f'Total: {total_reads:,} reads across {len(barcode_counts):,} unique barcodes',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x[::2])  # Show every other label to avoid crowding
    ax.set_xticklabels(barcodes_plot[::2], rotation=90, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f'7462_top_barcodes_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")
    plt.close()

    # Plot 2: Distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    ax = axes[0]
    ax.hist(counts, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):,.0f}')
    ax.axvline(np.median(counts), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(counts):,.0f}')
    ax.set_xlabel('Read Count per Barcode', fontsize=11)
    ax.set_ylabel('Number of Barcodes', fontsize=11)
    ax.set_title('7462 Abundance Distribution (Linear Scale)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Log scale
    ax = axes[1]
    ax.hist(counts, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):,.0f}')
    ax.axvline(np.median(counts), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(counts):,.0f}')
    ax.set_xlabel('Read Count per Barcode', fontsize=11)
    ax.set_ylabel('Number of Barcodes', fontsize=11)
    ax.set_title('7462 Abundance Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f'7462_distribution_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Plot 3: Cumulative distribution
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_counts = np.sort(counts)[::-1]  # Sort descending
    cumsum = np.cumsum(sorted_counts)
    cumsum_percent = 100 * cumsum / total_reads

    ax.plot(range(len(cumsum_percent)), cumsum_percent, linewidth=2, color='steelblue')

    # Add reference lines
    for pct in [50, 80, 90, 95, 99]:
        idx = np.argmax(cumsum_percent >= pct)
        ax.axhline(pct, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(idx, pct + 2, f'{idx:,} barcodes = {pct}%', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Number of Barcodes (ranked by abundance)', fontsize=11)
    ax.set_ylabel('Cumulative % of Total Reads', fontsize=11)
    ax.set_title('Cumulative Distribution: How Many Barcodes Account for X% of Reads?',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, len(cumsum_percent))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f'7462_cumulative_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Save detailed results to CSV
    df = pd.DataFrame([
        {
            'rank': i+1,
            'barcode': bc,
            'count': stats['count'],
            'percent': stats['percent']
        }
        for i, (bc, stats) in enumerate(sorted_barcodes)
    ])

    csv_file = f'7462_abundance_all_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("ERROR: Missing required argument")
        print()
        print("Usage:")
        print("  python analyze_7462_library_abundance.py <collapsed_barcodes.pkl> [--top N]")
        print()
        print("Arguments:")
        print("  collapsed_barcodes.pkl    Collapsed barcode file")
        print("  --top N                   Number of top barcodes to display (default: 50)")
        print()
        print("Example:")
        print("  python analyze_7462_library_abundance.py collapsed_barcodes_collapsed_*.pkl")
        print("  python analyze_7462_library_abundance.py collapsed_barcodes_collapsed_*.pkl --top 100")
        return

    collapsed_file = sys.argv[1]
    top_n = 50

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--top' and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])

    analyze_7462_abundance(collapsed_file, top_n=top_n)


if __name__ == "__main__":
    main()
