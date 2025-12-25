#!/usr/bin/env python3
"""
Analyze raw 6978 library to identify barcodes expected to integrate multiple times.

Logic:
- Each fish draws ~3000 barcodes from the library
- If a barcode is abundant enough, it will integrate multiple times across 12 fish
- Calculate threshold: what library fraction leads to multiple integrations?

Usage:
    python analyze_raw_6978_library.py <extracted_barcodes_6978.pkl>
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def analyze_raw_library(library_file, avg_barcodes_per_fish=3000, n_fish=12):
    """Analyze raw library to identify high-abundance barcodes."""
    print("="*80)
    print("RAW 6978 LIBRARY ANALYSIS")
    print("="*80)
    print()

    # Load library data
    print(f"Loading: {library_file}")
    with open(library_file, 'rb') as f:
        library_data = pickle.load(f)

    # Extract barcode counts
    # The structure might be different, check what's available
    if isinstance(library_data, dict):
        if 'overall_barcode_counts' in library_data:
            barcode_counts = library_data['overall_barcode_counts']
        elif 'barcode_counts' in library_data:
            barcode_counts = library_data['barcode_counts']
        else:
            # Try to find the actual counts
            print(f"Available keys: {list(library_data.keys())}")
            # Assume first dict-like value
            for key, value in library_data.items():
                if isinstance(value, dict) and len(value) > 100:
                    barcode_counts = value
                    print(f"Using key '{key}' as barcode counts")
                    break

    print(f"\nTotal barcodes in library: {len(barcode_counts):,}")
    total_reads = sum(barcode_counts.values())
    print(f"Total reads: {total_reads:,}")

    # Calculate RPM for each barcode
    barcode_rpm = {bc: (count / total_reads) * 1e6 for bc, count in barcode_counts.items()}

    # Calculate expected integrations threshold
    print(f"\n{'='*80}")
    print("CALCULATING MULTI-INTEGRATION THRESHOLD")
    print(f"{'='*80}")
    print(f"\nAssumptions:")
    print(f"  - Average barcodes per fish: {avg_barcodes_per_fish:,}")
    print(f"  - Number of fish: {n_fish}")
    print(f"  - Total barcodes across all fish: {avg_barcodes_per_fish * n_fish:,}")

    # For a barcode at fraction p of library:
    # Expected integrations per fish = avg_barcodes_per_fish * p
    # Expected total integrations = n_fish * avg_barcodes_per_fish * p
    #
    # Threshold: Expected TOTAL integrations across all 12 fish = 1
    # This is the point where the barcode starts appearing multiple times
    # (could be in multiple fish OR multiple times in same fish)

    expected_integrations_threshold = 1.0

    # Solve for library fraction:
    # n_fish * avg_barcodes_per_fish * fraction = expected_integrations_threshold
    total_barcodes_drawn = n_fish * avg_barcodes_per_fish
    fraction_threshold = expected_integrations_threshold / total_barcodes_drawn
    rpm_threshold = fraction_threshold * 1e6

    # Calculate λ per fish at this threshold
    lambda_per_fish_at_threshold = avg_barcodes_per_fish * fraction_threshold

    print(f"\nThreshold calculation:")
    print(f"  Expected total integrations threshold: {expected_integrations_threshold:.1f}")
    print(f"  Total barcodes drawn (12 fish × {avg_barcodes_per_fish:,}): {total_barcodes_drawn:,}")
    print(f"  Library fraction threshold: {fraction_threshold:.6f} ({fraction_threshold*100:.4f}%)")
    print(f"  RPM threshold: {rpm_threshold:,.1f}")
    print(f"  λ per fish at threshold: {lambda_per_fish_at_threshold:.4f}")

    print(f"\nInterpretation:")
    print(f"  Barcodes with RPM ≥ {rpm_threshold:,.1f} are expected to integrate ≥1 time total")
    print(f"  This could mean:")
    print(f"    - Multiple integrations in the same fish, OR")
    print(f"    - Single integrations in multiple fish, OR")
    print(f"    - A combination of both")

    # Find barcodes above threshold
    high_abundance_barcodes = {bc: rpm for bc, rpm in barcode_rpm.items() if rpm >= rpm_threshold}

    print(f"\n{'='*80}")
    print("HIGH-ABUNDANCE BARCODES (ABOVE THRESHOLD)")
    print(f"{'='*80}")
    print(f"\nBarcodes with RPM ≥ {rpm_threshold:,.1f}: {len(high_abundance_barcodes):,}")

    if high_abundance_barcodes:
        # Sort by RPM
        sorted_barcodes = sorted(high_abundance_barcodes.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop 50 high-abundance barcodes:")
        print(f"  {'Rank':>5} {'Barcode':<50} {'RPM':>15} {'Expected Integrations':>25}")
        print(f"  {'-'*5} {'-'*50} {'-'*15} {'-'*25}")

        for rank, (bc, rpm) in enumerate(sorted_barcodes[:50], 1):
            lib_fraction = rpm / 1e6
            expected_integrations = n_fish * avg_barcodes_per_fish * lib_fraction
            print(f"  {rank:>5} {bc[:50]:<50} {rpm:>15.1f} {expected_integrations:>25.1f}")

        # Statistics
        high_rpms = [rpm for bc, rpm in sorted_barcodes]
        print(f"\nStatistics for high-abundance barcodes:")
        print(f"  Mean RPM: {np.mean(high_rpms):,.1f}")
        print(f"  Median RPM: {np.median(high_rpms):,.1f}")
        print(f"  Max RPM: {np.max(high_rpms):,.1f}")
        print(f"  Min RPM (above threshold): {np.min(high_rpms):,.1f}")

        # Calculate expected total integrations
        total_expected = sum(rpm / 1e6 * n_fish * avg_barcodes_per_fish for rpm in high_rpms)
        print(f"\n  Total expected integrations from these barcodes: {total_expected:,.1f}")
        print(f"  (These {len(high_abundance_barcodes):,} barcodes should account for ~{total_expected:,.0f} integrations)")

    else:
        print("\n  No barcodes found above threshold!")
        print(f"  This suggests library was diverse - no single barcode dominates")

    # Overall distribution
    print(f"\n{'='*80}")
    print("OVERALL RPM DISTRIBUTION")
    print(f"{'='*80}")

    all_rpms = sorted(barcode_rpm.values(), reverse=True)

    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    print(f"\nRPM percentiles:")
    print(f"  {'Percentile':<15} {'RPM':>15} {'Expected Integrations':>25}")
    print(f"  {'-'*15} {'-'*15} {'-'*25}")

    for p in percentiles:
        idx = int((100 - p) / 100 * len(all_rpms))
        idx = min(idx, len(all_rpms) - 1)
        rpm = all_rpms[idx]
        lib_fraction = rpm / 1e6
        expected_integrations = n_fish * avg_barcodes_per_fish * lib_fraction
        print(f"  Top {p:>5.1f}%{' '*6} {rpm:>15.2f} {expected_integrations:>25.2f}")

    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")

    plot_library_analysis(all_rpms, rpm_threshold, high_abundance_barcodes)

    print("\nDone!")


def plot_library_analysis(all_rpms, rpm_threshold, high_abundance_barcodes):
    """Generate plots for library analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RPM distribution (log scale)
    ax = axes[0, 0]
    ax.hist(np.log10(all_rpms), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.log10(rpm_threshold), color='red', linestyle='--', linewidth=2,
              label=f'Threshold (RPM={rpm_threshold:.1f})')
    ax.set_xlabel('log10(RPM)', fontsize=11)
    ax.set_ylabel('Number of Barcodes', fontsize=11)
    ax.set_title('6978 Library RPM Distribution\n(All barcodes)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative fraction of reads
    ax = axes[0, 1]
    cumulative_fraction = np.cumsum(sorted(all_rpms, reverse=True)) / 1e6
    ax.plot(range(1, len(cumulative_fraction) + 1), cumulative_fraction, color='navy', linewidth=2)
    ax.set_xlabel('Number of Top Barcodes', fontsize=11)
    ax.set_ylabel('Cumulative Fraction of Library', fontsize=11)
    ax.set_title('Library Concentration\n(How many barcodes account for X% of library?)',
                fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    for frac in [0.1, 0.5, 0.9]:
        idx = np.searchsorted(cumulative_fraction, frac)
        ax.axhline(frac, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(cumulative_fraction) * 0.7, frac + 0.02, f'{int(frac*100)}% at ~{idx:,} barcodes',
               fontsize=9)

    # Plot 3: Rank vs RPM
    ax = axes[1, 0]
    ranks = range(1, min(10000, len(all_rpms)) + 1)
    ax.plot(ranks, all_rpms[:len(ranks)], color='darkgreen', linewidth=2)
    ax.axhline(rpm_threshold, color='red', linestyle='--', linewidth=2,
              label=f'Threshold (RPM={rpm_threshold:.1f})')
    ax.set_xlabel('Barcode Rank', fontsize=11)
    ax.set_ylabel('RPM', fontsize=11)
    ax.set_title('Barcode Abundance by Rank\n(Top 10k barcodes)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: High-abundance barcode distribution
    ax = axes[1, 1]
    if high_abundance_barcodes:
        high_rpms = sorted(high_abundance_barcodes.values(), reverse=True)
        ax.bar(range(len(high_rpms)), high_rpms, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Barcode Index (sorted by RPM)', fontsize=11)
        ax.set_ylabel('RPM', fontsize=11)
        ax.set_title(f'High-Abundance Barcodes (n={len(high_rpms):,})\n(Expected to multi-integrate)',
                    fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No barcodes above threshold',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'raw_6978_library_analysis_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("ERROR: Missing required argument")
        print()
        print("Usage:")
        print("  python analyze_raw_6978_library.py <extracted_barcodes_6978.pkl>")
        print()
        print("Optional arguments:")
        print("  --avg-barcodes-per-fish N   (default: 3000)")
        print("  --n-fish N                  (default: 12)")
        print()
        print("Example:")
        print("  python analyze_raw_6978_library.py extracted_barcodes_6978_*.pkl")
        print("  python analyze_raw_6978_library.py extracted_barcodes_6978_*.pkl --avg-barcodes-per-fish 2500")
        return

    library_file = sys.argv[1]

    # Parse optional arguments
    avg_barcodes_per_fish = 3000
    n_fish = 12

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--avg-barcodes-per-fish' and i + 1 < len(sys.argv):
            avg_barcodes_per_fish = int(sys.argv[i + 1])
        elif arg == '--n-fish' and i + 1 < len(sys.argv):
            n_fish = int(sys.argv[i + 1])

    analyze_raw_library(library_file, avg_barcodes_per_fish, n_fish)


if __name__ == "__main__":
    main()
