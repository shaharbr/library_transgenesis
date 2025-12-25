#!/usr/bin/env python3
"""
Generate manuscript figures for barcode abundance distribution analysis.

Generates figures for manuscript section on barcode abundance variance:
- Fig 3E: RPM histograms for 7462 library and combined 6978
- Supp X1: RPM histograms for 7462 and each fish separately
- Supp X2: Diversity metrics (CV, Shannon, Quartile Ratio) for 7462 and each fish

RPM = Reads Per Million (normalized barcode abundance)
CV = Coefficient of Variation
Shannon = Shannon diversity index
Quartile Ratio = Q3/Q1 (3rd quartile divided by 1st quartile)

Usage:
    python analyze_barcode_abundance_manuscript.py <high_confidence_7462.pkl> <high_confidence_6978.pkl>
"""

import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from datetime import datetime

sns.set_style("whitegrid")


def load_barcode_data(file_path):
    """Load barcode data from pickle file."""
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    barcodes = data['barcodes']
    sample_type = data.get('sample_type', 'unknown')
    print(f"  Sample type: {sample_type}")
    print(f"  Number of barcodes: {len(barcodes):,}")

    return barcodes, sample_type


def calculate_rpm(counts):
    """
    Convert raw counts to RPM (Reads Per Million).

    Args:
        counts: Array of raw read counts

    Returns:
        Array of RPM values
    """
    total = np.sum(counts)
    if total == 0:
        return counts
    return (counts / total) * 1e6


def calculate_diversity_metrics(counts):
    """
    Calculate diversity metrics for a set of barcode counts.

    Args:
        counts: List/array of barcode counts (can be raw counts or RPM)

    Returns:
        Dictionary with diversity metrics
    """
    if len(counts) == 0:
        return {
            'n_barcodes': 0,
            'total_reads': 0,
            'cv': np.nan,
            'shannon_diversity': np.nan,
            'quartile_ratio': np.nan,  # Quartile ratio (Q3 / Q1)
        }

    counts = np.array(counts)
    counts = counts[counts > 0]  # Filter non-zero

    if len(counts) == 0:
        return {
            'n_barcodes': 0,
            'total_reads': 0,
            'cv': np.nan,
            'shannon_diversity': np.nan,
            'quartile_ratio': np.nan,
        }

    # Basic statistics
    n_barcodes = len(counts)
    total_reads = np.sum(counts)
    mean_count = np.mean(counts)

    # CV (coefficient of variation)
    if mean_count > 0:
        cv = np.std(counts) / mean_count
    else:
        cv = np.nan

    # Shannon diversity
    # H = -sum(p_i * log(p_i)) where p_i is proportion of reads for barcode i
    proportions = counts / total_reads
    shannon = entropy(proportions, base=2)  # Using base 2 for bits

    # Quartile ratio (Q3 / Q1)
    if len(counts) >= 4:
        q1 = np.percentile(counts, 25)
        q3 = np.percentile(counts, 75)
        if q1 > 0:
            quartile_ratio = q3 / q1
        else:
            quartile_ratio = np.nan
    else:
        quartile_ratio = np.nan

    return {
        'n_barcodes': n_barcodes,
        'total_reads': total_reads,
        'cv': cv,
        'shannon_diversity': shannon,
        'quartile_ratio': quartile_ratio,
    }


def extract_7462_data(barcodes_7462):
    """
    Extract barcode counts for 7462 library.

    Returns:
        Array of counts
    """
    counts = []
    for barcode, info in barcodes_7462.items():
        count = info.get('7462', 0)
        if count > 0:
            counts.append(count)

    return np.array(counts)


def extract_6978_combined_data(barcodes_6978):
    """
    Extract combined barcode counts across all fish in 6978.

    Returns:
        Array of combined counts (sum across all fish for each barcode)
    """
    counts = []
    for barcode, info in barcodes_6978.items():
        fish_counts = info.get('6978', {})
        total_count = sum(fish_counts.values())
        if total_count > 0:
            counts.append(total_count)

    return np.array(counts)


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


def extract_6978_per_fish_data(barcodes_6978):
    """
    Extract barcode counts for each fish separately in 6978.

    Returns:
        Dictionary mapping fish ID to array of counts (ordered by specified fish order)
    """
    # Fish order as specified
    fish_order = [
        'ACCTT', 'AATCG', 'AGAGA', 'CGGAG', 'CTACT', 'GAAAC',
        'GCGCA', 'CCTGC', 'TAGGT', 'GTCGG', 'TTTAA', 'TGCCC'
    ]

    # Get all fish from data
    all_fish_in_data = set()
    for info in barcodes_6978.values():
        all_fish_in_data.update(info.get('6978', {}).keys())

    # Order fish according to specified order, then any extras
    all_fish = [f for f in fish_order if f in all_fish_in_data]
    all_fish.extend(sorted([f for f in all_fish_in_data if f not in fish_order]))

    # Extract counts per fish
    fish_data = {}
    for fish in all_fish:
        counts = []
        for barcode, info in barcodes_6978.items():
            count = info.get('6978', {}).get(fish, 0)
            if count > 0:
                counts.append(count)
        fish_data[fish] = np.array(counts)

    return fish_data


def plot_fig3e(counts_7462, counts_6978_combined, output_prefix):
    """
    Generate Fig 3E: RPM histograms for 7462 and combined 6978.

    Two side-by-side histograms showing distribution of barcode abundance (in RPM).
    """
    print("\n" + "="*80)
    print("GENERATING FIG 3E: RPM HISTOGRAMS (7462 AND COMBINED 6978)")
    print("="*80)

    # Convert to RPM
    rpm_7462 = calculate_rpm(counts_7462)
    rpm_6978 = calculate_rpm(counts_6978_combined)

    # Determine global x-axis limits (logarithmic)
    all_rpm = np.concatenate([rpm_7462, rpm_6978])
    all_rpm_positive = all_rpm[all_rpm > 0]
    x_min = np.min(all_rpm_positive) * 0.5  # Slightly below minimum
    x_max = np.max(all_rpm_positive) * 2.0  # Slightly above maximum

    # Create logarithmic bins
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot both histograms first to determine y-axis range
    hist_7462, _ = np.histogram(rpm_7462, bins=log_bins)
    hist_6978, _ = np.histogram(rpm_6978, bins=log_bins)

    # Determine global y-axis limits
    y_max = max(np.max(hist_7462), np.max(hist_6978)) * 1.2
    y_min = 0.5  # Lower limit for log scale

    # Plot 7462 (dark gray)
    ax = axes[0]
    ax.hist(rpm_7462, bins=log_bins, alpha=0.7, color='#404040', edgecolor='black')
    ax.set_xlabel('Barcode Abundance (RPM)', fontsize=12)
    ax.set_ylabel('Number of Barcodes', fontsize=12)
    ax.set_title('Injected Library', fontsize=13, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add statistics
    median_rpm = np.median(rpm_7462)
    mean_rpm = np.mean(rpm_7462)
    ax.axvline(median_rpm, color='red', linestyle='--', linewidth=2, label=f'Median={median_rpm:.1f}')
    ax.axvline(mean_rpm, color='orange', linestyle='--', linewidth=2, label=f'Mean={mean_rpm:.1f}')
    ax.legend(fontsize=10)

    # Plot combined 6978 (medium gray)
    ax = axes[1]
    ax.hist(rpm_6978, bins=log_bins, alpha=0.7, color='#808080', edgecolor='black')
    ax.set_xlabel('Barcode Abundance (RPM)', fontsize=12)
    ax.set_ylabel('Number of Barcodes', fontsize=12)
    ax.set_title('Combined 6978 (All Fish)', fontsize=13, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add statistics
    median_rpm = np.median(rpm_6978)
    mean_rpm = np.mean(rpm_6978)
    ax.axvline(median_rpm, color='red', linestyle='--', linewidth=2, label=f'Median={median_rpm:.1f}')
    ax.axvline(mean_rpm, color='orange', linestyle='--', linewidth=2, label=f'Mean={mean_rpm:.1f}')
    ax.legend(fontsize=10)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_fig3e_rpm_histograms_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_fig3e_rpm_histograms_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"  Saved SVG: {output_svg}")

    plt.close()


def plot_suppx1(counts_7462, fish_data_6978, output_prefix):
    """
    Generate Supp X1: RPM histograms for 7462 and each fish separately.

    Grid of histograms showing distribution of barcode abundance (in RPM) for
    7462 and each individual fish.
    """
    print("\n" + "="*80)
    print("GENERATING SUPP X1: RPM HISTOGRAMS PER FISH")
    print("="*80)

    n_fish = len(fish_data_6978)
    all_fish = list(fish_data_6978.keys())  # Already in correct order

    # Convert all data to RPM first to determine global x-axis limits
    rpm_7462 = calculate_rpm(counts_7462)
    all_rpm_data = [rpm_7462]
    for fish in all_fish:
        rpm = calculate_rpm(fish_data_6978[fish])
        all_rpm_data.append(rpm)

    # Determine global x-axis limits (logarithmic)
    all_rpm = np.concatenate(all_rpm_data)
    all_rpm_positive = all_rpm[all_rpm > 0]
    x_min = np.min(all_rpm_positive) * 0.5  # Slightly below minimum
    x_max = np.max(all_rpm_positive) * 2.0  # Slightly above maximum

    # Create logarithmic bins
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    # Calculate histograms for all fish to determine global y-axis limits
    hist_counts = []
    for fish in all_fish:
        rpm = calculate_rpm(fish_data_6978[fish])
        hist_fish, _ = np.histogram(rpm, bins=log_bins)
        hist_counts.append(np.max(hist_fish))

    # Determine global y-axis limits
    y_max = max(hist_counts) * 1.2
    y_min = 0.5  # Lower limit for log scale

    # Create grid: 3 rows × 4 columns for 12 fish
    n_cols = 4
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))

    # Plot each fish in 3x4 grid (medium gray)
    for idx, fish in enumerate(all_fish):
        row = idx // 4  # Rows 0, 1, 2
        col = idx % 4   # Columns 0, 1, 2, 3

        counts = fish_data_6978[fish]
        rpm = calculate_rpm(counts)

        ax = axes[row, col]
        ax.hist(rpm, bins=log_bins, alpha=0.7, color='#808080', edgecolor='black')
        ax.set_xlabel('Barcode Abundance (RPM)', fontsize=11)
        ax.set_ylabel('Number of Barcodes', fontsize=11)
        ax.set_title(get_fish_display_name(fish), fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if len(rpm) > 0:
            median_rpm = np.median(rpm)
            ax.axvline(median_rpm, color='red', linestyle='--', linewidth=2, label=f'Median={median_rpm:.1f}')
            ax.legend(fontsize=9)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_suppx1_rpm_histograms_per_fish_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_suppx1_rpm_histograms_per_fish_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"  Saved SVG: {output_svg}")

    plt.close()


def plot_suppx2(counts_7462, fish_data_6978, output_prefix):
    """
    Generate Supp X2: Diversity metrics (CV, Shannon, Quartile Ratio) for 7462 and each fish.

    Bar plots showing three diversity metrics across 7462 and all fish.
    """
    print("\n" + "="*80)
    print("GENERATING SUPP X2: DIVERSITY METRICS (CV, SHANNON, QUARTILE RATIO)")
    print("="*80)

    all_fish = list(fish_data_6978.keys())  # Already in correct order

    # Calculate metrics for 7462
    metrics_7462 = calculate_diversity_metrics(counts_7462)

    # Calculate metrics for each fish
    metrics_per_fish = {}
    for fish in all_fish:
        counts = fish_data_6978[fish]
        metrics_per_fish[fish] = calculate_diversity_metrics(counts)

    # Prepare data for plotting
    sample_ids = ['7462'] + all_fish
    sample_labels = ['Injected Library'] + [get_fish_display_name(f) for f in all_fish]
    cv_values = [metrics_7462['cv']] + [metrics_per_fish[f]['cv'] for f in all_fish]
    shannon_values = [metrics_7462['shannon_diversity']] + [metrics_per_fish[f]['shannon_diversity'] for f in all_fish]
    quartile_values = [metrics_7462['quartile_ratio']] + [metrics_per_fish[f]['quartile_ratio'] for f in all_fish]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Colors: dark gray for 7462, medium gray for fish
    colors = ['#404040'] + ['#808080'] * len(all_fish)

    # Plot CV
    ax = axes[0]
    bars = ax.bar(range(len(sample_labels)), cv_values, alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('CV (Coefficient of Variation)', fontsize=12)
    ax.set_title('Coefficient of Variation', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels above bars
    for i, (bar, value) in enumerate(zip(bars, cv_values)):
        if np.isfinite(value):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # Add mean line for fish only
    fish_cv_values = [v for v in cv_values[1:] if np.isfinite(v)]
    if len(fish_cv_values) > 0:
        mean_cv = np.mean(fish_cv_values)
        ax.axhline(mean_cv, color='red', linestyle='--', linewidth=2,
                  label=f'Mean (fish only)={mean_cv:.2f}')
        ax.legend(fontsize=10)

    # Plot Shannon
    ax = axes[1]
    bars = ax.bar(range(len(sample_labels)), shannon_values, alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Shannon Diversity (bits)', fontsize=12)
    ax.set_title('Shannon Diversity Index', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels above bars
    for i, (bar, value) in enumerate(zip(bars, shannon_values)):
        if np.isfinite(value):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # Add mean line for fish only
    fish_shannon_values = [v for v in shannon_values[1:] if np.isfinite(v)]
    if len(fish_shannon_values) > 0:
        mean_shannon = np.mean(fish_shannon_values)
        ax.axhline(mean_shannon, color='red', linestyle='--', linewidth=2,
                  label=f'Mean (fish only)={mean_shannon:.2f}')
        ax.legend(fontsize=10)

    # Plot Quartile Ratio (linear y-axis)
    ax = axes[2]
    bars = ax.bar(range(len(sample_labels)), quartile_values, alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Quartile Ratio (Q3/Q1)', fontsize=12)
    ax.set_title('Quartile Ratio (Q3/Q1)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    # Linear y-axis (no log scale)

    # Add value labels above bars
    for i, (bar, value) in enumerate(zip(bars, quartile_values)):
        if np.isfinite(value):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # Add mean line for fish only
    fish_quartile_values = [v for v in quartile_values[1:] if np.isfinite(v)]
    if len(fish_quartile_values) > 0:
        mean_quartile = np.mean(fish_quartile_values)
        ax.axhline(mean_quartile, color='red', linestyle='--', linewidth=2,
                  label=f'Mean (fish only)={mean_quartile:.2f}')
        ax.legend(fontsize=10)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_suppx2_diversity_metrics_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_suppx2_diversity_metrics_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"  Saved SVG: {output_svg}")

    plt.close()

    # Also save statistics to CSV
    stats_data = []
    for i, (sample_id, sample_label) in enumerate(zip(sample_ids, sample_labels)):
        stats_data.append({
            'sample_id': sample_id,
            'sample_label': sample_label,
            'cv': cv_values[i],
            'shannon_diversity': shannon_values[i],
            'quartile_ratio': quartile_values[i],
        })

    df_stats = pd.DataFrame(stats_data)
    stats_file = f'{output_prefix}_suppx2_diversity_metrics_{timestamp}.csv'
    df_stats.to_csv(stats_file, index=False)
    print(f"  Saved statistics: {stats_file}")


def plot_combined_figure3(counts_7462, counts_6978_combined, fish_data_6978,
                          barcodes_7462_list, barcodes_6978_list, output_prefix):
    """
    Generate combined Figure 3 with three panels:
    - Left (70% width, 100% height): Barcode counts per fish
    - Top right (30% width, 50% height): DNA logos
    - Bottom right (30% width, 50% height): RPM histograms
    """
    print("\n" + "="*80)
    print("GENERATING COMBINED FIGURE 3")
    print("="*80)

    # Create figure with custom layout using GridSpec
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(3, 4, width_ratios=[7, 2, 3, 3], height_ratios=[3,1,4],
                          hspace=0.3, wspace=0.4)

    # Left panel: Barcode counts (spans both rows, first column)
    ax_counts = fig.add_subplot(gs[:, 0:2])

    # Top right: DNA logos (side by side)
    ax_logo_7462 = fig.add_subplot(gs[0, 2])
    ax_logo_6978 = fig.add_subplot(gs[0, 3])

    # Bottom right: RPM histograms (side by side)
    ax_hist_7462 = fig.add_subplot(gs[2, 2])
    ax_hist_6978 = fig.add_subplot(gs[2, 3])

    # Hide the rightmost column
    fig.add_subplot(gs[:, 3]).axis('off')

    # ========================================================================
    # Panel A: Barcode counts per fish (left, full height)
    # ========================================================================
    all_fish = list(fish_data_6978.keys())
    barcode_counts = []
    for fish in all_fish:
        n_barcodes = len(fish_data_6978[fish])
        barcode_counts.append(n_barcodes)

    fish_labels = [get_fish_display_name(fish) for fish in all_fish]

    bars = ax_counts.bar(range(len(fish_labels)), barcode_counts, alpha=0.7,
                         color='#808080', edgecolor='black', linewidth=1.5)

    ax_counts.set_xlabel('Fish Sample', fontsize=18, fontweight='bold')
    ax_counts.set_ylabel('Number of Unique Barcodes', fontsize=18, fontweight='bold')
    ax_counts.set_title('A. Barcode Integration per Fish', fontsize=20, fontweight='bold', loc='left')
    ax_counts.set_xticks(range(len(fish_labels)))
    ax_counts.set_xticklabels(fish_labels, rotation=45, ha='right', fontsize=16)
    ax_counts.tick_params(axis='y', labelsize=16)
    ax_counts.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, count in zip(bars, barcode_counts):
        height = bar.get_height()
        ax_counts.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{count:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add mean line
    mean_count = np.mean(barcode_counts)
    ax_counts.axhline(mean_count, color='red', linestyle='--', linewidth=3,
                     label=f'Mean = {mean_count:,.0f}')
    ax_counts.legend(fontsize=16)

    # ========================================================================
    # Panel B: DNA Logos (top right, side by side)
    # ========================================================================
    # Calculate DNA logo data
    from collections import Counter

    def get_barcode_length(barcodes):
        lengths = Counter(len(bc) for bc in barcodes)
        return lengths.most_common(1)[0][0]

    def filter_to_length(barcodes, length):
        return [bc for bc in barcodes if len(bc) == length]

    def calculate_positional_frequencies(barcodes, length):
        freqs = np.zeros((length, 4))
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for bc in barcodes:
            for pos, nuc in enumerate(bc):
                if nuc in nuc_to_idx:
                    freqs[pos, nuc_to_idx[nuc]] += 1
        freqs = freqs / len(barcodes) if len(barcodes) > 0 else freqs
        return freqs

    length = get_barcode_length(barcodes_7462_list + barcodes_6978_list)
    bc_7462_filt = filter_to_length(barcodes_7462_list, length)
    bc_6978_filt = filter_to_length(barcodes_6978_list, length)

    freqs_7462 = calculate_positional_frequencies(bc_7462_filt, length)
    freqs_6978 = calculate_positional_frequencies(bc_6978_filt, length)

    positions = np.arange(length)
    colors_nuc = {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}
    nucleotides = ['A', 'C', 'G', 'T']

    # Plot library DNA logo
    bottom = np.zeros(length)
    for i, nuc in enumerate(nucleotides):
        values = freqs_7462[:, i]
        ax_logo_7462.bar(positions, values, bottom=bottom, color=colors_nuc[nuc],
                        label=nuc if i == 0 else '', alpha=0.9, edgecolor='black', linewidth=0.5)
        bottom += values

    ax_logo_7462.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax_logo_7462.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax_logo_7462.set_title('B. Injected Library', fontsize=16, fontweight='bold', loc='left')
    ax_logo_7462.set_ylim(0, 1)
    ax_logo_7462.set_xlim(-0.5, length - 0.5)
    ax_logo_7462.tick_params(axis='both', labelsize=12)
    ax_logo_7462.grid(True, alpha=0.3, axis='y')

    # Plot combined fish DNA logo
    bottom = np.zeros(length)
    for i, nuc in enumerate(nucleotides):
        values = freqs_6978[:, i]
        ax_logo_6978.bar(positions, values, bottom=bottom, color=colors_nuc[nuc],
                        label=nuc, alpha=0.9, edgecolor='black', linewidth=0.5)
        bottom += values

    ax_logo_6978.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax_logo_6978.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax_logo_6978.set_title('Combined Fish', fontsize=16, fontweight='bold', loc='left')
    ax_logo_6978.set_ylim(0, 1)
    ax_logo_6978.set_xlim(-0.5, length - 0.5)
    ax_logo_6978.tick_params(axis='both', labelsize=12)
    ax_logo_6978.legend(loc='upper right', fontsize=12, ncol=4, framealpha=0.9)
    ax_logo_6978.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Panel C: RPM Histograms (bottom right, side by side)
    # ========================================================================
    rpm_7462 = calculate_rpm(counts_7462)
    rpm_6978 = calculate_rpm(counts_6978_combined)

    # Determine global x-axis and y-axis limits
    all_rpm = np.concatenate([rpm_7462, rpm_6978])
    all_rpm_positive = all_rpm[all_rpm > 0]
    x_min = np.min(all_rpm_positive) * 0.5
    x_max = np.max(all_rpm_positive) * 2.0

    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 40)

    # Calculate histograms to determine y-axis limits
    hist_7462, _ = np.histogram(rpm_7462, bins=log_bins)
    hist_6978, _ = np.histogram(rpm_6978, bins=log_bins)
    y_max = max(np.max(hist_7462), np.max(hist_6978)) * 1.2
    y_min = 0.5

    # Plot library histogram
    ax_hist_7462.hist(rpm_7462, bins=log_bins, alpha=0.7, color='#404040',
                     edgecolor='black', linewidth=1)
    ax_hist_7462.set_xlabel('Barcode Abundance (RPM)', fontsize=14, fontweight='bold')
    ax_hist_7462.set_ylabel('Number of Barcodes', fontsize=14, fontweight='bold')
    ax_hist_7462.set_title('C. Injected Library', fontsize=16, fontweight='bold', loc='left')
    ax_hist_7462.set_xlim(x_min, x_max)
    ax_hist_7462.set_ylim(y_min, y_max)
    ax_hist_7462.set_xscale('log')
    ax_hist_7462.set_yscale('log')
    ax_hist_7462.tick_params(axis='both', labelsize=12)
    ax_hist_7462.grid(True, alpha=0.3)

    # Plot combined fish histogram
    ax_hist_6978.hist(rpm_6978, bins=log_bins, alpha=0.7, color='#808080',
                     edgecolor='black', linewidth=1)
    ax_hist_6978.set_xlabel('Barcode Abundance (RPM)', fontsize=14, fontweight='bold')
    ax_hist_6978.set_ylabel('Number of Barcodes', fontsize=14, fontweight='bold')
    ax_hist_6978.set_title('Combined Fish', fontsize=16, fontweight='bold', loc='left')
    ax_hist_6978.set_xlim(x_min, x_max)
    ax_hist_6978.set_ylim(y_min, y_max)
    ax_hist_6978.set_xscale('log')
    ax_hist_6978.set_yscale('log')
    ax_hist_6978.tick_params(axis='both', labelsize=12)
    ax_hist_6978.grid(True, alpha=0.3)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_png = f'{output_prefix}_figure3_combined_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n  Saved PNG: {output_png}")

    output_svg = f'{output_prefix}_figure3_combined_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"  Saved SVG: {output_svg}")

    plt.close()


def plot_barcode_counts_per_fish(fish_data_6978, output_prefix):
    """
    Generate bar graph showing the number of unique barcodes integrated in each fish.

    Shows total barcode count per fish (not separated by shared/unique).
    """
    print("\n" + "="*80)
    print("GENERATING BARCODE COUNT PER FISH BAR GRAPH")
    print("="*80)

    all_fish = list(fish_data_6978.keys())  # Already in correct order

    # Count barcodes per fish
    barcode_counts = []
    for fish in all_fish:
        n_barcodes = len(fish_data_6978[fish])
        barcode_counts.append(n_barcodes)
        print(f"  {get_fish_display_name(fish)}: {n_barcodes:,} barcodes")

    # Prepare labels
    fish_labels = [get_fish_display_name(fish) for fish in all_fish]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot (medium gray for all fish)
    bars = ax.bar(range(len(fish_labels)), barcode_counts, alpha=0.7,
                  color='#808080', edgecolor='black')

    ax.set_xlabel('Fish Sample', fontsize=12)
    ax.set_ylabel('Number of Unique Barcodes', fontsize=12)
    ax.set_title('Number of Integrated Barcodes per Fish', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(fish_labels)))
    ax.set_xticklabels(fish_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels above bars
    for i, (bar, count) in enumerate(zip(bars, barcode_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{count:,}',
               ha='center', va='bottom', fontsize=9)

    # Add mean line
    mean_count = np.mean(barcode_counts)
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=2,
              label=f'Mean = {mean_count:,.0f}')
    ax.legend(fontsize=10)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_barcode_counts_per_fish_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n  Saved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_barcode_counts_per_fish_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"  Saved SVG: {output_svg}")

    plt.close()

    # Also save counts to CSV
    stats_data = []
    for fish, fish_label, count in zip(all_fish, fish_labels, barcode_counts):
        stats_data.append({
            'fish_id': fish,
            'fish_label': fish_label,
            'n_barcodes': count
        })

    df_stats = pd.DataFrame(stats_data)
    csv_file = f'{output_prefix}_barcode_counts_per_fish_{timestamp}.csv'
    df_stats.to_csv(csv_file, index=False)
    print(f"  Saved statistics: {csv_file}")


def main():
    if len(sys.argv) < 3:
        print("ERROR: Missing required arguments")
        print()
        print("Usage:")
        print("  python analyze_barcode_abundance_manuscript.py <high_confidence_7462.pkl> <high_confidence_6978.pkl> [--output PREFIX]")
        print()
        print("Arguments:")
        print("  high_confidence_7462.pkl    High-confidence barcode file for 7462")
        print("  high_confidence_6978.pkl    High-confidence barcode file for 6978")
        print("  --output PREFIX             Output file prefix (default: abundance)")
        print()
        print("Generates:")
        print("  - Fig 3E: RPM histograms for 7462 and combined 6978")
        print("  - Supp X1: RPM histograms for 7462 and each fish separately")
        print("  - Supp X2: Diversity metrics (CV, Shannon, Quartile Ratio) for 7462 and each fish")
        print("  - Barcode counts per fish bar graph")
        print()
        print("Example:")
        print("  python analyze_barcode_abundance_manuscript.py high_confidence_barcodes_7462*.pkl high_confidence_barcodes_6978*.pkl")
        return

    file_7462 = sys.argv[1]
    file_6978 = sys.argv[2]

    output_prefix = "abundance"
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == '--output' and i + 1 < len(sys.argv):
            output_prefix = sys.argv[i + 1]

    print("="*80)
    print("BARCODE ABUNDANCE DISTRIBUTION ANALYSIS - MANUSCRIPT FIGURES")
    print("="*80)
    print()

    # Load data
    print("Loading barcode data...")
    barcodes_7462, sample_type_7462 = load_barcode_data(file_7462)
    barcodes_6978, sample_type_6978 = load_barcode_data(file_6978)
    print()

    # Verify sample types
    if sample_type_7462 != '7462':
        print(f"WARNING: Expected sample_type '7462', got '{sample_type_7462}'")
    if sample_type_6978 != '6978':
        print(f"WARNING: Expected sample_type '6978', got '{sample_type_6978}'")

    # Extract count data
    print("Extracting barcode counts...")
    counts_7462 = extract_7462_data(barcodes_7462)
    counts_6978_combined = extract_6978_combined_data(barcodes_6978)
    fish_data_6978 = extract_6978_per_fish_data(barcodes_6978)

    print(f"  7462: {len(counts_7462):,} barcodes")
    print(f"  6978 combined: {len(counts_6978_combined):,} barcodes")
    print(f"  6978 fish: {len(fish_data_6978)} fish")
    print()

    # Extract barcode sequences for DNA logos
    print("Extracting barcode sequences...")
    barcodes_7462_list = list(barcodes_7462.keys())
    barcodes_6978_list = list(barcodes_6978.keys())
    print(f"  7462 sequences: {len(barcodes_7462_list):,}")
    print(f"  6978 sequences: {len(barcodes_6978_list):,}")
    print()

    # Generate figures
    plot_combined_figure3(counts_7462, counts_6978_combined, fish_data_6978,
                         barcodes_7462_list, barcodes_6978_list, output_prefix)
    plot_fig3e(counts_7462, counts_6978_combined, output_prefix)
    plot_suppx1(counts_7462, fish_data_6978, output_prefix)
    plot_suppx2(counts_7462, fish_data_6978, output_prefix)
    plot_barcode_counts_per_fish(fish_data_6978, output_prefix)

    print("\n" + "="*80)
    print("MANUSCRIPT FIGURES COMPLETE!")
    print("="*80)
    print()
    print("Generated figures:")
    print("  - Figure 3 Combined: Barcode counts + DNA logos + RPM histograms")
    print("  - Fig 3E: RPM histograms (7462 and combined 6978)")
    print("  - Supp X1: RPM histograms per fish")
    print("  - Supp X2: Diversity metrics (CV, Shannon, Quartile Ratio)")
    print("  - Barcode counts per fish bar graph")


if __name__ == "__main__":
    main()
