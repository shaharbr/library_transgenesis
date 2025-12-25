#!/usr/bin/env python3
"""
Sequence bias analysis for manuscript figures.

Generates:
- Fig 3D: DNA logos for 7462 library and combined 6978 fish
- Supp Y1: DNA logos for 7462 and each fish separately
- Supp Y3: Pairwise Hamming distance violin plots

Usage:
    python analyze_sequence_bias_manuscript.py <hc_7462.pkl> <hc_6978.pkl>
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
import pandas as pd
from datetime import datetime

sns.set_style("whitegrid")


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


def load_filtered_barcodes(hc_7462_file, hc_6978_file):
    """Load both filtered barcode files."""
    print("="*80)
    print("LOADING FILTERED BARCODE DATA")
    print("="*80)
    print()

    # Load 7462
    print(f"Loading 7462: {hc_7462_file}")
    with open(hc_7462_file, 'rb') as f:
        data_7462 = pickle.load(f)

    if data_7462.get('sample_type') != '7462':
        print("WARNING: First file doesn't have sample_type='7462'")

    barcodes_7462 = list(data_7462['barcodes'].keys())
    print(f"  7462 barcodes: {len(barcodes_7462):,}")

    # Load 6978
    print(f"Loading 6978: {hc_6978_file}")
    with open(hc_6978_file, 'rb') as f:
        data_6978 = pickle.load(f)

    if data_6978.get('sample_type') != '6978':
        print("WARNING: Second file doesn't have sample_type='6978'")

    barcodes_6978_dict = data_6978['barcodes']

    # Get per-fish barcodes in specified order
    fish_order = [
        'ACCTT', 'AATCG', 'AGAGA', 'CGGAG', 'CTACT', 'GAAAC',
        'GCGCA', 'CCTGC', 'TAGGT', 'GTCGG', 'TTTAA', 'TGCCC'
    ]

    # Get all fish from data
    all_fish_in_data = set()
    for barcode_info in barcodes_6978_dict.values():
        all_fish_in_data.update(barcode_info['6978'].keys())

    # Order fish according to specified order, then any extras
    all_fish = [f for f in fish_order if f in all_fish_in_data]
    all_fish.extend(sorted([f for f in all_fish_in_data if f not in fish_order]))

    barcodes_per_fish = {}
    for fish in all_fish:
        fish_barcodes = [
            bc for bc, info in barcodes_6978_dict.items()
            if info['6978'].get(fish, 0) > 0
        ]
        barcodes_per_fish[fish] = fish_barcodes
        print(f"  {fish} ({get_fish_display_name(fish)}): {len(fish_barcodes):,} barcodes")

    # Combined 6978
    barcodes_6978_combined = list(barcodes_6978_dict.keys())
    print(f"\nCombined 6978: {len(barcodes_6978_combined):,} barcodes")

    return barcodes_7462, barcodes_6978_combined, barcodes_per_fish, all_fish


def get_barcode_length(barcodes):
    """Get most common barcode length."""
    lengths = [len(bc) for bc in barcodes]
    most_common = Counter(lengths).most_common(1)[0][0]
    return most_common


def filter_to_length(barcodes, length):
    """Filter barcodes to specific length."""
    return [bc for bc in barcodes if len(bc) == length]


def calculate_positional_frequencies(barcodes, length):
    """Calculate nucleotide frequencies at each position."""
    freqs = np.zeros((length, 4))  # positions × nucleotides (A, C, G, T)
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    for bc in barcodes:
        for pos, nuc in enumerate(bc):
            if nuc in nuc_to_idx:
                freqs[pos, nuc_to_idx[nuc]] += 1

    # Normalize
    freqs = freqs / len(barcodes)
    return freqs


def plot_dna_logo(freqs, title, ax):
    """Plot DNA logo as stacked bar chart."""
    length = freqs.shape[0]
    positions = np.arange(length)
    colors_nuc = {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}
    nucleotides = ['A', 'C', 'G', 'T']

    bottom = np.zeros(length)
    for i, nuc in enumerate(nucleotides):
        values = freqs[:, i]
        ax.bar(positions, values, bottom=bottom, color=colors_nuc[nuc],
               label=nuc, alpha=0.9, edgecolor='black', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, length - 0.5)
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')


def generate_figure_3d(barcodes_7462, barcodes_6978_combined, output_prefix):
    """
    Generate Figure 3D: DNA logos for 7462 library and combined 6978.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3D")
    print("="*80)

    # Get common length
    length = get_barcode_length(barcodes_7462 + barcodes_6978_combined)
    print(f"\nBarcode length: {length}bp")

    # Filter to common length
    bc_7462 = filter_to_length(barcodes_7462, length)
    bc_6978 = filter_to_length(barcodes_6978_combined, length)

    print(f"7462: {len(bc_7462):,} barcodes")
    print(f"Combined 6978: {len(bc_6978):,} barcodes")

    # Calculate frequencies
    freqs_7462 = calculate_positional_frequencies(bc_7462, length)
    freqs_6978 = calculate_positional_frequencies(bc_6978, length)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_dna_logo(freqs_7462, 'Injected Library', axes[0])
    plot_dna_logo(freqs_6978, 'Combined Fish', axes[1])

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_Fig3D_DNA_logos_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_Fig3D_DNA_logos_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"Saved SVG: {output_svg}")

    plt.close()


def generate_supp_y1(barcodes_7462, barcodes_per_fish, all_fish, output_prefix):
    """
    Generate Supplemental Y1: DNA logos for 7462 and each fish separately.
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTAL Y1")
    print("="*80)

    # Get common length
    all_barcodes = barcodes_7462.copy()
    for fish_bcs in barcodes_per_fish.values():
        all_barcodes.extend(fish_bcs)

    length = get_barcode_length(all_barcodes)
    print(f"\nBarcode length: {length}bp")

    # Filter 7462
    bc_7462 = filter_to_length(barcodes_7462, length)
    print(f"7462: {len(bc_7462):,} barcodes")

    # Filter per fish
    bc_per_fish = {}
    for fish in all_fish:
        bc_fish = filter_to_length(barcodes_per_fish[fish], length)
        bc_per_fish[fish] = bc_fish
        print(f"{fish}: {len(bc_fish):,} barcodes")

    # Calculate frequencies
    freqs_7462 = calculate_positional_frequencies(bc_7462, length)
    freqs_per_fish = {fish: calculate_positional_frequencies(bc_per_fish[fish], length)
                      for fish in all_fish}

    # Plot - 3 rows × 4 columns for 12 fish
    n_fish = len(all_fish)
    n_cols = 4
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

    # Plot fish in 3x4 grid
    for idx, fish in enumerate(all_fish):
        row = idx // 4  # Rows 0, 1, 2
        col = idx % 4   # Columns 0, 1, 2, 3
        fish_name = get_fish_display_name(fish)
        plot_dna_logo(freqs_per_fish[fish],
                     f'{fish_name} (n={len(bc_per_fish[fish]):,})',
                     axes[row, col])

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as PNG
    output_png = f'{output_prefix}_SuppY1_DNA_logos_per_fish_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_SuppY1_DNA_logos_per_fish_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"Saved SVG: {output_svg}")

    plt.close()


def hamming_distance(s1, s2):
    """Calculate Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def calculate_pairwise_hamming(barcodes, n_pairs=20000):
    """
    Calculate pairwise Hamming distances for a set of barcodes.

    Highly optimized version using numpy view arrays for maximum speed.
    """
    if len(barcodes) < 2:
        return np.array([])

    # Convert barcodes to numpy array with dtype for single characters
    # This is much faster than converting to list first
    barcode_length = len(barcodes[0])
    bc_array = np.frombuffer(''.join(barcodes).encode('ascii'), dtype='S1').reshape(len(barcodes), barcode_length)

    n_total_pairs = len(barcodes) * (len(barcodes) - 1) // 2

    if n_total_pairs <= n_pairs:
        # Calculate all pairs using vectorized operations
        distances = []
        for i in range(len(barcodes) - 1):
            # Compare barcode i with all subsequent barcodes at once
            diffs = bc_array[i] != bc_array[i+1:]
            pair_distances = np.sum(diffs, axis=1)
            distances.extend(pair_distances)
        return np.array(distances)
    else:
        # Random sampling with pre-generated indices (much faster)
        # Generate all random pairs at once
        indices = np.random.randint(0, len(barcodes), size=(n_pairs, 2))
        # Ensure idx1 != idx2
        mask = indices[:, 0] == indices[:, 1]
        while np.any(mask):
            indices[mask, 1] = np.random.randint(0, len(barcodes), size=np.sum(mask))
            mask = indices[:, 0] == indices[:, 1]

        # Vectorized distance calculation for all pairs at once
        bc1 = bc_array[indices[:, 0]]
        bc2 = bc_array[indices[:, 1]]
        distances = np.sum(bc1 != bc2, axis=1)
        return distances


def generate_random_barcodes(n, length):
    """Generate random barcodes of specified length."""
    nucleotides = ['A', 'C', 'G', 'T']
    barcodes = []
    for _ in range(n):
        bc = ''.join(np.random.choice(nucleotides, length))
        barcodes.append(bc)
    return barcodes


def generate_supp_y3(barcodes_7462, barcodes_per_fish, all_fish, output_prefix, n_pairs=10000):
    """
    Generate Supplemental Y3: Pairwise Hamming distance violin plots.
    Includes 7462, each fish, and random barcodes.
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTAL Y3")
    print("="*80)

    # Get common length
    all_barcodes = barcodes_7462.copy()
    for fish_bcs in barcodes_per_fish.values():
        all_barcodes.extend(fish_bcs)

    length = get_barcode_length(all_barcodes)
    print(f"\nBarcode length: {length}bp")
    print(f"Sampling {n_pairs:,} pairs per sample")

    # Filter to common length
    bc_7462 = filter_to_length(barcodes_7462, length)
    bc_per_fish = {fish: filter_to_length(barcodes_per_fish[fish], length)
                   for fish in all_fish}

    # Generate random barcodes (same number as 7462 for comparison)
    bc_random = generate_random_barcodes(len(bc_7462), length)

    # Calculate pairwise distances
    print("\nCalculating pairwise Hamming distances...")

    print("  7462...")
    dist_7462 = calculate_pairwise_hamming(bc_7462, n_pairs)

    dist_per_fish = {}
    for fish in all_fish:
        print(f"  {fish}...")
        dist_per_fish[fish] = calculate_pairwise_hamming(bc_per_fish[fish], n_pairs)

    print("  Random barcodes...")
    dist_random = calculate_pairwise_hamming(bc_random, n_pairs)

    # Print statistics
    print("\n" + "="*80)
    print("PAIRWISE HAMMING DISTANCE STATISTICS")
    print("="*80)
    print()
    print(f"{'Sample':<20} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)

    stats_data = []

    # 7462
    if len(dist_7462) > 0:
        print(f"{'7462 Library':<20} {np.mean(dist_7462):>10.2f} {np.median(dist_7462):>10.2f} "
              f"{np.std(dist_7462):>10.2f} {np.min(dist_7462):>10.0f} {np.max(dist_7462):>10.0f}")
        stats_data.append({
            'sample': '7462 Library',
            'mean': np.mean(dist_7462),
            'median': np.median(dist_7462),
            'std': np.std(dist_7462)
        })

    # Each fish
    for fish in all_fish:
        if len(dist_per_fish[fish]) > 0:
            print(f"{fish:<20} {np.mean(dist_per_fish[fish]):>10.2f} "
                  f"{np.median(dist_per_fish[fish]):>10.2f} {np.std(dist_per_fish[fish]):>10.2f} "
                  f"{np.min(dist_per_fish[fish]):>10.0f} {np.max(dist_per_fish[fish]):>10.0f}")
            stats_data.append({
                'sample': fish,
                'mean': np.mean(dist_per_fish[fish]),
                'median': np.median(dist_per_fish[fish]),
                'std': np.std(dist_per_fish[fish])
            })

    # Random
    if len(dist_random) > 0:
        print(f"{'Random barcodes':<20} {np.mean(dist_random):>10.2f} "
              f"{np.median(dist_random):>10.2f} {np.std(dist_random):>10.2f} "
              f"{np.min(dist_random):>10.0f} {np.max(dist_random):>10.0f}")
        stats_data.append({
            'sample': 'Random',
            'mean': np.mean(dist_random),
            'median': np.median(dist_random),
            'std': np.std(dist_random)
        })

    # Save statistics to CSV
    df_stats = pd.DataFrame(stats_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'{output_prefix}_SuppY3_hamming_statistics_{timestamp}.csv'
    df_stats.to_csv(csv_file, index=False)
    print(f"\nStatistics saved: {csv_file}")

    # Prepare data for violin plot
    plot_data = []

    # 7462
    for d in dist_7462:
        plot_data.append({'Sample': 'Injected\nLibrary', 'Distance': d})

    # Each fish (with display names)
    for fish in all_fish:
        fish_name = get_fish_display_name(fish)
        for d in dist_per_fish[fish]:
            plot_data.append({'Sample': fish_name, 'Distance': d})

    # Random
    for d in dist_random:
        plot_data.append({'Sample': 'Random\nBarcodes', 'Distance': d})

    df_plot = pd.DataFrame(plot_data)

    # Plot violin plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Order: 7462, all fish (with display names), random
    fish_display_names = [get_fish_display_name(fish) for fish in all_fish]
    order = ['Injected\nLibrary'] + fish_display_names + ['Random\nBarcodes']

    # Grayscale colors: dark gray for library, medium gray for fish, light gray for random
    palette = ['#404040'] + ['#808080']*len(all_fish) + ['#C0C0C0']

    sns.violinplot(data=df_plot, x='Sample', y='Distance', order=order,
                   palette=palette,
                   ax=ax, inner='box', cut=0)

    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Pairwise Hamming Distance', fontsize=12)
    ax.set_title(f'Pairwise Hamming Distance Distribution ({length}bp barcodes)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    # Add mean value labels above each violin
    for i, sample_name in enumerate(order):
        sample_data = df_plot[df_plot['Sample'] == sample_name]['Distance']
        if len(sample_data) > 0:
            mean_val = sample_data.mean()
            y_max = sample_data.max()
            ax.text(i, y_max * 1.02, f'{mean_val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save as PNG
    output_png = f'{output_prefix}_SuppY3_hamming_violin_{timestamp}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved PNG: {output_png}")

    # Save as SVG
    output_svg = f'{output_prefix}_SuppY3_hamming_violin_{timestamp}.svg'
    plt.savefig(output_svg, bbox_inches='tight')
    print(f"Saved SVG: {output_svg}")

    plt.close()


def main():
    if len(sys.argv) < 3:
        print("ERROR: Missing required arguments")
        print()
        print("Usage:")
        print("  python analyze_sequence_bias_manuscript.py <hc_7462.pkl> <hc_6978.pkl>")
        print()
        print("Arguments:")
        print("  hc_7462.pkl    Filtered 7462 barcodes")
        print("  hc_6978.pkl    Filtered 6978 barcodes")
        print()
        print("Generates:")
        print("  - Fig 3D: DNA logos for 7462 and combined 6978")
        print("  - Supp Y1: DNA logos for 7462 and each fish")
        print("  - Supp Y3: Pairwise Hamming distance violin plots")
        print()
        print("Example:")
        print("  python analyze_sequence_bias_manuscript.py high_confidence_barcodes_7462*.pkl high_confidence_barcodes_6978*.pkl")
        return

    hc_7462_file = sys.argv[1]
    hc_6978_file = sys.argv[2]

    # Load data
    barcodes_7462, barcodes_6978_combined, barcodes_per_fish, all_fish = load_filtered_barcodes(
        hc_7462_file, hc_6978_file
    )

    output_prefix = "manuscript_figures"

    # Generate figures
    generate_figure_3d(barcodes_7462, barcodes_6978_combined, output_prefix)
    generate_supp_y1(barcodes_7462, barcodes_per_fish, all_fish, output_prefix)
    generate_supp_y3(barcodes_7462, barcodes_per_fish, all_fish, output_prefix, n_pairs=20000)

    print("\n" + "="*80)
    print("ALL MANUSCRIPT FIGURES GENERATED!")
    print("="*80)


if __name__ == "__main__":
    main()
