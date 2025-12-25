#!/usr/bin/env python3
"""
Unified Barcode Extraction for 6978 and 7462 samples

After unified demultiplexing, both sample types have the same read structure:
...[12bp_anchor_before][RANDOM_15bp][CACGC]...

Anchor sequences:
- Before (12bp): AGCCCCCAGGGA (allow up to 20% mismatches/indels)
- After (5bp): CACGC (exact match required)

Expected barcode positions:
- 7462: Random barcode starts around 30-35bp from read start
- 6978 (post-RC): Random barcode starts around 200-210bp from read start

Algorithm:
1. Find anchor_before (12bp, allow up to 2-3 mismatches/indels)
2. Verify barcode position is in expected range (30-35bp for 7462, 195-215bp for 6978)
3. Try 15bp barcode first, check if CACGC appears exactly after
4. If not, try 14bp, then 16bp
5. Require exact match of CACGC after barcode
"""

import time
import pickle
from collections import defaultdict
from datetime import datetime


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


def hamming_distance(s1, s2):
    """Fast Hamming distance (substitutions only, same length)"""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def find_anchor_progressive(seq, anchor, expected_pos, search_window=2, max_distance=None):
    """
    OPTIMIZED: Find anchor sequence using cascading search strategy.

    Strategy:
    1. Exact match search (fastest)
    2. Hamming distance ≤1 (substitutions only, faster than Levenshtein)
    3. Hamming distance ≤2
    4. Levenshtein distance ≤2 (allows indels, slower)
    ... up to max_distance

    Args:
        seq: Sequence to search
        anchor: Anchor sequence to find
        expected_pos: Expected position of anchor start
        search_window: Number of bp to search on each side of expected_pos (default: 2)
        max_distance: Maximum distance (default: 20% of anchor length, max 3)

    Returns: (position, distance) or (None, None) if not found
    """
    if max_distance is None:
        max_distance = min(3, max(1, int(len(anchor) * 0.2)))  # Cap at 3 for speed

    anchor_len = len(anchor)
    search_start = max(0, expected_pos - search_window)
    search_end = min(len(seq) - anchor_len + 1, expected_pos + search_window + 1)

    # Stage 1: Exact match (fastest - use string search)
    search_region = seq[search_start:search_end + anchor_len]
    if anchor in search_region:
        pos = seq.index(anchor, search_start)
        if pos < search_end:
            return pos, 0

    # Stage 2: Hamming distance (substitutions only - fast)
    for target_dist in [1, 2]:
        if target_dist > max_distance:
            break

        for pos in range(search_start, search_end):
            window = seq[pos:pos + anchor_len]
            if len(window) < anchor_len:
                continue

            # Fast Hamming distance check
            hdist = hamming_distance(window, anchor)
            if hdist == target_dist:
                return pos, hdist

    # Stage 3: Levenshtein distance (allows indels - slower, only if needed)
    if max_distance >= 2:
        for target_dist in range(2, max_distance + 1):
            for pos in range(search_start, search_end):
                window = seq[pos:pos + anchor_len]
                if len(window) < anchor_len:
                    continue

                dist = levenshtein_distance(window, anchor)
                if dist == target_dist:
                    return pos, dist

    return None, None


def extract_random_barcode_unified(seq, expected_barcode_start, sample_type="unknown"):
    """
    Extract random barcode from sequence using unified algorithm.

    Read structure (applies to both 6978-RC and 7462):
    ...[AGCCCCCAGGGA][RANDOM_15bp][CACGC]...
         12bp before               5bp after (exact match)

    Algorithm:
    1. Find before anchor (12bp, allow up to 2-3 mismatches)
    2. Verify barcode starts in expected position range
    3. Try 15bp barcode, check if CACGC appears exactly after
    4. If not found, try 14bp barcode
    5. If not found, try 16bp barcode

    Args:
        seq: Read sequence
        expected_barcode_start: Expected position where barcode starts
        sample_type: Type of sample (for position validation)

    Returns: (random_barcode, extraction_info) or (None, failure_reason)
    """
    # Define anchors
    anchor_before = "AGCCCCCAGGGA"  # 12bp before barcode
    strict_anchor_check = "CACGC"   # 5bp strict check after barcode

    # For 6978: search from the END of the read (barcode is ~53bp from end, includes 5bp sample barcode at end)
    # For 7462: search from the START of the read (barcode is ~30-35bp from start)
    if sample_type == "6978":
        # For 6978, barcode should be ~53bp from end (range: 50-58bp from end)
        # Structure from end: [5bp sample barcode][~28bp][CACGC 5bp][random barcode 15bp][AGCCCCCAGGGA 12bp]
        # This means barcode_start should be at: len(seq) - 53 ± a few bases
        # Expected anchor before position from end: 53 + 15 = 68bp from end
        # Or in absolute terms: len(seq) - 68
        min_seq_len = 75  # Need at least ~75bp for the barcode region
        if len(seq) < min_seq_len:
            return None, "sequence_too_short"

        # Search for anchor from the end
        # Barcode is ~53bp from end, so barcode_start = len(seq) - 53
        # Before anchor starts 12bp before that: len(seq) - 53 - 12 = len(seq) - 65
        expected_before_pos_from_end = 65  # Distance from end to before anchor start
        expected_before_pos = len(seq) - expected_before_pos_from_end

        # Search for before anchor (window: ±5bp to be more flexible)
        before_pos, before_dist = find_anchor_progressive(
            seq, anchor_before, expected_before_pos, search_window=5
        )

        if before_pos is None:
            return None, "anchor_before_not_found"

        barcode_start = before_pos + len(anchor_before)

        # Validate position from end (barcode should start 50-58bp from end)
        distance_from_end = len(seq) - barcode_start
        if not (50 <= distance_from_end <= 58):
            return None, f"barcode_position_out_of_range (found {distance_from_end}bp from end, expected 50-58)"

    else:
        # For 7462: search from start (barcode at ~30-35bp from start)
        min_seq_len = expected_barcode_start + 15 + len(strict_anchor_check)
        if len(seq) < min_seq_len:
            return None, "sequence_too_short"

        expected_before_pos = expected_barcode_start - len(anchor_before)

        # Search for before anchor (small window: ±2bp)
        before_pos, before_dist = find_anchor_progressive(
            seq, anchor_before, expected_before_pos, search_window=2
        )

        if before_pos is None:
            return None, "anchor_before_not_found"

        barcode_start = before_pos + len(anchor_before)

        # Validate position from start (barcode should start at 30-35bp)
        if not (30 <= barcode_start <= 35):
            return None, f"barcode_position_out_of_range (found at {barcode_start}, expected 30-35)"

    # Try different barcode lengths: 15bp (preferred), then 14bp, then 16bp
    for barcode_len in [15, 14, 16]:
        barcode_end = barcode_start + barcode_len

        # Check if we have enough sequence for strict anchor check (5bp)
        if barcode_end + len(strict_anchor_check) > len(seq):
            continue

        # Extract barcode
        barcode = seq[barcode_start:barcode_end]

        # Validate barcode
        if 'N' in barcode:
            continue

        # Must have at least 2 different bases (avoid homopolymers)
        if len(set(barcode)) < 2:
            continue

        # STRICT VALIDATION: Require exact match of CACGC after the barcode
        # Extract the sequence right after the barcode
        after_sequence = seq[barcode_end:barcode_end + len(strict_anchor_check)]

        # Require exact match
        if after_sequence != strict_anchor_check:
            continue

        # Success! We found both anchors with this barcode length
        extraction_info = {
            'barcode_length': barcode_len,
            'before_anchor_distance': before_dist,
            'barcode_start': barcode_start,
            'barcode_end': barcode_end,
            'before_anchor_pos': before_pos
        }

        return barcode, extraction_info

    # None of the barcode lengths worked
    return None, "no_valid_barcode_length"


def extract_barcodes_from_unified(unified_pickle_file,
                                   output_prefix="extracted_barcodes"):
    """
    Extract random barcodes from unified (demultiplexed/processed) reads.

    Args:
        unified_pickle_file: Output from unified_demultiplex.py
        output_prefix: Prefix for output files

    Returns: (results_dict, pickle_file_path)
    """
    print("="*70)
    print("UNIFIED BARCODE EXTRACTION")
    print("="*70)
    print(f"Input: {unified_pickle_file}")
    print(f"Output prefix: {output_prefix}")
    print()

    # Load unified data
    print("Loading processed reads...")
    with open(unified_pickle_file, 'rb') as f:
        unified_data = pickle.load(f)

    reads_by_sample = unified_data['reads_by_sample']
    sample_type = unified_data.get('sample_type', 'unknown')
    orientation = unified_data.get('orientation', 'unknown')

    print(f"Sample type: {sample_type}")
    print(f"Orientation: {orientation}")
    print(f"Samples loaded: {len(reads_by_sample)}")
    print()

    # Determine expected barcode position based on sample type
    # For 6978: R2 reads after RC, barcode is ~53bp from END of read (includes 5bp sample barcode)
    # For 7462: R1 reads, barcode is ~33bp from START of read
    if sample_type == '7462':
        expected_barcode_start = 33  # Around 30-35bp (after 12bp anchor at ~21bp)
        print(f"Expected barcode position: ~{expected_barcode_start}bp from read start (7462 R1)")
    elif sample_type == '6978':
        expected_barcode_start = None  # For 6978, we search from the end (not used)
        print(f"Expected barcode position: ~53bp from read END (6978 R2-RC, includes 5bp sample barcode)")
    else:
        # Default to 7462 position
        expected_barcode_start = 33
        print(f"Unknown sample type, using default position: ~{expected_barcode_start}bp")
    print()

    print("Anchor sequences:")
    print(f"  Before (12bp): AGCCCCCAGGGA (flexible: allow up to 2-3bp mismatches)")
    print(f"  After (5bp): CACGC (strict: exact match required)")
    print(f"  Barcode lengths tried: 15bp, 14bp, 16bp")
    print(f"  Position validation:")
    if sample_type == '7462':
        print(f"    Barcode must start at 30-35bp from read start")
    elif sample_type == '6978':
        print(f"    Barcode must start at 50-58bp from read END")
    else:
        print(f"    Default: 30-35bp from start")
    print()

    # Extract barcodes
    sample_barcodes = defaultdict(lambda: defaultdict(int))
    overall_barcode_counts = defaultdict(int)
    sample_read_counts = defaultdict(int)

    # Statistics per sample
    sample_stats = {}

    total_reads = 0
    successful_extractions = 0
    failed_extractions = 0

    # Track barcode lengths
    barcode_lengths = defaultdict(int)

    # Track failure reasons
    failure_reasons = defaultdict(int)

    # Track anchor distances
    anchor_before_distances = defaultdict(int)
    # After anchor requires exact match (CACGC), no distance tracking

    start_time = time.time()

    print("Extracting random barcodes...")
    print()

    for sample, reads in reads_by_sample.items():
        sample_total = len(reads)
        sample_success = 0
        sample_fail = 0

        print(f"  Processing {sample}: {sample_total:,} reads...")

        for read_info in reads:
            total_reads += 1
            seq = read_info['seq']

            result = extract_random_barcode_unified(seq, expected_barcode_start, sample_type)

            if result[0]:  # Successfully extracted
                random_bc, extraction_info = result
                sample_barcodes[sample][random_bc] += 1
                overall_barcode_counts[random_bc] += 1
                sample_read_counts[sample] += 1
                barcode_lengths[extraction_info['barcode_length']] += 1
                anchor_before_distances[extraction_info['before_anchor_distance']] += 1
                # After anchor is exact match (CACGC), no distance tracking needed
                successful_extractions += 1
                sample_success += 1
            else:
                failure_reason = result[1]
                failure_reasons[failure_reason] += 1
                failed_extractions += 1
                sample_fail += 1

            # Progress indicator
            if total_reads % 100000 == 0:
                elapsed = time.time() - start_time
                rate = total_reads / elapsed
                success_rate = (successful_extractions / total_reads * 100) if total_reads > 0 else 0
                print(f"    {total_reads:,} reads processed ({rate:,.0f}/sec, {success_rate:.1f}% success)")

        sample_stats[sample] = {
            'total_reads': sample_total,
            'successful': sample_success,
            'failed': sample_fail,
            'success_rate': (sample_success / sample_total * 100) if sample_total > 0 else 0,
            'unique_barcodes': len(sample_barcodes[sample])
        }

    elapsed = time.time() - start_time
    overall_success_rate = (successful_extractions / total_reads * 100) if total_reads > 0 else 0

    print()
    print("="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Total reads: {total_reads:,}")
    print(f"Successfully extracted: {successful_extractions:,} ({overall_success_rate:.1f}%)")
    print(f"Failed extractions: {failed_extractions:,} ({failed_extractions/total_reads*100:.1f}%)")
    print(f"Processing time: {elapsed:.1f}s ({total_reads / elapsed:,.0f} reads/sec)")
    print()

    print(f"Samples processed: {len(sample_stats)}")
    print(f"Total unique random barcodes: {len(overall_barcode_counts):,}")
    print()

    print("Barcode length distribution:")
    for length in sorted(barcode_lengths.keys()):
        count = barcode_lengths[length]
        pct = count / successful_extractions * 100
        print(f"  {length}bp: {count:,} ({pct:.1f}%)")
    print()

    print("Anchor matching statistics:")
    print("  Before anchor distance:")
    for dist in sorted(anchor_before_distances.keys()):
        count = anchor_before_distances[dist]
        pct = count / successful_extractions * 100
        print(f"    {dist} mismatches/indels: {count:,} ({pct:.1f}%)")
    print("  After anchor: Exact match required (CACGC)")
    print()

    if failure_reasons:
        print("Failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / failed_extractions * 100 if failed_extractions > 0 else 0
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
        print()

    print("Per-sample results:")
    for sample in sorted(sample_stats.keys()):
        stats = sample_stats[sample]
        print(f"  {sample}:")
        print(f"    Total reads: {stats['total_reads']:,}")
        print(f"    Successful: {stats['successful']:,} ({stats['success_rate']:.1f}%)")
        print(f"    Failed: {stats['failed']:,}")
        print(f"    Unique barcodes: {stats['unique_barcodes']:,}")

    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'sample_barcodes': dict(sample_barcodes),
        'overall_barcode_counts': dict(overall_barcode_counts),
        'sample_read_counts': dict(sample_read_counts),
        'sample_stats': sample_stats,
        'barcode_lengths': dict(barcode_lengths),
        'failure_reasons': dict(failure_reasons),
        'anchor_before_distances': dict(anchor_before_distances),
        # After anchor requires exact match, no distances tracked
        'stats': {
            'total_reads': total_reads,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': overall_success_rate,
            'processing_time': elapsed,
            'samples_processed': len(sample_stats),
            'unique_barcodes': len(overall_barcode_counts)
        },
        'timestamp': timestamp,
        'source_file': unified_pickle_file,
        'sample_type': sample_type,
        'expected_barcode_start': expected_barcode_start
    }

    # Save to pickle
    pickle_file = f"{output_prefix}_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)

    print()
    print("="*70)
    print("RESULTS SAVED")
    print("="*70)
    print(f"Pickle file: {pickle_file}")
    print()
    print("This file contains:")
    print("  - sample_barcodes: Dict[sample -> Dict[barcode -> count]]")
    print("  - overall_barcode_counts: Dict[barcode -> total_count]")
    print("  - sample_read_counts: Dict[sample -> total_reads]")
    print("  - sample_stats: Per-sample extraction statistics")
    print("  - barcode_lengths: Length distribution of extracted barcodes")
    print("  - failure_reasons: Why extractions failed")
    print("  - anchor_*_distances: Quality of anchor matching")
    print("  - stats: Overall statistics")
    print()

    return results, pickle_file


if __name__ == "__main__":
    import sys
    import glob

    print("="*70)
    print("UNIFIED BARCODE EXTRACTION")
    print("="*70)
    print()

    # Get input file from command line or find most recent
    if len(sys.argv) > 1:
        unified_pickle_file = sys.argv[1]
    else:
        # Find most recent unified processed file
        patterns = [
            "demultiplexed_6978_*.pkl",
            "processed_7462_*.pkl"
        ]

        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(pattern))

        if all_files:
            unified_pickle_file = sorted(all_files, reverse=True)[0]
            print(f"Using most recent processed file: {unified_pickle_file}")
            print()
        else:
            print("ERROR: No processed file found")
            print()
            print("Expected files:")
            print("  - demultiplexed_6978_*.pkl (from unified_demultiplex.py)")
            print("  - processed_7462_*.pkl (from unified_demultiplex.py)")
            print()
            print("Usage:")
            print("  python unified_extract_barcodes.py <unified_pickle_file>")
            sys.exit(1)

    # Determine output prefix based on input file
    if '6978' in unified_pickle_file:
        output_prefix = "extracted_barcodes_6978"
    elif '7462' in unified_pickle_file:
        output_prefix = "extracted_barcodes_7462"
    else:
        output_prefix = "extracted_barcodes"

    try:
        results, pickle_file = extract_barcodes_from_unified(
            unified_pickle_file,
            output_prefix=output_prefix
        )

        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Extracted barcodes saved to: {pickle_file}")
        print()
        print("Next steps:")
        print("  1. Repeat for other sample type if needed")
        print("  2. Run unified_collapse_barcodes.py to collapse variants")

    except FileNotFoundError:
        print(f"ERROR: File '{unified_pickle_file}' not found")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
