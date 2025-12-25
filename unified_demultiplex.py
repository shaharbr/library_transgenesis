#!/usr/bin/env python3
"""
Unified Demultiplexing for 6978 and 7462 samples

This script handles both:
1. 6978 (R2): Demultiplexes by sample barcode, then reverse complements reads
2. 7462 (R1): Processes as single sample (no demultiplexing needed)

After processing, both samples have reads in the same orientation (forward)
suitable for unified barcode extraction.

6978 R2 structure (before RC):
[SAMPLE_5bp][TAGCGGGGGAGGGACGTAATTACAGCTAGCGTG][RANDOM_15bp][TCCCTGG...]

After reverse complement, matches 7462 R1 structure:
...CAGGGA[RANDOM_15bp]CACGCTAGC...

7462 R1 structure (no change needed):
...CAGGGA[RANDOM_15bp]CACGCTAGC...
"""

import time
import pickle
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from datetime import datetime

# Sample barcodes for 6978
SAMPLE_BARCODES = {
    'ACCTT', 'AATCG', 'AGAGA', 'CGGAG', 'CTACT', 'GAAAC',
    'GCGCA', 'CCTGC', 'TAGGT', 'GTCGG', 'TTTAA', 'TGCCC'
}


def reverse_complement(seq):
    """Return reverse complement of DNA sequence"""
    return str(Seq(seq).reverse_complement())


def identify_sample_barcode(seq):
    """
    Identify sample barcode from first 5bp of read (for 6978 only)

    Returns: sample_barcode or None or "AMBIGUOUS"
    """
    if len(seq) < 5:
        return None

    sample_bc = seq[0:5]

    # Check for N's
    if 'N' in sample_bc:
        return None

    # Exact match
    if sample_bc in SAMPLE_BARCODES:
        return sample_bc

    # Error correction: Hamming distance ≤1
    matches = []
    for valid_bc in SAMPLE_BARCODES:
        mismatches = sum(1 for a, b in zip(sample_bc, valid_bc) if a != b)
        if mismatches == 1:
            matches.append(valid_bc)

    if len(matches) == 1:
        # Unique match - error correct
        return matches[0]
    elif len(matches) > 1:
        # Ambiguous - multiple possible corrections
        return "AMBIGUOUS"
    else:
        # No valid match within distance 1
        return None


def process_6978_sample(fastq_file, output_prefix="demultiplexed_6978"):
    """
    Process 6978 sample:
    1. Demultiplex by sample barcode
    2. Reverse complement all reads
    3. Output in unified format

    Returns: (results_dict, pickle_file_path)
    """
    print("="*70)
    print("PROCESSING 6978 SAMPLE (DEMULTIPLEX + REVERSE COMPLEMENT)")
    print("="*70)
    print(f"Input: {fastq_file}")
    print(f"Output prefix: {output_prefix}")
    print()

    # Store reads by sample
    reads_by_sample = defaultdict(list)

    # Statistics
    total_reads = 0
    demultiplexed_reads = 0
    ambiguous_barcodes = 0
    failed_demux = 0
    sample_counts = defaultdict(int)

    start_time = time.time()

    print("Processing FASTQ file...")
    print("  - Identifying sample barcodes")
    print("  - Reverse complementing reads")
    print()

    for record in SeqIO.parse(fastq_file, "fastq"):
        total_reads += 1

        if total_reads % 100000 == 0:
            elapsed = time.time() - start_time
            rate = total_reads / elapsed
            success_rate = (demultiplexed_reads / total_reads * 100) if total_reads > 0 else 0
            print(f"  Processed {total_reads:,} reads ({rate:,.0f}/sec, {success_rate:.1f}% demultiplexed)")

        read_seq = str(record.seq).upper()
        sample_bc = identify_sample_barcode(read_seq)

        if sample_bc == "AMBIGUOUS":
            ambiguous_barcodes += 1
        elif sample_bc:
            # REVERSE COMPLEMENT THE READ
            rc_seq = reverse_complement(read_seq)
            rc_qual = record.letter_annotations['phred_quality'][::-1]  # Reverse quality scores

            # Store read info with RC sequence
            reads_by_sample[sample_bc].append({
                'id': record.id,
                'seq': rc_seq,  # Store RC sequence
                'qual': rc_qual,
                'original_seq': read_seq  # Keep original for debugging if needed
            })
            sample_counts[sample_bc] += 1
            demultiplexed_reads += 1
        else:
            failed_demux += 1

    elapsed = time.time() - start_time
    success_rate = demultiplexed_reads / total_reads * 100

    print()
    print("="*70)
    print("6978 PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total reads: {total_reads:,}")
    print(f"Successfully demultiplexed: {demultiplexed_reads:,} ({success_rate:.1f}%)")
    print(f"Ambiguous sample barcodes: {ambiguous_barcodes:,} ({ambiguous_barcodes/total_reads*100:.2f}%)")
    print(f"Failed demultiplexing: {failed_demux:,} ({failed_demux/total_reads*100:.1f}%)")
    print(f"Processing time: {elapsed:.1f}s ({total_reads / elapsed:,.0f} reads/sec)")
    print()

    print(f"Samples detected: {len(sample_counts)}")
    print()

    print("Sample distribution:")
    for sample, count in sorted(sample_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / demultiplexed_reads * 100
        print(f"  {sample}: {count:,} reads ({percentage:.1f}%)")

    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'reads_by_sample': dict(reads_by_sample),
        'sample_counts': dict(sample_counts),
        'stats': {
            'total_reads': total_reads,
            'demultiplexed_reads': demultiplexed_reads,
            'ambiguous_barcodes': ambiguous_barcodes,
            'failed_demux': failed_demux,
            'success_rate': success_rate,
            'processing_time': elapsed,
            'samples_detected': len(sample_counts)
        },
        'timestamp': timestamp,
        'source_file': fastq_file,
        'sample_type': '6978',
        'orientation': 'forward (reverse complemented from R2)'
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
    print("  - reads_by_sample: Dict[sample -> List[read_info]]")
    print("  - All reads are REVERSE COMPLEMENTED (forward orientation)")
    print("  - sample_counts: Dict[sample -> read_count]")
    print("  - stats: Demultiplexing statistics")
    print()

    # Save summary CSV
    summary_file = f"{output_prefix}_summary_{timestamp}.csv"
    summary_data = []
    for sample in sorted(sample_counts.keys()):
        summary_data.append({
            'Sample': sample,
            'Total_Reads': sample_counts[sample],
            'Percent_of_Total': f"{sample_counts[sample]/demultiplexed_reads*100:.2f}%"
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    print(f"Summary CSV: {summary_file}")
    print()

    return results, pickle_file


def process_7462_sample(fastq_file, sample_name="7462", output_prefix="processed_7462"):
    """
    Process 7462 sample:
    1. No demultiplexing needed (single sample)
    2. No reverse complement needed (already forward orientation)
    3. Output in unified format (same structure as 6978)

    Returns: (results_dict, pickle_file_path)
    """
    print("="*70)
    print("PROCESSING 7462 SAMPLE (NO DEMULTIPLEX)")
    print("="*70)
    print(f"Input: {fastq_file}")
    print(f"Sample name: {sample_name}")
    print(f"Output prefix: {output_prefix}")
    print()

    # Store all reads under single sample name
    reads_by_sample = {sample_name: []}

    # Statistics
    total_reads = 0

    start_time = time.time()

    print("Processing FASTQ file...")
    print("  - No demultiplexing (single sample)")
    print("  - No reverse complement (already forward orientation)")
    print()

    for record in SeqIO.parse(fastq_file, "fastq"):
        total_reads += 1

        if total_reads % 100000 == 0:
            elapsed = time.time() - start_time
            rate = total_reads / elapsed
            print(f"  Processed {total_reads:,} reads ({rate:,.0f}/sec)")

        read_seq = str(record.seq).upper()

        # Store read info (no modification needed)
        reads_by_sample[sample_name].append({
            'id': record.id,
            'seq': read_seq,
            'qual': record.letter_annotations['phred_quality']
        })

    elapsed = time.time() - start_time

    print()
    print("="*70)
    print("7462 PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total reads: {total_reads:,}")
    print(f"Processing time: {elapsed:.1f}s ({total_reads / elapsed:,.0f} reads/sec)")
    print()

    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'reads_by_sample': reads_by_sample,
        'sample_counts': {sample_name: total_reads},
        'stats': {
            'total_reads': total_reads,
            'processing_time': elapsed,
            'samples_detected': 1
        },
        'timestamp': timestamp,
        'source_file': fastq_file,
        'sample_type': '7462',
        'orientation': 'forward (original R1)'
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
    print("  - reads_by_sample: Dict['{sample_name}' -> List[read_info]]")
    print("  - All reads are in FORWARD orientation (original R1)")
    print("  - sample_counts: Dict[sample -> read_count]")
    print("  - stats: Processing statistics")
    print()

    # Save summary CSV
    summary_file = f"{output_prefix}_summary_{timestamp}.csv"
    summary_data = [{
        'Sample': sample_name,
        'Total_Reads': total_reads,
        'Percent_of_Total': '100.00%'
    }]

    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    print(f"Summary CSV: {summary_file}")
    print()

    return results, pickle_file


if __name__ == "__main__":
    import sys

    print("="*70)
    print("UNIFIED SAMPLE PROCESSING")
    print("="*70)
    print()
    print("This script processes both 6978 and 7462 samples into a unified format")
    print("suitable for the same barcode extraction pipeline.")
    print()
    print("Usage:")
    print("  python unified_demultiplex.py 6978 <fastq_file>")
    print("  python unified_demultiplex.py 7462 <fastq_file>")
    print()

    if len(sys.argv) < 3:
        print("Processing default files...")
        print()

        # Process 6978
        print("\n" + "="*70)
        print("PROCESSING 6978 SAMPLE")
        print("="*70)
        fastq_6978 = "6978_001_S58_R2_001.fastq"
        try:
            results_6978, pickle_6978 = process_6978_sample(fastq_6978)
            print(f"\n✓ 6978 processing complete: {pickle_6978}")
        except FileNotFoundError:
            print(f"✗ File not found: {fastq_6978}")
        except Exception as e:
            print(f"✗ Error processing 6978: {e}")

        # Process 7462
        print("\n" + "="*70)
        print("PROCESSING 7462 SAMPLE")
        print("="*70)
        fastq_7462 = "7462_001_S101_R1_001.fastq"
        try:
            results_7462, pickle_7462 = process_7462_sample(fastq_7462)
            print(f"\n✓ 7462 processing complete: {pickle_7462}")
        except FileNotFoundError:
            print(f"✗ File not found: {fastq_7462}")
        except Exception as e:
            print(f"✗ Error processing 7462: {e}")

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("Run unified_extract_barcodes.py to extract barcodes from both samples")
        print("using the unified extraction algorithm.")
    else:
        sample_type = sys.argv[1]
        fastq_file = sys.argv[2]

        if sample_type == "6978":
            results, pickle_file = process_6978_sample(fastq_file)
        elif sample_type == "7462":
            results, pickle_file = process_7462_sample(fastq_file)
        else:
            print(f"ERROR: Unknown sample type '{sample_type}'")
            print("Valid types: 6978, 7462")
            sys.exit(1)

        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Processed reads saved to: {pickle_file}")
        print()
        print("Next step: Run unified_extract_barcodes.py")
