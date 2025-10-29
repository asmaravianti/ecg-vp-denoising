"""
Visualization functions for ECG denoising and compression analysis.

This module provides plotting utilities for Week 2 deliverables:
- Rate-distortion curves (PRD-CR, WWPRD-CR)
- SNR bar charts at different compression ratios
- Reconstruction overlays with quality metrics

Author: Person B
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_rate_distortion_curves(
    results_dict: Dict[str, Dict],
    output_path: str,
    title: str = "Rate-Distortion Curves"
) -> None:
    """
    Plot PRD-CR and WWPRD-CR curves on a single figure.
    
    This is the main deliverable for Week 2 - shows the trade-off between
    compression ratio and reconstruction quality.
    
    Args:
        results_dict: Dictionary with CR as keys, each containing:
                     {'PRD': float, 'WWPRD': float, 'SNR_improvement': float, ...}
        output_path: Where to save the figure
        title: Figure title
        
    Example:
        results = {
            4: {'PRD': 35.2, 'WWPRD': 30.1, 'SNR_improvement': 5.2},
            8: {'PRD': 28.5, 'WWPRD': 24.3, 'SNR_improvement': 6.1},
            16: {'PRD': 22.1, 'WWPRD': 18.7, 'SNR_improvement': 7.8}
        }
        plot_rate_distortion_curves(results, 'rd_curves.png')
    """
    # Extract data from results dictionary
    crs = sorted(results_dict.keys())  # Compression ratios [4, 8, 16, 32]
    prds = [results_dict[cr]['PRD'] for cr in crs]
    wwprds = [results_dict[cr]['WWPRD'] for cr in crs]
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # === LEFT PLOT: PRD vs CR ===
    ax1.plot(crs, prds, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='PRD', markerfacecolor='white', 
             markeredgewidth=2)
    
    # Add clinical quality thresholds
    ax1.axhline(y=4.33, color='green', linestyle='--', alpha=0.7, 
                label='Excellent (PRD < 4.33%)')
    ax1.axhline(y=9.0, color='orange', linestyle='--', alpha=0.7, 
                label='Very Good (PRD < 9%)')
    ax1.axhline(y=15.0, color='red', linestyle='--', alpha=0.7, 
                label='Good (PRD < 15%)')
    
    # Annotate each point with its value
    for cr, prd in zip(crs, prds):
        ax1.annotate(f'{prd:.1f}%', 
                     xy=(cr, prd), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', 
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor='white', 
                              edgecolor='gray', 
                              alpha=0.8))
    
    ax1.set_xlabel('Compression Ratio (CR)', fontweight='bold')
    ax1.set_ylabel('PRD (%)', fontweight='bold')
    ax1.set_title('PRD vs Compression Ratio')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # Log scale for better visualization
    
    # === RIGHT PLOT: WWPRD vs CR ===
    ax2.plot(crs, wwprds, 's-', linewidth=2, markersize=8, 
             color='#A23B72', label='WWPRD', markerfacecolor='white', 
             markeredgewidth=2)
    
    # Add WWPRD clinical quality thresholds
    ax2.axhline(y=7.4, color='green', linestyle='--', alpha=0.7, 
                label='Excellent (WWPRD < 7.4%)')
    ax2.axhline(y=14.8, color='orange', linestyle='--', alpha=0.7, 
                label='Very Good (WWPRD < 14.8%)')
    ax2.axhline(y=24.7, color='red', linestyle='--', alpha=0.7, 
                label='Good (WWPRD < 24.7%)')
    
    # Annotate each point with its value
    for cr, wwprd in zip(crs, wwprds):
        ax2.annotate(f'{wwprd:.1f}%', 
                     xy=(cr, wwprd), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', 
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor='white', 
                              edgecolor='gray', 
                              alpha=0.8))
    
    ax2.set_xlabel('Compression Ratio (CR)', fontweight='bold')
    ax2.set_ylabel('WWPRD (%)', fontweight='bold')
    ax2.set_title('WWPRD vs Compression Ratio')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved rate-distortion curves to {output_path}")
    plt.close()


def plot_snr_bar_chart(
    results_dict: Dict[str, Dict],
    output_path: str,
    title: str = "SNR Improvement at Different Compression Ratios"
) -> None:
    """
    Create bar chart showing SNR improvement for each compression ratio.
    
    This shows how denoising quality changes with compression level.
    Higher SNR improvement = better denoising.
    
    Args:
        results_dict: Dictionary with CR as keys, each containing SNR metrics
        output_path: Where to save the figure
        title: Figure title
    """
    # Extract data
    crs = sorted(results_dict.keys())
    snr_in = [results_dict[cr].get('SNR_in', 0) for cr in crs]
    snr_out = [results_dict[cr].get('SNR_out', 0) for cr in crs]
    snr_improvement = [results_dict[cr]['SNR_improvement'] for cr in crs]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # === LEFT PLOT: Input vs Output SNR ===
    x = np.arange(len(crs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, snr_in, width, label='Input SNR (noisy)', 
                    color='#E63946', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, snr_out, width, label='Output SNR (denoised)', 
                    color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    ax1.set_xlabel('Compression Ratio', fontweight='bold')
    ax1.set_ylabel('SNR (dB)', fontweight='bold')
    ax1.set_title('Input vs Output SNR')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'CR={cr}' for cr in crs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === RIGHT PLOT: SNR Improvement ===
    bars3 = ax2.bar(x, snr_improvement, color='#457B9D', alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    # Color bars by performance (gradient)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(crs)))
    for bar, color in zip(bars3, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (bar, improvement) in enumerate(zip(bars3, snr_improvement)):
        height = bar.get_height()
        ax2.annotate(f'{improvement:.2f} dB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9,
                    fontweight='bold')
    
    ax2.set_xlabel('Compression Ratio', fontweight='bold')
    ax2.set_ylabel('SNR Improvement (dB)', fontweight='bold')
    ax2.set_title('Denoising Effectiveness by CR')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'CR={cr}' for cr in crs])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved SNR bar chart to {output_path}")
    plt.close()


def plot_reconstruction_overlay(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    metrics: Dict[str, float],
    output_path: str,
    compression_ratio: int,
    time_range: Optional[Tuple[int, int]] = None,
    fs: int = 360
) -> None:
    """
    Create detailed reconstruction overlay with zoomed QRS regions.
    
    This shows visual quality at a specific compression ratio.
    Includes full signal view + zoomed view of critical QRS complex.
    
    Args:
        clean_signal: Original clean ECG (1D array)
        noisy_signal: Noisy input ECG
        reconstructed_signal: Model's reconstruction
        metrics: Dict with PRD, WWPRD, SNR_improvement, etc.
        output_path: Where to save the figure
        compression_ratio: CR value for title
        time_range: Optional (start, end) indices for zoom
        fs: Sampling frequency (Hz)
    """
    # Convert sample indices to time (seconds)
    n_samples = len(clean_signal)
    time = np.arange(n_samples) / fs
    
    # Create figure with 2 subplots (full view + zoom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'ECG Reconstruction at CR = {compression_ratio}:1', 
                 fontsize=14, fontweight='bold')
    
    # === TOP PLOT: Full Signal ===
    ax1.plot(time, clean_signal, 'g-', linewidth=1.5, label='Clean (ground truth)', alpha=0.8)
    ax1.plot(time, noisy_signal, color='gray', linewidth=1, label='Noisy input', alpha=0.5)
    ax1.plot(time, reconstructed_signal, 'r--', linewidth=1.5, label='Reconstructed', alpha=0.9)
    
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Amplitude (normalized)', fontweight='bold')
    ax1.set_title('Full ECG Window')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = (
        f"PRD: {metrics.get('PRD', 0):.2f}%\n"
        f"WWPRD: {metrics.get('WWPRD', 0):.2f}%\n"
        f"SNR Improvement: {metrics.get('SNR_improvement', 0):.2f} dB\n"
        f"CR: {compression_ratio}:1"
    )
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # === BOTTOM PLOT: Zoomed QRS Complex ===
    if time_range is None:
        # Auto-detect QRS: find highest derivative region
        derivative = np.abs(np.diff(clean_signal))
        qrs_center = np.argmax(derivative)
        time_range = (max(0, qrs_center - 50), min(n_samples, qrs_center + 100))
    
    start, end = time_range
    time_zoom = time[start:end]
    
    ax2.plot(time_zoom, clean_signal[start:end], 'g-', linewidth=2, 
             label='Clean', alpha=0.8, marker='o', markersize=3)
    ax2.plot(time_zoom, noisy_signal[start:end], color='gray', linewidth=1.5, 
             label='Noisy', alpha=0.5)
    ax2.plot(time_zoom, reconstructed_signal[start:end], 'r--', linewidth=2, 
             label='Reconstructed', alpha=0.9, marker='s', markersize=3)
    
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Amplitude (normalized)', fontweight='bold')
    ax2.set_title('Zoomed View: QRS Complex Detail')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight the zoom region in top plot
    ax1.axvspan(time_zoom[0], time_zoom[-1], alpha=0.2, color='yellow', 
                label='Zoom region')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved reconstruction overlay to {output_path}")
    plt.close()


def plot_multiple_cr_comparison(
    signals_dict: Dict[int, Dict[str, np.ndarray]],
    metrics_dict: Dict[int, Dict[str, float]],
    output_path: str,
    fs: int = 360
) -> None:
    """
    Compare reconstructions at multiple CRs in a single figure.
    
    Shows how quality degrades with increasing compression.
    Useful for Week 2 deliverable: "Updated overlays at two CRs (e.g., 8 and 16)"
    
    Args:
        signals_dict: {CR: {'clean': array, 'noisy': array, 'reconstructed': array}}
        metrics_dict: {CR: {'PRD': float, 'WWPRD': float, ...}}
        output_path: Where to save the figure
        fs: Sampling frequency
    """
    crs = sorted(signals_dict.keys())
    n_crs = len(crs)
    
    # Create subplots (one row per CR)
    fig, axes = plt.subplots(n_crs, 1, figsize=(14, 4 * n_crs))
    if n_crs == 1:
        axes = [axes]
    
    fig.suptitle('ECG Reconstruction Quality at Different Compression Ratios', 
                 fontsize=14, fontweight='bold')
    
    for idx, (cr, ax) in enumerate(zip(crs, axes)):
        signals = signals_dict[cr]
        metrics = metrics_dict[cr]
        
        clean = signals['clean']
        noisy = signals['noisy']
        recon = signals['reconstructed']
        
        time = np.arange(len(clean)) / fs
        
        # Plot signals
        ax.plot(time, clean, 'g-', linewidth=1.5, label='Clean', alpha=0.8)
        ax.plot(time, noisy, color='gray', linewidth=1, label='Noisy', alpha=0.4)
        ax.plot(time, recon, 'r--', linewidth=1.5, label='Reconstructed', alpha=0.9)
        
        # Styling
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title(f'CR = {cr}:1 | PRD = {metrics["PRD"]:.2f}% | '
                    f'WWPRD = {metrics["WWPRD"]:.2f}% | '
                    f'SNR Improvement = {metrics["SNR_improvement"]:.2f} dB',
                    fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved multi-CR comparison to {output_path}")
    plt.close()


def create_week2_summary_figure(
    results_dict: Dict[int, Dict],
    output_path: str
) -> None:
    """
    Create comprehensive Week 2 summary figure with all key plots.
    
    This is the "show everything in one image" for your professor.
    Includes: PRD-CR, WWPRD-CR, SNR improvement, and quality classification.
    
    Args:
        results_dict: Complete results from CR sweep
        output_path: Where to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    crs = sorted(results_dict.keys())
    prds = [results_dict[cr]['PRD'] for cr in crs]
    wwprds = [results_dict[cr]['WWPRD'] for cr in crs]
    snr_improvements = [results_dict[cr]['SNR_improvement'] for cr in crs]
    
    # === Plot 1: PRD vs CR ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(crs, prds, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=9.0, color='orange', linestyle='--', alpha=0.5, label='Very Good')
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('PRD (%)')
    ax1.set_title('PRD vs CR', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # === Plot 2: WWPRD vs CR ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(crs, wwprds, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.axhline(y=14.8, color='orange', linestyle='--', alpha=0.5, label='Very Good')
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('WWPRD (%)')
    ax2.set_title('WWPRD vs CR', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # === Plot 3: SNR Improvement ===
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(crs))
    bars = ax3.bar(x, snr_improvements, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(crs))))
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('SNR Improvement (dB)')
    ax3.set_title('Denoising Effectiveness', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'CR={cr}' for cr in crs])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, snr_improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # === Plot 4: Quality Classification Table ===
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create table data
    table_data = [['CR', 'PRD (%)', 'WWPRD (%)', 'SNR Imp (dB)', 'PRD Quality', 'WWPRD Quality']]
    
    for cr in crs:
        prd = results_dict[cr]['PRD']
        wwprd = results_dict[cr]['WWPRD']
        snr = results_dict[cr]['SNR_improvement']
        
        # Classify quality
        if prd < 4.33:
            prd_qual = 'Excellent'
        elif prd < 9.0:
            prd_qual = 'Very Good'
        elif prd < 15.0:
            prd_qual = 'Good'
        else:
            prd_qual = 'Fair'
        
        if wwprd < 7.4:
            wwprd_qual = 'Excellent'
        elif wwprd < 14.8:
            wwprd_qual = 'Very Good'
        elif wwprd < 24.7:
            wwprd_qual = 'Good'
        else:
            wwprd_qual = 'Fair'
        
        table_data.append([f'{cr}:1', f'{prd:.2f}', f'{wwprd:.2f}', 
                          f'{snr:.2f}', prd_qual, wwprd_qual])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.15, 0.15, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color data rows alternating
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax4.set_title('Week 2 Results Summary', fontweight='bold', fontsize=12, pad=20)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved Week 2 summary figure to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing visualization functions...")
    
    # Dummy results for testing
    test_results = {
        4: {'PRD': 35.2, 'WWPRD': 30.1, 'SNR_improvement': 5.2, 'SNR_in': 6.0, 'SNR_out': 11.2},
        8: {'PRD': 28.5, 'WWPRD': 24.3, 'SNR_improvement': 6.1, 'SNR_in': 6.0, 'SNR_out': 12.1},
        16: {'PRD': 22.1, 'WWPRD': 18.7, 'SNR_improvement': 7.8, 'SNR_in': 6.0, 'SNR_out': 13.8},
        32: {'PRD': 18.3, 'WWPRD': 15.2, 'SNR_improvement': 9.5, 'SNR_in': 6.0, 'SNR_out': 15.5}
    }
    
    plot_rate_distortion_curves(test_results, 'test_rd_curves.png')
    plot_snr_bar_chart(test_results, 'test_snr_chart.png')
    create_week2_summary_figure(test_results, 'test_week2_summary.png')
    
    print("✓ All tests passed!")

