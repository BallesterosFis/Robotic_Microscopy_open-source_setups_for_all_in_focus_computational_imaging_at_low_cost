import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import os
import sys
import cv2
import pywt
from scipy.optimize import curve_fit
from datetime import datetime
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: 'reportlab' not found. PDF report generation will be skipped.")

# --- CONFIGURATION ---
WAVELET_TYPE = "haar"       
DOF_THRESHOLD = 0.80         
FWHM_THRESHOLD = 0.50       
ROI_FIXED_SIZE = 1024       
Z_STEP = 0.2              

# --- ROI SELECTION ---
def select_square_roi(image):
    """
    Opens a window to let the user click the center of the Region of Interest (ROI).
    """
    if image is None: raise ValueError("Error: Null image provided.")
    
    if len(image.shape) == 3: image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: image_rgb = image

    h, w = image.shape[:2]
    effective_side = min(ROI_FIXED_SIZE, h, w)
    
    display_scale = 1.0
    if h > 900 or w > 1600: 
        display_scale = min(900/h, 1600/w)
        image_display = cv2.resize(image_rgb, (0,0), fx=display_scale, fy=display_scale)
    else:
        image_display = image_rgb

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax.imshow(image_display)
    ax.set_title(f"CLICK CENTER of ROI ({effective_side}x{effective_side})")
    plt.axis('off') 
    
    pts = plt.ginput(1, timeout=-1)
    plt.close(fig)
    
    if not pts:
        cx, cy = w//2, h//2
    else:
        cx = int(pts[0][0] / display_scale)
        cy = int(pts[0][1] / display_scale)

    x_min = max(0, cx - effective_side//2)
    y_min = max(0, cy - effective_side//2)
    
    if x_min + effective_side > w: x_min = w - effective_side
    if y_min + effective_side > h: y_min = h - effective_side
    
    return int(x_min), int(y_min), effective_side

# --- METRIC FUNCTIONS ---
def laplacian_variance(img):
    """Calculates variance of Laplacian (Edges)."""
    return cv2.Laplacian(img, cv2.CV_64F).var()

def tenengrad(img):
    """Calculates Tenengrad (Gradient Magnitude Squared)."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)

def wavelet_energy(img, wavelet="haar", level=1):
    """Calculates Sum of Squared Detail Coefficients (High Frequencies)."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs[-1] 
    return np.mean(cH**2 + cV**2 + cD**2)

# CORE CALCULATION LOGIC
def calculate_precise_dof(z_values, metric_values, threshold_ratio):
    """
    Calculates the Depth of Field (DoF) width by finding where the curve
    crosses a threshold relative to its baseline (prominence).
    """
    y = np.array(metric_values)
    z = np.array(z_values)

    edge_count = max(1, len(y) // 10) 
    baseline = (np.mean(y[:edge_count]) + np.mean(y[-edge_count:])) / 2

    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    peak_z = z[peak_idx]

    prominence = peak_val - baseline

    if prominence < (baseline * 0.05) or prominence <= 0:
        return None, None, 0.0, peak_z, peak_val

    cutoff_val = baseline + (prominence * threshold_ratio)

    
    #Search Left (Start Z)
    start_z = z[0] 
    for i in range(peak_idx, 0, -1):
        if y[i-1] < cutoff_val:
            y1, y2 = y[i-1], y[i]
            z1, z2 = z[i-1], z[i]
            fraction = (cutoff_val - y1) / (y2 - y1)
            start_z = z1 + fraction * (z2 - z1)
            break

    end_z = z[-1] 
    for i in range(peak_idx, len(y) - 1):
        if y[i+1] < cutoff_val:
            y1, y2 = y[i], y[i+1]
            z1, z2 = z[i], z[i+1]
            fraction = (cutoff_val - y1) / (y2 - y1) 
            end_z = z1 + fraction * (z2 - z1)
            break
            
    width = end_z - start_z
    return start_z, end_z, width, peak_z, peak_val

# PLOTTING
def create_summary_plot(z_positions, lap_vals, ten_vals, wav_vals, dof_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Focus Analysis (Step: {Z_STEP} µm)', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    
    def visual_norm(arr):
        mn, mx = np.min(arr), np.max(arr)
        return (arr - mn) / (mx - mn) if mx > mn else arr

    ax1.plot(z_positions, visual_norm(lap_vals), 'b-o', label='Laplacian', markersize=3, alpha=0.8, linewidth=1.5)
    ax1.plot(z_positions, visual_norm(ten_vals), 'g-s', label='Tenengrad', markersize=3, alpha=0.8, linewidth=1.5)
    ax1.plot(z_positions, visual_norm(wav_vals), 'm-^', label='Wavelet', markersize=3, alpha=0.8, linewidth=1.5)

    ax1.set_title('Normalized Profiles & Calculated 80% Ranges')
    ax1.set_xlabel('Z-Position (µm)')
    ax1.set_ylabel('Normalization of Figure of Merit')
    ax1.legend(loc='lower center', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    labels = [k.split()[0] for k in dof_results.keys()]
    w80 = [d['width_80'] for d in dof_results.values()]
    w50 = [d['width_50'] for d in dof_results.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    r1 = ax2.bar(x - width/2, w80, width, label='Strict (80%)', color='#1f77b4')
    r2 = ax2.bar(x + width/2, w50, width, label='FWHM (50%)', color='#ff7f0e')
    
    ax2.set_title('DoF Width Comparison')
    ax2.set_ylabel('Width (µm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    for rects in [r1, r2]:
        for rect in rects:
            h = rect.get_height()
            label_text = f'{h:.1f}' if h > 0 else "Noise"
            color_text = 'black' if h > 0 else 'red'
            ax2.text(rect.get_x() + rect.get_width()/2., h, label_text, 
                     ha='center', va='bottom', fontsize=9, color=color_text)

    ax3 = axes[1, 0]
    ax3.plot(z_positions, lap_vals, 'b-o', markersize=3, alpha=0.6, label='Laplacian')
    ax3.set_ylabel('Laplacian Value', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)
      
    ax3b = ax3.twinx()
    ax3b.plot(z_positions, ten_vals, 'g-s', markersize=3, alpha=0.5, label='Tenengrad')
    ax3b.plot(z_positions, wav_vals, 'm-^', markersize=3, alpha=0.5, label='Wavelet')
    ax3b.set_ylabel('Tenengrad / Wavelet', color='k')
    ax3.set_title('Absolute Raw Metric Values')
    ax3.set_xlabel('Z-Position (µm)')


    ax4 = axes[1, 1]
    peaks = [d['peak_z'] for d in dof_results.values()]
    
    bars = ax4.bar(labels, peaks, color=['blue', 'green', 'purple'], alpha=0.7)
    
    valid_peaks = [p for p in peaks if p is not None]
    if valid_peaks:
        mean_p = np.mean(valid_peaks)
        span = max(0.5, (max(valid_peaks) - min(valid_peaks)) * 2) 
        ax4.set_ylim(max(0, mean_p - span), mean_p + span)

    ax4.set_title('Position of Maximum Focus (Peak Z)')
    ax4.set_ylabel('Z-Position (µm)')
    ax4.grid(axis='y', alpha=0.3)
    
    for rect in bars:
        h = rect.get_height()
        ax4.text(rect.get_x() + rect.get_width()/2., h, f'{h:.2f}', 
                ha='center', va='bottom', fontweight='bold', color='white',
                path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])

    plt.tight_layout()
    return fig

def generate_pdf(folder, metrics, dof_res, roi_s):
    if not REPORTLAB_AVAILABLE: return
    
    save_path = os.path.abspath(folder)
    pdf_file = os.path.join(save_path, f"Focus_Report_{datetime.now().strftime('%H%M%S')}.pdf")
    
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Focus Analysis Report: {folder}", styles['Title']), Spacer(1, 12)]
    
    story.append(Paragraph(f"ROI Size: {roi_s}x{roi_s} pixels", styles['Normal']))
    story.append(Paragraph(f"Z-Step: {Z_STEP} µm", styles['Normal']))
    story.append(Spacer(1, 12))

    headers = ["Metric", "Peak Z (µm)", "Range 80%", "Width 80%", "Range 50%", "Width 50%"]
    data = [headers]
    
    for k, v in dof_res.items():
        pk = f"{v['peak_z']:.2f}"
        
        r80 = f"{v['start_80']:.1f}-{v['end_80']:.1f}" if v['start_80'] else "Noise"
        w80 = f"{v['width_80']:.2f}"
        
        r50 = f"{v['start_50']:.1f}-{v['end_50']:.1f}" if v['start_50'] else "Noise"
        w50 = f"{v['width_50']:.2f}"
        
        row = [k.split()[0], pk, r80, w80, r50, w50]
        data.append(row)
        
    t = Table(data)
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 20))
    
    img_path = os.path.join(save_path, "temp_plot.png")
    if os.path.exists(img_path):
        story.append(ReportLabImage(img_path, width=7*inch, height=5*inch))
        
    try: 
        doc.build(story)
        print(f"PDF Report saved: {pdf_file}")
    except Exception as e: 
        print(f"Error building PDF: {e}")

def analyze_stack(folder_name):
    path = os.path.abspath(folder_name)
    
    if not os.path.exists(path):
        print(f"Error: Folder '{path}' not found.")
        return

    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.tif','.jpeg','.bmp'))])
    if not files: 
        print("No images found in folder.")
        return
    
    ref_path = os.path.join(path, files[len(files)//2])
    
    ref_stream = np.fromfile(ref_path, dtype=np.uint8)
    ref = cv2.imdecode(ref_stream, cv2.IMREAD_COLOR)
    
    if ref is None:
        print(f"Error reading image: {ref_path}")
        return


    x, y, s = select_square_roi(ref)
    print(f"Selected ROI: x={x}, y={y}, size={s}")
    
    print(f"Processing {len(files)} images...")
    L, T, W = [], [], []
    

    for f in files:
        img_path = os.path.join(path, f)
        
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {f}")
            L.append(0)
            T.append(0)
            W.append(0)
            continue

        roi = img[y:y+s, x:x+s]
        
        L.append(laplacian_variance(roi))
        T.append(tenengrad(roi))
        W.append(wavelet_energy(roi, WAVELET_TYPE))
        
    z_axis = np.arange(len(L)) * Z_STEP
    metrics = {
        'Laplacian': L, 
        'Tenengrad': T, 
        f'Wavelet ({WAVELET_TYPE})': W
    }
    
    results = {}
    
    print("\n--- Results ---")
    for name, data in metrics.items():
        s80, e80, w80, peak_z, peak_val = calculate_precise_dof(z_axis, data, DOF_THRESHOLD)
        
        s50, e50, w50, _, _ = calculate_precise_dof(z_axis, data, FWHM_THRESHOLD)
        
        results[name] = {
            'peak_z': peak_z,
            'start_80': s80, 'end_80': e80, 'width_80': w80,
            'start_50': s50, 'end_50': e50, 'width_50': w50
        }
        
        print(f"{name}: Peak Z={peak_z:.2f}µm | 80% Width={w80:.2f}µm | FWHM={w50:.2f}µm")
        
    fig = create_summary_plot(z_axis, metrics['Laplacian'], metrics['Tenengrad'], metrics[f'Wavelet ({WAVELET_TYPE})'], results)
    
    out_dir = os.path.join(path, "graphs")
    os.makedirs(out_dir, exist_ok=True)
    
    plot_path = os.path.join(out_dir, "focus_analysis_plot.png")
    fig.savefig(plot_path)
    fig.savefig(os.path.join(path, "temp_plot.png"))
    
    print(f"Plot saved to: {plot_path}")
    
    generate_pdf(folder_name, metrics, results, s)
    
    plt.show()

if __name__ == "__main__":
    folder = input("Enter Image Folder Path: ").strip()
    folder = folder.strip('"').strip("'")
    
    if folder: 
        analyze_stack(folder)
