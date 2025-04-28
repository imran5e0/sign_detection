#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def load_and_pad(path, size=800):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = 255 * np.ones((size, size, 3), np.uint8)
    y0, x0 = (size - nh)//2, (size - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas

def binary_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 15, 10)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kern, iterations=4)
    return opened

def cluster_signature(mask, eps=2, min_samples=20):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack((xs, ys))
    if len(pts) == 0: return pts
    lbls = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    counts = np.bincount(lbls[lbls>=0])
    if len(counts) == 0: return pts
    return pts[lbls == np.argmax(counts)]

def pca_band_filter(pts, width=15):
    mean = pts.mean(axis=0)
    cen = pts - mean
    cov = np.cov(cen, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc = eigvecs[:, np.argmax(eigvals)]
    perp = np.array([-pc[1], pc[0]])
    d = np.abs(cen @ perp)
    return pts[d < width]

def principal_axis(pts):
    mean = pts.mean(axis=0)
    cen  = pts - mean
    cov  = np.cov(cen, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc = eigvecs[:, np.argmax(eigvals)]
    projections = cen @ pc
    if projections.sum() < 0:
        pc = -pc
    return mean, pc

# collect test images
paths = [f for f in sorted(os.listdir('.'))
         if f.startswith('test') and f.lower().endswith(
             ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]

n = len(paths)
fig, axes = plt.subplots(4, n, figsize=(4*n, 16), squeeze=False)

for col, path in enumerate(paths):
    img = load_and_pad(path)

    # ── row 0: scatter + PCA
    mask = binary_mask(img)
    pts = cluster_signature(mask)
    pts = pca_band_filter(pts, width=15)

    ax0 = axes[0][col]
    if pts.size == 0:
        ax0.set_title(path + "\n(no data)")
        ax0.axis('off')
        angle = 0
    else:
        mean, pc = principal_axis(pts)
        raw = np.degrees(np.arctan2(pc[1], pc[0]))
        if raw > 90: raw -= 180
        if raw <= -90: raw += 180
        angle = raw

        x0, y0 = pts[:,0].min(), pts[:,1].min()
        x1, y1 = pts[:,0].max(), pts[:,1].max()
        pad = 20
        x0b, y0b = max(0, x0-pad), max(0, y0-pad)
        x1b, y1b = min(img.shape[1], x1+pad), min(img.shape[0], y1+pad)
        xs = pts[:,0] - x0b
        ys = (y1b - pts[:,1])
        ax0.axhline(0, color='gray', linestyle='--')
        ax0.axvline(0, color='gray', linestyle='--')
        ax0.scatter(xs, ys, s=2, alpha=0.6)

        mean_s = np.array([mean[0]-x0b, y1b-mean[1]])
        pc_s = np.array([pc[0], -pc[1]])
        if pc_s[0] < 0: pc_s = -pc_s
        L = max(xs.max(), ys.max())
        line = np.vstack([mean_s - pc_s*L, mean_s + pc_s*L])
        ax0.plot(line[:,0], line[:,1], 'r-', linewidth=2,
                 label=f"PCA θ={angle:.1f}°")
        ax0.set_title(path)
        ax0.set_xlabel('x (px)'); ax0.set_ylabel('y (px)')
        ax0.legend(); ax0.set_aspect('equal')

    # ── row 1: original
    ax1 = axes[1][col]
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("original")
    ax1.axis('off')

    # ── row 2: rotate on large square canvas
    ax2 = axes[2][col]
    H, W = img.shape[:2]
    D = int(np.ceil(np.hypot(W, H)))
    big = 255 * np.ones((D, D, 3), np.uint8)
    y_off, x_off = (D-H)//2, (D-W)//2
    big[y_off:y_off+H, x_off:x_off+W] = img
    # ─── here we flip the sign so that negative angles rotate CCW ───
    M = cv2.getRotationMatrix2D((D/2, D/2), angle, 1.0)
    rotated = cv2.warpAffine(big, M, (D, D),
                             flags=cv2.INTER_CUBIC,
                             borderValue=(255,255,255))
    ax2.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"rotated {angle:.1f}°")
    ax2.axis('off')

    # ── row 3: validated on rotated
    ax3 = axes[3][col]
    mask2 = binary_mask(rotated)
    pts2 = cluster_signature(mask2)
    pts2 = pca_band_filter(pts2, width=15)
    if pts2.size == 0:
        validated = rotated
    else:
        x0r, y0r = pts2[:,0].min(), pts2[:,1].min()
        x1r, y1r = pts2[:,0].max(), pts2[:,1].max()
        box = mask2[y0r:y1r, x0r:x1r]
        half = box.shape[1]//2
        left = box[:, :half].sum()
        right = box[:, half:].sum()
        if not np.any(box.sum(axis=0)==0) and right>left:
            validated = cv2.rotate(rotated, cv2.ROTATE_180)
        else:
            validated = rotated

    ax3.imshow(cv2.cvtColor(validated, cv2.COLOR_BGR2RGB))
    ax3.set_title("validated")
    ax3.axis('off')

plt.tight_layout()
plt.show()
