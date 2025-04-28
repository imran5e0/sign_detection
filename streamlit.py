#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.set_page_config(page_title="Signature Deskew & Validation", layout="centered")
st.title("Signature Deskew & Validation")

# ────────────────────────────────
def load_and_pad(path_or_img, size=800):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)
    else:
        img = path_or_img
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
    m = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )
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
    cen = pts - mean
    cov = np.cov(cen, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc = eigvecs[:, np.argmax(eigvals)]
    projections = cen @ pc
    if projections.sum() < 0:
        pc = -pc
    return mean, pc
# ────────────────────────────────

uploaded = st.file_uploader("Upload a signature image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if not uploaded:
    st.info("Please upload an image.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ── processing
img = load_and_pad(orig)
mask = binary_mask(img)
pts  = cluster_signature(mask)
pts  = pca_band_filter(pts, width=15)

if pts.size == 0:
    st.error("Could not find signature points.")
    st.stop()

mean, pc = principal_axis(pts)
raw = np.degrees(np.arctan2(pc[1], pc[0]))
if raw > 90: raw -= 180
if raw <= -90: raw += 180
angle = raw

# rotate on big canvas
H, W = img.shape[:2]
D = int(np.ceil(np.hypot(W, H)))
big = 255 * np.ones((D, D, 3), np.uint8)
y_off, x_off = (D-H)//2, (D-W)//2
big[y_off:y_off+H, x_off:x_off+W] = img

M = cv2.getRotationMatrix2D((D/2, D/2), angle, 1.0)
rotated = cv2.warpAffine(big, M, (D, D),
                         flags=cv2.INTER_CUBIC,
                         borderValue=(255,255,255))

# validation step
mask2 = binary_mask(rotated)
pts2  = cluster_signature(mask2)
pts2  = pca_band_filter(pts2, width=15)

if pts2.size == 0:
    validated = rotated
else:
    x0r, y0r = pts2[:,0].min(), pts2[:,1].min()
    x1r, y1r = pts2[:,0].max(), pts2[:,1].max()
    box = mask2[y0r:y1r, x0r:x1r]
    half = box.shape[1]//2
    left, right = box[:,:half].sum(), box[:,half:].sum()
    if not np.any(box.sum(axis=0)==0) and right > left:
        validated = cv2.rotate(rotated, cv2.ROTATE_180)
    else:
        validated = rotated

# ── Display
st.image(cv2.cvtColor(validated, cv2.COLOR_BGR2RGB), caption=f"Final validated (θ={angle:.1f}°)", use_column_width=True)
