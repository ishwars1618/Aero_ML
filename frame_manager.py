import cv2, time, threading, numpy as np, torch

# written by chatgpt

# pip install opencv-python
import cv2, time, numpy as np, torch
import torch.nn.functional as F

# ---------- Config ----------
INPUT_SIZE = (128, 128)     # (H, W) model input resolution
##INPUT_SIZE = (128, 128)     # (H, W) model input resolution
LABELS = ["Nose", "L", "R", "T", "V"]  # optional
##LABELS = ["L", "R", "T", "O"]  # optional
##COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]  # BGR
COLORS = [(255,0,0),(0,255,0),(0,0,255), (255,255,0)]  # BGR
DRAW_RADIUS = 3
DRAW_THICKNESS = 2
USE_SOFT_ARGMAX = True     # True for differentiable-ish expected-value coords

# ---------- Utils ----------
def preprocess_frame(frame_bgr, device):
    """Resize, BGR->RGB, [0,1], normalize, to tensor (1,C,H,W)."""
    h0, w0 = frame_bgr.shape[:2]
    fr = cv2.resize(frame_bgr, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)  # (1,3,H,W)
    return t, (w0, h0)

def _argmax_coords(hm):  # hm: (B,C,Hh,Wh)
    B, C, Hh, Wh = hm.shape
    flat = hm.view(B, C, -1)
    idx = flat.argmax(dim=2)                                 # (B,C)
    y = (idx // Wh).float()
    x = (idx %  Wh).float()
    return x, y                                              # each (B,C)

def _softargmax_coords(hm, tau=-3.0):  # hm: logits or probs, (B,C,Hh,Wh)
    # improve stability by subtracting max per map
    B, C, Hh, Wh = hm.shape
    xv = torch.linspace(0, Wh-1, Wh, device=hm.device).view(1,1,1,Wh)
    yv = torch.linspace(0, Hh-1, Hh, device=hm.device).view(1,1,Hh,1)
    logits = hm / max(tau, 1e-6)
    logits = logits - logits.amax(dim=(2,3), keepdim=True)
    p = torch.softmax(logits.view(B, C, -1), dim=-1).view(B, C, Hh, Wh)
    x = (p * xv).sum(dim=(2,3))
    y = (p * yv).sum(dim=(2,3))
    return x, y

def _expected_coords(hm):  # hm: logits or probs, (B,C,Hh,Wh)
    # improve stability by subtracting max per map
    B, C, Hh, Wh = hm.shape
    xs = torch.linspace(0.5, Wh-0.5, Wh)
    ys = torch.linspace(0.5, Wh-0.5, Wh)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # (N, 2), N = grid_size^2
    pts = pts.view((1, 1,)+pts.shape).repeat((B, C, 1, 1))
    pts[0][0]
    rearranged_probabilities = hm.view((B, C, Wh*Wh, 1)).repeat((1, 1, 1, 2))
    #print(rearranged_probabilities.shape)
    weighted_coords = rearranged_probabilities * pts
    ev_coords = torch.sum(weighted_coords, dim=2)
    #print(ev_coords)
    return ev_coords[0, :, 0], ev_coords[0, :, 1]

def heatmaps_to_coords(heatmaps):
    """
    heatmaps: Tensor (B, C, Hh, Wh) — can be raw logits.
    Returns pixel coords in the ORIGINAL frame space: (B, C, 2) as (x, y).
    Assumes model input is resized from original to INPUT_SIZE, then we scale back.
    """
    B, C, Hh, Wh = heatmaps.shape

    """
    # choose coord extractor
    if USE_SOFT_ARGMAX:
        xh, yh = _softargmax_coords(heatmaps)
    else:
        # If heatmaps are logits, you may want sigmoid first (optional):
        # heatmaps = torch.sigmoid(heatmaps)
        xh, yh = _argmax_coords(heatmaps)
    """
    xh, yh = _expected_coords(heatmaps)
    # Map heatmap coords -> model input coords -> back to original frame coords
    sx = INPUT_SIZE[1] / float(Wh)
    sy = INPUT_SIZE[0] / float(Hh)
    xin = xh * sx
    yin = yh * sy
    return xin, yin  # in model-input coordinate space (we rescale to original later)

def put_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

def draw_points(frame_bgr, xs_in, ys_in, orig_wh):
    """
    xs_in, ys_in: (C,) coords in model-input space
    orig_wh: (w0, h0) original frame size
    """
    h_in, w_in = INPUT_SIZE
    w0, h0 = orig_wh
    scale_x = w0 / float(w_in)
    scale_y = h0 / float(h_in)

    for i in range(xs_in.shape[0]):
    ##for i in range(xs_in.shape[0]-1):
        x0 = int(xs_in[i].item() * scale_x)
        y0 = int(ys_in[i].item() * scale_y)
        color = COLORS[i % len(COLORS)]
        cv2.circle(frame_bgr, (x0, y0), DRAW_RADIUS, color, DRAW_THICKNESS, lineType=cv2.LINE_AA)
        if i < len(LABELS):
            cv2.putText(frame_bgr, LABELS[i], (x0+5, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)