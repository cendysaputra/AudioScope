#!/usr/bin/env python
"""Animate the AudioScope pipeline on one real clip (README figure).

    python eval/make_process_gif.py --checkpoint eval/runs/transfer-panns-cnn14-fc1-v1/fold1/model.pt \
        --audio data/esc50/audio/1-54084-A-42.wav --out docs/process.gif

Everything drawn comes from the checkpoint: its front end makes the log-mel
spectrogram, its Grad-CAM layer makes the overlay, and its softmax gives the
top-5 bars. Use a clip from the checkpoint's held-out fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from audioscope.features.audio import fix_length, load_audio  # noqa: E402
from audioscope.inference import Predictor  # noqa: E402

INK, INK2, MUTED, GRID = "#191919", "#666666", "#bdbdbd", "#e6e6e6"
ACCENT = "#257de0"
STEPS = ["1  Audio", "2  Log-mel spectrogram", "3  Cnn14 + Grad-CAM", "4  Prediction"]
# frames per phase: waveform sweep, spectrogram sweep, CAM fade-in, bars grow, hold
PHASES = [24, 24, 14, 14, 26]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="docs/process.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    pred = Predictor.from_path(args.checkpoint)
    fcfg = pred.fcfg
    wave = fix_length(load_audio(args.audio, fcfg.sample_rate)[0], fcfg.num_samples)
    spec = pred.features(wave)[0, 0].numpy()  # (mels, frames)
    with torch.no_grad():
        probs = torch.softmax(pred.model(pred.features(wave))[0], 0).numpy()
    cam = pred.gradcam(wave)
    top = np.argsort(probs)[::-1][:5]
    names = [pred.class_names[i].replace("_", " ") for i in top]
    seconds = fcfg.num_samples / fcfg.sample_rate
    t_wave = np.arange(len(wave)) / fcfg.sample_rate

    fig = plt.figure(figsize=(10, 5.4), dpi=100)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 1.6], hspace=0.45, wspace=0.08,
                          left=0.07, right=0.97, top=0.84, bottom=0.1)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_w)
    ax_p = fig.add_subplot(gs[:, 1])

    step_txt = []
    for i, s in enumerate(STEPS):
        step_txt.append(fig.text(0.07 + i * 0.235, 0.93, s, fontsize=11, color=MUTED, weight="bold"))
    fig.text(0.07, 0.885, f"{Path(args.audio).name}  ·  ESC-50 test fold {pred.ckpt['held_out_fold']}  ·  "
             f"{pred.ckpt.get('run_id')}", fontsize=8.5, color=INK2)

    # waveform
    ax_w.set_xlim(0, seconds)
    lim = float(np.abs(wave).max()) * 1.1
    ax_w.set_ylim(-lim, lim)
    (wline,) = ax_w.plot([], [], color=ACCENT, lw=0.6)
    head_w = ax_w.axvline(0, color=INK, lw=1)
    ax_w.set_ylabel("amplitude", fontsize=8, color=INK2)
    # spectrogram (+ CAM)
    extent = (0, seconds, 0, spec.shape[0])
    blank = np.full_like(spec, np.nan)
    im = ax_s.imshow(blank, origin="lower", aspect="auto", cmap="magma", extent=extent,
                     vmin=np.percentile(spec, 2), vmax=spec.max())
    cam_rgba = plt.get_cmap("viridis")(cam)
    cam_im = ax_s.imshow(np.zeros_like(cam_rgba), origin="lower", aspect="auto", extent=extent)
    head_s = ax_s.axvline(0, color="white", lw=1)
    centers = pred.band_centers_hz()
    ticks = [0, 16, 32, 48, 63]
    ax_s.set_yticks(ticks)
    ax_s.set_yticklabels([f"{centers[t] / 1000:.1f}k" if centers[t] >= 1000 else f"{centers[t]:.0f}" for t in ticks])
    ax_s.set_ylabel("Hz (mel bands)", fontsize=8, color=INK2)
    ax_s.set_xlabel("time (s)", fontsize=8, color=INK2)
    cam_label = ax_s.text(0.01, 0.95, "", transform=ax_s.transAxes, fontsize=8.5, color="white", va="top")
    # prediction bars
    bars = ax_p.barh(range(5), [0] * 5, color=[ACCENT] + [MUTED] * 4, height=0.6)
    ax_p.set_xlim(0, 1.25)
    ax_p.set_ylim(4.6, -0.6)
    ax_p.set_yticks([])
    ax_p.set_xticks([])
    ax_p.set_title("top-5 classes", fontsize=9, color=INK2, loc="left")
    labels = [ax_p.text(0.02, i - 0.42, n, fontsize=9, color=INK, va="bottom") for i, n in enumerate(names)]
    values = [ax_p.text(0, i, "", fontsize=9, color=INK, va="center") for i in range(5)]
    for t in labels:
        t.set_alpha(0)
    for a in (ax_w, ax_s, ax_p):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_color(GRID)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=7.5)
    ax_p.spines["left"].set_visible(False)
    ax_p.spines["bottom"].set_visible(False)
    plt.setp(ax_w.get_xticklabels(), visible=False)

    bounds = np.cumsum([0] + PHASES)

    def set_step(k: int) -> None:
        for i, t in enumerate(step_txt):
            t.set_color(ACCENT if i == k else (INK if i < k else MUTED))

    def update(f: int):
        if f < bounds[1]:  # waveform sweep
            set_step(0)
            x = seconds * (f + 1) / PHASES[0]
            n = int(len(wave) * (f + 1) / PHASES[0])
            wline.set_data(t_wave[:n], wave[:n].numpy())
            head_w.set_xdata([x, x])
        elif f < bounds[2]:  # spectrogram sweep
            set_step(1)
            wline.set_data(t_wave, wave.numpy())
            g = (f - bounds[1] + 1) / PHASES[1]
            cols = int(spec.shape[1] * g)
            shown = blank.copy()
            shown[:, :cols] = spec[:, :cols]
            im.set_data(shown)
            head_w.set_xdata([seconds * g] * 2)
            head_s.set_xdata([seconds * g] * 2)
        elif f < bounds[3]:  # CAM fade-in
            set_step(2)
            im.set_data(spec)
            head_w.set_visible(False)
            head_s.set_visible(False)
            g = (f - bounds[2] + 1) / PHASES[2]
            rgba = cam_rgba.copy()
            rgba[..., 3] = 0.75 * cam * g
            cam_im.set_data(rgba)
            cam_label.set_text(f"Grad-CAM for '{names[0]}' ({pred.gradcam_layer})")
        else:  # bars grow, then hold
            set_step(3)
            g = min(1.0, (f - bounds[3] + 1) / PHASES[3])
            for i, (b, idx) in enumerate(zip(bars, top)):
                w = float(probs[idx]) * g
                b.set_width(w)
                values[i].set_position((w + 0.02, i))
                values[i].set_text(f"{100 * probs[idx] * g:.1f}%")
                labels[i].set_alpha(1)
        return []

    anim = FuncAnimation(fig, update, frames=int(bounds[-1]), blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}: {int(bounds[-1])} frames, top-1 {names[0]} {100 * probs[top[0]]:.1f}%")


if __name__ == "__main__":
    main()
