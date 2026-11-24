"""AudioScope demo: waveform, log-mel spectrogram, top-k classes, and Grad-CAM.

    streamlit run app/streamlit_app.py

Checkpoints are discovered under ``eval/runs/*/fold*/model.pt`` and
``checkpoints/*.pt``. The front end is rebuilt from the config saved inside
the selected checkpoint, so the spectrogram on screen is the one the model
scores. Without a checkpoint the page still draws the waveform and the
spectrogram (using configs/baseline.yaml) and says why there is no prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from audioscope.config import load_config
from audioscope.features import FeatureConfig, LogMel, fix_length, load_audio
from audioscope.viz import plot_logmel, plot_waveform

REPO = Path(__file__).resolve().parents[1]
torch.set_grad_enabled(False)


def find_checkpoints() -> list[Path]:
    found = sorted((REPO / "eval" / "runs").glob("*/fold*/model.pt")) + sorted((REPO / "checkpoints").glob("*.pt"))
    return found


@st.cache_resource(show_spinner="Loading checkpoint…")
def get_predictor(path: str):
    from audioscope.inference import Predictor

    return Predictor.from_path(path)


def main() -> None:
    st.set_page_config(page_title="AudioScope", layout="wide")
    st.title("AudioScope")
    st.caption("Environmental sound classification on log-mel spectrograms (ESC-50).")

    ckpts = find_checkpoints()
    with st.sidebar:
        st.header("Model")
        if ckpts:
            labels = [str(p.relative_to(REPO)) for p in ckpts]
            choice = st.selectbox("Checkpoint", labels)
            predictor = get_predictor(str(REPO / choice))
            fcfg = predictor.fcfg
            info = predictor.ckpt
            st.markdown(
                f"**Checkpoint:** `{choice}`  \n"
                f"**Config:** `{info['config'].get('config_path', info['config'].get('name'))}`  \n"
                f"**Model:** `{info['model_name']}`  \n"
                f"**Trained on:** {info['dataset']} (fold {info['held_out_fold']} held out)  \n"
                f"**Run:** `{info.get('run_id')}` · git `{str(info.get('git_revision'))[:7]}`  \n"
                f"**Grad-CAM layer:** `{info['gradcam_layer']}`"
            )
        else:
            predictor = None
            fcfg = FeatureConfig.from_dict(load_config(REPO / "configs" / "baseline.yaml")["features"])
            st.info(
                "No trained checkpoint found. Run `python eval/crossval.py --config configs/baseline.yaml "
                "--out eval/runs/<run-id>` to create one. Figures below use configs/baseline.yaml."
            )
        k = st.slider("Top-k", 1, 10, 5)
        show_cam = st.checkbox("Grad-CAM overlay", value=True, disabled=predictor is None)

    col_up, col_rec = st.columns(2)
    with col_up:
        upload = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "mp3"])
    with col_rec:
        recording = st.audio_input("…or record from the microphone") if hasattr(st, "audio_input") else None

    source = upload or recording
    if source is None:
        st.write("Upload a clip or record one to see its waveform, spectrogram, and prediction.")
        return

    data = source.getvalue()
    st.audio(data)
    try:
        wave, sr = load_audio(data, fcfg.sample_rate)
    except Exception as exc:  # noqa: BLE001 - show any decode error to the user
        st.error(f"Could not decode audio: {exc}")
        return
    full_seconds = len(wave) / sr
    clip = fix_length(wave, fcfg.num_samples)
    if full_seconds > fcfg.clip_seconds + 1e-3:
        st.warning(f"The clip is {full_seconds:.1f} s; the model scores the first {fcfg.clip_seconds:g} s.")
    elif full_seconds < fcfg.clip_seconds - 1e-3:
        st.info(f"The clip is {full_seconds:.1f} s; it is zero-padded to {fcfg.clip_seconds:g} s, as in training.")

    st.subheader("Waveform")
    st.pyplot(plot_waveform(clip.numpy(), sr), clear_figure=True)

    spec = LogMel(fcfg)(clip).numpy()
    cam = None
    if predictor is not None:
        pred = predictor.predict(wave, k=k)
        if show_cam:
            cam = predictor.gradcam(wave, class_idx=pred.indices[0])

    st.subheader("Log-mel spectrogram" + (" with Grad-CAM" if cam is not None else ""))
    title = f"Grad-CAM for '{pred.labels[0]}'" if cam is not None else None
    st.pyplot(plot_logmel(spec, fcfg, title=title, cam=cam), clear_figure=True)
    st.caption(
        f"{fcfg.n_mels} mel bands, {fcfg.sample_rate} Hz, n_fft {fcfg.n_fft}, hop {fcfg.hop_length}. "
        "The y axis shows mel-band center frequencies. Bright overlay = time–frequency regions that raised "
        "the score of the predicted class."
    )

    st.subheader("Prediction")
    if predictor is None:
        st.warning("Predictions and Grad-CAM need a trained checkpoint. None was found.")
        return
    st.table(
        {
            "class": pred.labels,
            "confidence": [f"{p:.1%}" for p in pred.probs],
        }
    )
    st.caption(
        f"Softmax over the {len(predictor.class_names)} {predictor.ckpt['dataset']} classes the model was trained on. "
        "A confidence is relative to that closed label set; it does not say the sound is one of them."
    )


if __name__ == "__main__":
    main()
