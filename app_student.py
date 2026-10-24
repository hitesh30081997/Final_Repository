"""
app_student.py
Gradio demo for YOUR trained SLU model (from the Code folder), in the same style
as app3.py (NeMo Conformer SLURP demo): record/upload speech, optionally mix in
background noise at a chosen SNR, and see the extracted intent + slots.

It auto-detects which kind of checkpoint you give it:

  A) Sequence-KD student   (train_seq_kd.py  ->  ckpt_seq_kd/best_student.pt)
     HuBERT/WavLM encoder + the teacher's Transformer decoder. Output format is
     identical to the NeMo model:  {'scenario': .., 'action': .., 'entities': [..]}
     Needs the teacher .nemo file (for decoder architecture + tokenizer).

  B) CTC student           (train.py / train_kd.py  ->  ckpt*/best_model.pt)
     HuBERT intent classifier + slot-tagging CTC head. Needs intent_labels.json
     and ctc_vocab.json (or bpe_tokenizer.json) next to the checkpoint.

Optionally the NeMo teacher's prediction is shown next to yours on the same
(noisy) audio, so the audience can compare both models live.

Run (from any folder):
    python3 app_student.py
    python3 app_student.py --ckpt /path/to/best_student.pt --teacher_nemo /path/to/model.nemo
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import gradio as gr

# =========================================================================== #
# EDIT THESE PATHS (or pass them on the command line, see --help)
# =========================================================================== #
CODE_DIR = "/home/common/Code"                                    # your unzipped Code folder
STUDENT_CKPT = "/home/common/Code/ckpt_seq_kd/best_student.pt"    # your trained model
TEACHER_NEMO = ("/home/common/Downloads/nvidia-conformer-large-slurp-other-default-v1/"
                "slu_conformer_transformer_large_slurp_1.nemo")    # same file app3.py uses
NOISE_DIR = "/home/common/noises"                                 # noises/<class>/<file>.wav

# The checkpoint stores the encoder name used at training time. If that was a local
# folder on another machine (e.g. /home/hp2/models/Wavlm-base-plus), set this to a
# folder that exists here or to the Hugging Face name. Your checkpoint already holds
# all the fine-tuned encoder weights; this is only used to build the architecture.
# None = use the stored name, or pick the matching public model if that path is missing.
HUBERT_NAME_OVERRIDE = None        # e.g. "microsoft/wavlm-base-plus"

# Only used for type-B checkpoints (best_model.pt from train.py / train_kd.py).
# If kd_args.json is next to the checkpoint, these are read from it automatically.
CTC_HUBERT_NAME = "facebook/hubert-base-ls960"
CTC_SEMANTIC_LAYER = 8
CTC_LAYER_FUSION = "single"        # "single" | "weighted_sum" (must match training)
CTC_FUSION_LAYERS = None           # e.g. "1,2,3,4,5,6,7,8,9,10,11,12"
CTC_VOCAB_TYPE = "char"            # "char" | "bpe"
# =========================================================================== #

SNR_OPTIONS = [0, 5, 10]  # dB
BEAM_OPTIONS = [1, 4, 8, 32]
TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def parse_args():
    p = argparse.ArgumentParser(description="Demo for your trained SLU model")
    p.add_argument("--code_dir", default=CODE_DIR)
    p.add_argument("--ckpt", default=STUDENT_CKPT)
    p.add_argument("--teacher_nemo", default=TEACHER_NEMO)
    p.add_argument("--noise_dir", default=NOISE_DIR)
    p.add_argument("--hubert_name", default=HUBERT_NAME_OVERRIDE,
                   help="Override the encoder name/path stored in the checkpoint")
    p.add_argument("--no_teacher", action="store_true",
                   help="Don't load the NeMo teacher for comparison (type-B checkpoints only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


ARGS = parse_args()
DEVICE = ARGS.device
sys.path.insert(0, str(Path(ARGS.code_dir).resolve()))  # so we can import your modules

# Imported after sys.path is set (these come from your Code folder)
from semantics_utils import parse_semantics  # noqa: E402


PUBLIC_ENCODERS = {               # local folder name hint -> Hugging Face model
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-base": "microsoft/wavlm-base",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-base": "facebook/hubert-base-ls960",
    "hubert-large": "facebook/hubert-large-ll60k",
    "wav2vec2-base": "facebook/wav2vec2-base",
}


def resolve_encoder_name(stored):
    """Encoder name to build the model with (weights then come from your checkpoint)."""
    if ARGS.hubert_name:
        return ARGS.hubert_name
    looks_local = stored.startswith(("/", ".", "~")) or stored.count("/") > 1
    if not looks_local or os.path.isdir(os.path.expanduser(stored)):
        return stored
    base = os.path.basename(stored.rstrip("/")).lower()
    for hint, hub_name in PUBLIC_ENCODERS.items():   # longest/most specific hints listed first
        if base.startswith(hint):
            print(f"Encoder path {stored!r} not found here -> using {hub_name!r} for the architecture "
                  "(your fine-tuned weights are loaded from the checkpoint)")
            return hub_name
    raise SystemExit(f"Encoder path {stored!r} from the checkpoint does not exist on this machine. "
                     "Set HUBERT_NAME_OVERRIDE (or --hubert_name) to a local copy or a Hugging Face name.")


# =========================================================================== #
# Teacher (NeMo Conformer SLURP) - loaded with the same patches as training
# =========================================================================== #
def load_teacher():
    from teacher_nemo import load_teacher_model
    print(f"Loading NeMo teacher from {ARGS.teacher_nemo} ...")
    return load_teacher_model(ARGS.teacher_nemo, DEVICE)


class TeacherRunner:
    def __init__(self, teacher):
        self.m = teacher
        self.tok = teacher.tokenizer
        self._gens = {}

    def _gen(self, beam):
        from teacher_nemo import make_nemo_generator
        if beam not in self._gens:
            self._gens[beam] = make_nemo_generator(self.m.embedding, self.m.decoder,
                                                   self.m.classifier, self.tok, beam_size=beam)
        return self._gens[beam]

    @torch.no_grad()
    def predict(self, wav, beam):
        from nemo.collections.asr.parts.utils.slu_utils import get_seq_mask
        from teacher_nemo import run_generator, tokens_to_id_lists
        lens = torch.tensor([wav.shape[1]], device=DEVICE)
        f, fl = self.m.preprocessor(input_signal=wav, length=lens)
        enc, el = self.m.encoder(audio_signal=f, length=fl)
        enc = enc.transpose(1, 2)
        tokens = run_generator(self._gen(beam), enc, get_seq_mask(enc, el))
        ids = tokens_to_id_lists(tokens, self.tok.bos_id, self.tok.eos_id)[0]
        return "".join(self.tok.tokenizer.decode_ids(ids))


# =========================================================================== #
# Type A: sequence-KD student (best_student.pt)
# =========================================================================== #
class SeqStudentRunner:
    kind = "seq"

    def __init__(self, ckpt, teacher):
        from model_seq import HubertSemanticSLU
        from seq_dataset import CharVocab
        cfg = ckpt["config"]
        self.tok = teacher.tokenizer
        self.model = HubertSemanticSLU(
            teacher, hubert_name=resolve_encoder_name(cfg["hubert_name"]), layer_fusion=cfg["layer_fusion"],
            semantic_layer=cfg["semantic_layer"],
            fusion_layers=[int(x) for x in cfg["fusion_layers"].split(",")] if cfg["fusion_layers"] else None,
            adapter_layers=cfg["adapter_layers"], ctc_vocab_size=cfg.get("ctc_vocab_size", 0),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEVICE).eval()
        self.char_vocab = CharVocab(ckpt["ctc_chars"]) if ckpt.get("ctc_chars") else None
        self.description = f"Sequence-KD student ({cfg['hubert_name']} encoder + Transformer decoder)"
        self._gens = {}

    def _gen(self, beam):
        if beam not in self._gens:
            self._gens[beam] = self.model.build_generator(self.tok, beam_size=beam)
        return self._gens[beam]

    @torch.no_grad()
    def predict(self, wav, beam):
        att = torch.ones_like(wav, dtype=torch.long)
        bos, eos = self.tok.bos_id, self.tok.eos_id
        if beam <= 1:
            ids, ctc = self.model.greedy(wav, att, bos, eos, max_len=128, return_ctc=True)
        else:
            ids, ctc = self.model.beam_search(wav, att, self._gen(beam), bos, eos, return_ctc=True)
        raw = "".join(self.tok.tokenizer.decode_ids(ids[0]))
        transcript = ""
        if ctc is not None and self.char_vocab is not None:
            transcript = self.char_vocab.decode_ctc(ctc[0])
        return raw, transcript


# =========================================================================== #
# Type B: CTC student (best_model.pt from train.py / train_kd.py)
# =========================================================================== #
class CTCStudentRunner:
    kind = "ctc"

    def __init__(self, state_dict, ckpt_path):
        from model import HubertSLUModel
        from vocab import CTCVocab
        ckpt_dir = Path(ckpt_path).parent
        cfg = {"hubert_name": CTC_HUBERT_NAME, "semantic_layer": CTC_SEMANTIC_LAYER,
               "layer_fusion": CTC_LAYER_FUSION, "fusion_layers": CTC_FUSION_LAYERS,
               "vocab_type": CTC_VOCAB_TYPE}
        for name in ("kd_args.json", "args.json", "train_args.json"):
            if (ckpt_dir / name).exists():
                saved = json.loads((ckpt_dir / name).read_text())
                cfg.update({k: saved[k] for k in cfg if k in saved})
                print(f"Read model settings from {ckpt_dir / name}")
                break

        self.intent_list = json.loads((ckpt_dir / "intent_labels.json").read_text())
        if cfg["vocab_type"] == "bpe":
            from vocab_bpe import BPECTCVocab
            self.vocab = BPECTCVocab.load(ckpt_dir / "bpe_tokenizer.json")
        else:
            self.vocab = CTCVocab.load(ckpt_dir / "ctc_vocab.json")

        fl = cfg["fusion_layers"]
        self.model = HubertSLUModel(
            num_intents=len(self.intent_list), ctc_vocab_size=len(self.vocab),
            pad_id=self.vocab.pad_id, blank_id=self.vocab.blank_id,
            hubert_name=resolve_encoder_name(cfg["hubert_name"]), semantic_layer=cfg["semantic_layer"],
            layer_fusion=cfg["layer_fusion"],
            fusion_layers=[int(x) for x in fl.split(",")] if fl else None,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE).eval()
        self.description = (f"CTC student ({cfg['hubert_name']}, {cfg['layer_fusion']} fusion, "
                            f"{len(self.intent_list)} intents)")

    @torch.no_grad()
    def predict(self, wav, beam=None):
        from vocab import extract_slots_from_tagged_text, strip_tags
        att = torch.ones(wav.shape, dtype=torch.long, device=DEVICE)
        out = self.model(wav, att)
        probs = torch.softmax(out["intent_logits"][0], -1)
        conf, idx = probs.max(-1)
        intent = self.intent_list[idx.item()]
        greedy = out["ctc_log_probs"].argmax(-1).transpose(0, 1)[0]
        L = int(out["feat_lens"][0].item())
        tagged = self.vocab.decode(greedy[:L].tolist())
        scenario, _, action = intent.partition("_")
        result = {
            "scenario": scenario,
            "action": action,
            "entities": [{"type": t, "filler": f} for t, f in extract_slots_from_tagged_text(tagged)],
            "intent_confidence": round(conf.item(), 4),
        }
        return json.dumps(result), strip_tags(tagged)


# =========================================================================== #
# Load everything
# =========================================================================== #
print(f"Loading your model from {ARGS.ckpt} (device={DEVICE}) ...")
_raw = torch.load(ARGS.ckpt, map_location="cpu", weights_only=False)
IS_SEQ = isinstance(_raw, dict) and "state_dict" in _raw and "config" in _raw

TEACHER_MODEL = None
if IS_SEQ or (not ARGS.no_teacher and os.path.exists(ARGS.teacher_nemo)):
    TEACHER_MODEL = load_teacher()

STUDENT = SeqStudentRunner(_raw, TEACHER_MODEL) if IS_SEQ else CTCStudentRunner(_raw, ARGS.ckpt)
TEACHER = TeacherRunner(TEACHER_MODEL) if TEACHER_MODEL is not None else None
del _raw
print(f"Loaded: {STUDENT.description}")


# =========================================================================== #
# Audio + noise helpers (same behaviour as app3.py)
# =========================================================================== #
def scan_noise_dir(noise_dir):
    """Return {class_name: [(file_name, full_path), ...]} for every sub-folder."""
    catalog = {}
    if not os.path.isdir(noise_dir):
        print(f"WARNING: noise folder not found: {noise_dir}")
        return catalog
    for cls_name in sorted(os.listdir(noise_dir)):
        cls_path = os.path.join(noise_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        files = [(f, os.path.join(cls_path, f)) for f in sorted(os.listdir(cls_path))
                 if f.lower().endswith(AUDIO_EXTS)]
        if files:
            catalog[cls_name] = files
    return catalog


NOISE_CATALOG = scan_noise_dir(ARGS.noise_dir)
NOISE_CLASSES = list(NOISE_CATALOG.keys())
print(f"Found noise classes: {NOISE_CLASSES}")


def load_audio(path):
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def add_noise_at_snr(speech, noise, snr_db, rng=None):
    rng = rng or np.random.default_rng()
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = rng.integers(0, len(noise) - len(speech) + 1)
    noise = noise[start:start + len(speech)]
    speech_power, noise_power = np.mean(speech ** 2), np.mean(noise ** 2)
    if speech_power == 0 or noise_power == 0:
        return speech
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def save_temp_wav(audio):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, TARGET_SR)
    return path


def pretty(raw):
    """Model string -> pretty JSON (falls back to the raw string)."""
    try:
        d = json.loads(raw)
    except Exception:
        d = parse_semantics(raw)
    return json.dumps(d, indent=4) if d else raw


def sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


# =========================================================================== #
# Gradio callbacks
# =========================================================================== #
def analyse(audio_path, use_noise, noise_path, snr_db, beam, compare):
    if audio_path is None:
        return None, "Please upload or record an audio file", "", "", ""
    try:
        speech = load_audio(audio_path)
        if use_noise:
            if not noise_path:
                return (None, "Please select a noise sample, or untick 'Add background noise'",
                        "", "", "")
            speech = add_noise_at_snr(speech, load_audio(noise_path), float(snr_db))
        model_input = save_temp_wav(speech)
        wav = torch.from_numpy(speech).unsqueeze(0).to(DEVICE)
        beam = int(beam)

        t0 = time.perf_counter()
        raw, transcript = STUDENT.predict(wav, beam)
        sync()
        t_student = time.perf_counter() - t0
        info = f"**Your model:** {t_student * 1000:.0f} ms"

        teacher_out = ""
        if compare and TEACHER is not None:
            t0 = time.perf_counter()
            teacher_out = pretty(TEACHER.predict(wav, beam))
            sync()
            info += f"  |  **NeMo teacher:** {(time.perf_counter() - t0) * 1000:.0f} ms"
        info += f"  |  audio {len(speech) / TARGET_SR:.1f} s on {DEVICE}"
        if use_noise:
            info += f", noise at {snr_db} dB SNR"

        return model_input, pretty(raw), transcript, teacher_out, info
    except Exception as e:
        return None, f"An error occurred during inference: {e}", "", "", ""


def noise_choices(cls_name):
    return [(name, path) for name, path in NOISE_CATALOG.get(cls_name, [])]


def on_class_change(cls_name):
    choices = noise_choices(cls_name)
    first = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first), first


# =========================================================================== #
# UI
# =========================================================================== #
TITLE = "Noise-Robust SLU Demo (My Trained Model)"

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        f"This web interface runs **my trained model** — {STUDENT.description} — "
        "trained on SLURP for joint intent classification and slot filling. "
        "You can mix background noise into your speech at a chosen SNR to test how robust "
        "it is, and compare its prediction with the original NVIDIA NeMo Conformer model."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                   label="Input Speech")

            use_noise = gr.Checkbox(label="Add background noise", value=False)
            with gr.Group(visible=False) as noise_box:
                default_class = NOISE_CLASSES[0] if NOISE_CLASSES else None
                default_choices = noise_choices(default_class)
                default_noise = default_choices[0][1] if default_choices else None
                noise_class = gr.Dropdown(choices=NOISE_CLASSES, value=default_class,
                                          label="Noise class")
                noise_sample = gr.Dropdown(choices=default_choices, value=default_noise,
                                           label="Noise sample")
                noise_preview = gr.Audio(value=default_noise, type="filepath",
                                         label="Noise preview", interactive=False)
                snr = gr.Radio(choices=SNR_OPTIONS, value=SNR_OPTIONS[-1], label="SNR (dB)")

            beam = gr.Radio(choices=BEAM_OPTIONS,
                            value=32 if DEVICE.startswith("cuda") else 4,
                            label="Beam size (1 = greedy, fastest; 32 = published SLURP setting)",
                            visible=STUDENT.kind == "seq" or TEACHER is not None)
            compare = gr.Checkbox(label="Compare with NeMo Conformer (teacher)",
                                  value=TEACHER is not None, visible=TEACHER is not None)

            submit_btn = gr.Button("Analyse speech", variant="primary")

        with gr.Column():
            model_audio = gr.Audio(type="filepath", interactive=False,
                                   label="Audio sent to the model (16 kHz mono, with noise if selected)")
            output_text = gr.Code(label="My model — Extracted Semantics (Intent and Slots)",
                                  language="json")
            transcript = gr.Textbox(label="My model — Transcript (CTC head)", interactive=False,
                                    visible=STUDENT.kind == "ctc" or getattr(STUDENT, "char_vocab", None) is not None)
            teacher_text = gr.Code(label="NeMo Conformer (teacher) — Extracted Semantics",
                                   language="json", visible=TEACHER is not None)
            info = gr.Markdown()

    use_noise.change(lambda v: gr.update(visible=v), inputs=use_noise, outputs=noise_box)
    noise_class.change(on_class_change, inputs=noise_class, outputs=[noise_sample, noise_preview])
    noise_sample.change(lambda p: p, inputs=noise_sample, outputs=noise_preview)
    compare.change(lambda v: gr.update(visible=v), inputs=compare, outputs=teacher_text)

    submit_btn.click(
        fn=analyse,
        inputs=[audio_input, use_noise, noise_sample, snr, beam, compare],
        outputs=[model_audio, output_text, transcript, teacher_text, info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=ARGS.port, allowed_paths=[ARGS.noise_dir])
