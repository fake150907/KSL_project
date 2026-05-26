from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from inference import model_state as state  # noqa: E402
from inference.model_loader import ensure_models_loaded  # noqa: E402
from inference.predictor import predict_dual_scenario, predict_sequence_frames, softmax  # noqa: E402


SAMPLE_FPS = 12.0
SEQ_LEN = 32
TARGET = "SEN0354"
VIDEOS = [
    ROOT / "web" / "public" / "demo-videos" / "NIA_SL_SEN0354_REAL07_F.mp4",
    ROOT / "web" / "public" / "demo-videos" / "resident_realz03_01_hello.mp4",
]


def landmarks_to_xyzc_frame(results: object):
    points: list[list[float]] = []
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        points.extend([[0.0, 0.0, 0.0, 0.0] for _ in range(33)])
    else:
        pose_points = [
            [float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, "visibility", 1.0) or 0.0)]
            for lm in pose_landmarks.landmark[:33]
        ]
        while len(pose_points) < 33:
            pose_points.append([0.0, 0.0, 0.0, 0.0])
        points.extend(pose_points)

    for attr in ("left_hand_landmarks", "right_hand_landmarks"):
        landmarks = getattr(results, attr, None)
        if landmarks is None:
            points.extend([[0.0, 0.0, 0.0, 0.0] for _ in range(21)])
        else:
            hand_points = [
                [float(lm.x), float(lm.y), float(lm.z), 1.0]
                for lm in landmarks.landmark[:21]
            ]
            while len(hand_points) < 21:
                hand_points.append([0.0, 0.0, 0.0, 0.0])
            points.extend(hand_points)
    return state.np.asarray(points, dtype=state.np.float32)


def extract_frames(video: Path) -> tuple[list, list]:
    if state.cv2 is None or state.np is None or state.mp_holistic_instance is None:
        raise RuntimeError("Vision dependencies or MediaPipe are not available")

    cap = state.cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")

    fps = cap.get(state.cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(state.cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = float(count / fps) if fps else 0.0
    times = state.np.arange(0.0, duration + 1e-6, 1.0 / SAMPLE_FPS)

    frames = []
    frames_xyzc = []
    sampled = 0
    hand_frames = 0
    for current_time in times:
        cap.set(state.cv2.CAP_PROP_POS_MSEC, float(current_time * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        sampled += 1
        image_rgb = state.cv2.cvtColor(frame, state.cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        scale = min(640 / width, 360 / height, 1.0)
        if scale < 1.0:
            image_rgb = state.cv2.resize(
                image_rgb,
                (int(width * scale), int(height * scale)),
                interpolation=state.cv2.INTER_AREA,
            )
        results = state.mp_holistic_instance.process(image_rgb)
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hand_frames += 1
            frames.append(state.mediapipe_landmarks_to_frame(results, layout="mediapipe_xyzc"))
            frames_xyzc.append(landmarks_to_xyzc_frame(results))

    cap.release()
    print(f"\nVIDEO {video.name}")
    print(f"duration={duration:.3f}s sampled={sampled} hand_frames={hand_frames}")
    return frames, frames_xyzc


def predict_sentence_v2_direct(frames_xyzc: list, restricted: bool = False) -> tuple[str | None, float, list[dict]]:
    labels = state.sequence_labels.get("sentence_v2") or []
    model = state.sequence_models.get("sentence_v2")
    if not model or not labels or not frames_xyzc:
        return None, 0.0, []

    seq = state.np.stack([frame.reshape(-1) for frame in frames_xyzc]).astype(state.np.float32)
    valid_length = min(len(seq), 128)
    padded = state.np.zeros((128, 300), dtype=state.np.float32)
    padded[:valid_length] = seq[:valid_length]

    with state.torch.no_grad():
        x = state.torch.from_numpy(padded).float().unsqueeze(0)
        vl = state.torch.tensor([valid_length], dtype=state.torch.long)
        logits = model(x, vl)[0].numpy()
        if restricted:
            mask = state.np.full_like(logits, -state.np.inf)
            for idx in state.SCENARIO_SEN_INDICES:
                if 0 <= idx < logits.shape[0]:
                    mask[idx] = 0.0
            logits = logits + mask
        probs = softmax(logits / max(float(getattr(state, "TEMPERATURE_SENTENCE", 1.0)), 1e-6))

    order = state.np.argsort(probs)[::-1]
    top = [
        {"label": labels[int(i)], "confidence": float(probs[int(i)])}
        for i in order[:5]
    ]
    return labels[int(order[0])], float(probs[int(order[0])]), top


def print_result(name: str, label: str | None, conf: float, top: list[dict], tta_count: int) -> None:
    target_rank = next((idx + 1 for idx, item in enumerate(top) if item.get("label") == TARGET), None)
    target_conf = next((float(item.get("confidence") or 0.0) for item in top if item.get("label") == TARGET), 0.0)
    print(f"{name}: label={label} conf={conf:.4f} tta={tta_count} target_top_rank={target_rank} target_top_conf={target_conf:.4f}")
    print("top:", [(item.get("label"), round(float(item.get("confidence") or 0.0), 4), item.get("display_label")) for item in top])


def main() -> None:
    ensure_models_loaded()
    print("models:", sorted(state.sequence_models.keys()))
    print("scenario_sen_indices:", len(state.SCENARIO_SEN_INDICES))
    print("target_index:", (state.sequence_labels.get("sentence_v2") or []).index(TARGET))

    for video in VIDEOS:
        frames, frames_xyzc = extract_frames(video)
        label, conf, top, tta = predict_sequence_frames(
            "sentence_v2",
            frames,
            SEQ_LEN,
            getattr(state, "TEMPERATURE_SENTENCE", 1.0),
            use_tta=True,
        )
        print_result("sentence_full", label, conf, top, tta)

        label, conf, top, tta = predict_sequence_frames(
            "sentence_v2",
            frames,
            SEQ_LEN,
            getattr(state, "TEMPERATURE_SENTENCE", 1.0),
            use_tta=True,
            restrict_indices=state.SCENARIO_SEN_INDICES,
        )
        print_result("sentence_scenario_restricted", label, conf, top, tta)

        label, conf, top = predict_sentence_v2_direct(frames_xyzc, restricted=False)
        print_result("sentence_v2_direct_xyzc_full", label, conf, top, 1)

        label, conf, top = predict_sentence_v2_direct(frames_xyzc, restricted=True)
        print_result("sentence_v2_direct_xyzc_restricted", label, conf, top, 1)

        dual = predict_dual_scenario(
            frames,
            SEQ_LEN,
            getattr(state, "TEMPERATURE_SENTENCE", 1.0),
            use_tta=True,
        )
        sentence = dual.get("sentence") or {}
        word = dual.get("word") or {}
        print(
            "dual:",
            {
                "sentence": sentence.get("label"),
                "sentence_conf": round(float(sentence.get("confidence") or 0.0), 4),
                "word": word.get("label"),
                "word_conf": round(float(word.get("confidence") or 0.0), 4),
                "lookup_key": dual.get("lookup_key"),
                "scenario_text": dual.get("scenario_text"),
            },
        )


if __name__ == "__main__":
    main()
