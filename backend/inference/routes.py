from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_from_directory

from auth.routes import login_required
from config import Config
from inference.model_loader import ensure_models_loaded
import inference.model_state as model_state
from inference.model_state import (
    GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS,
    cv2,
    gloss_to_text_last_called_at,
    Image,
    memory_predictions,
    mp_holistic_instance,
    mp_holistic_lock,
    np,
    sequence_models,
    torch,
    sequence_to_tensor,
    mediapipe_landmarks_to_frame,
)
from inference.predictor import (
    display_label_for,
    get_session_misses,
    get_session_window,
    landmarks_have_points,
    landmarks_payload_to_frame,
    normalize_model_type,
    predict_dual_scenario,
    predict_sequence_frames,
    set_session_misses,
)
from src.services.gloss_to_text_service import gloss_to_text
from src.utils.config import load_config

inference_bp = Blueprint("inference", __name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(ROOT_DIR / "config" / "web_demo.yaml")

try:
    _config = load_config(CONFIG_PATH)
except Exception:
    _config = {
        "paths": {"checkpoints_dir": "src/models"},
        "data": {"sequence_length": 32},
        "preprocess": {"feature_dims": 3, "normalize": True},
        "realtime": {
            "landmark_layout": "mediapipe_xyz",
            "confidence_threshold": 0.35,
            "stable_min_count": 1,
            "max_missing_frames": 3,
            "temperature": 0.9,
            "tta_enabled": True,
        },
    }

SCENARIO_LOOKUP_PATH = Path(__file__).resolve().parent / "scenario_lookup.json"

def _load_scenario_lookup() -> dict[str, str]:
    if SCENARIO_LOOKUP_PATH.exists():
        with SCENARIO_LOOKUP_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return {}


_SCENARIO_LOOKUP = _load_scenario_lookup()

@inference_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@inference_bp.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "service": "ksl-backend"}), 200

def _generate_mjpeg_frames(camera_index: int = 0):
    if cv2 is None or np is None:
        yield b"--frame\r\nContent-Type: text/plain\r\n\r\nCamera dependencies not installed\r\n"
        return
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (130, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()

@inference_bp.route("/video_feed", methods=["GET"])
@login_required
def video_feed():
    camera_index = int(request.args.get("camera", 0))
    return Response(_generate_mjpeg_frames(camera_index),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@inference_bp.route("/validation_demos/<path:filename>", methods=["GET"])
def validation_demo_video(filename: str):
    return send_from_directory(ROOT_DIR / "data" / "raw" / "validation_mp4", filename)

@inference_bp.route("/api/gloss_to_text", methods=["POST"])
def api_gloss_to_text():
    import re
    ensure_models_loaded()
    data = request.get_json(force=True, silent=True) or {}
    client_id = str(data.get("client_id") or request.remote_addr or "default")
    now = time.monotonic()
    elapsed = now - gloss_to_text_last_called_at.get(client_id, 0.0)
    if elapsed < GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS:
        return jsonify({"error": "gloss_to_text API는 6초에 한 번만 호출할 수 있습니다.",
                        "retry_after": round(GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS - elapsed, 2)}), 429
    gloss_to_text_last_called_at[client_id] = now

    gloss_val = data.get("gloss", "")
    if isinstance(gloss_val, list):
        gloss = " + ".join(str(i).strip() for i in gloss_val if str(i).strip())
    else:
        gloss = str(gloss_val).strip()
    if not gloss:
        return jsonify({"error": "gloss field is required"}), 400

    gloss_tokens = [t.strip() for t in re.split(r"\s*\+\s*", gloss) if t.strip()]
    lookup_candidates: list[str] = []
    if len(gloss_tokens) >= 2:
        for left in gloss_tokens:
            for right in gloss_tokens:
                if left != right:
                    lookup_candidates.append(f"{left}+{right}")
    lookup_candidates.extend(gloss_tokens)
    for key in lookup_candidates:
        if key in _SCENARIO_LOOKUP:
            return jsonify({"gloss": gloss, "text": _SCENARIO_LOOKUP[key],
                            "lookup_hit": True, "lookup_key": key}), 200

    provider = data.get("provider")
    model   = data.get("model")
    result = gloss_to_text(
        gloss,
        provider=str(provider).strip() if provider else None,
        model=str(model).strip() if model else None,
    )
    return jsonify({"gloss": gloss, "text": result}), 200

@inference_bp.route("/api/predict", methods=["POST"])
def predict():
    from flask import session as flask_session
    flask_session.modified = False
    t0 = time.perf_counter()
    try:
        ensure_models_loaded()
        if cv2 is None or Image is None or np is None:
            return jsonify({"error": "Vision dependencies not installed"}), 503
        if model_state.mp_holistic_instance is None:
            return jsonify({"error": "MediaPipe not available"}), 503

        force_finalize  = request.form.get("force_finalize", "false").lower() in {"true", "1", "yes"}
        if not force_finalize and "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        model_type      = normalize_model_type(request.form.get("model_type", "sequence"))
        landmark_layout = request.form.get("landmark_layout", "mediapipe_xyz")
        if landmark_layout != "mediapipe_xyz":
            return jsonify({"error": f"Unsupported landmark_layout: {landmark_layout}"}), 400

        client_id           = request.form.get("client_id", "default")
        frame_id            = request.form.get("frame_id", "")
        seq_len             = int(_config["data"]["sequence_length"])
        confidence_threshold = float(request.form.get("confidence_threshold", _config.get("realtime", {}).get("confidence_threshold", 0.10)))
        window_size         = max(8, min(int(request.form.get("window_size", _config.get("realtime", {}).get("window_size", seq_len))), seq_len))
        stable_min_count    = int(request.form.get("stable_min_count", _config.get("realtime", {}).get("stable_min_count", 2)))
        max_missing_frames  = int(request.form.get("max_missing_frames", _config.get("realtime", {}).get("max_missing_frames", 3)))
        min_segment_frames  = int(request.form.get("min_segment_frames", _config.get("realtime", {}).get("min_segment_frames", 8)))
        temperature         = float(request.form.get("temperature", _config.get("realtime", {}).get("temperature", 0.9)))
        use_tta             = request.form.get("tta_enabled", str(_config.get("realtime", {}).get("tta_enabled", True))).lower() not in {"false", "0", "no"}
        run_model           = request.form.get("run_model", "true").lower() != "false"
        upload_bytes        = int(float(request.form.get("upload_bytes", 0) or 0))
        scenario_mode       = request.form.get("scenario_mode", "false").lower() in {"true", "1", "yes", "resident"}
        demo_video_time_sec   = request.form.get("demo_video_time_sec")
        demo_segment_start_sec = request.form.get("demo_segment_start_sec")
        demo_finalize_reason  = request.form.get("demo_finalize_reason") or request.form.get("live_finalize_reason")

        window = get_session_window(client_id)

        base_pred: dict = {
            "label": None, "confidence": 0.0, "top_predictions": [],
            "window_progress": len(window), "window_size": window_size,
            "missing_frames": get_session_misses(client_id),
            "max_missing_frames": max_missing_frames,
            "min_segment_frames": min_segment_frames,
            "sequence_length": seq_len, "temperature": temperature,
            "tta_enabled": use_tta, "run_model": run_model,
            "force_finalize": force_finalize, "frame_id": frame_id,
            "landmark_layout": landmark_layout, "model_type": model_type,
            "scenario_mode": scenario_mode, "processing_mode": "server_mediapipe",
            "upload_bytes": upload_bytes,
            "demo_video_time_sec": demo_video_time_sec,
            "demo_segment_start_sec": demo_segment_start_sec,
            "demo_finalize_reason": demo_finalize_reason,
        }

        def _run_finalize(seg_frames: list, pred: dict) -> None:
            if len(seg_frames) >= min_segment_frames:
                label, conf, top, tta_n = predict_sequence_frames(model_type, seg_frames, seq_len, temperature, use_tta)
                pred.update({"top_predictions": top, "raw_label": label,
                             "display_label": display_label_for(label),
                             "raw_confidence": conf, "tta_count": tta_n,
                             "confidence": conf, "window_filled": True,
                             "segment_finalized": True, "segment_frames": len(seg_frames)})
                if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                    try:
                        dual = predict_dual_scenario(seg_frames, seq_len, temperature, use_tta)
                        pred["scenario"] = dual
                        if dual.get("scenario_text"):
                            pred["scenario_text"] = dual["scenario_text"]
                        # 임시 디버그 로그 추가
                        print(f"[dual] word={dual.get('word', {}).get('label')} word_conf={dual.get('word', {}).get('confidence', 0):.3f}")
                        print(f"[dual] sen={dual.get('sentence', {}).get('label')} sen_conf={dual.get('sentence', {}).get('confidence', 0):.3f}")
                        print(f"[dual] lookup_hit={dual.get('lookup_hit')} scenario_text={dual.get('scenario_text')}")
                        print(f"[dual] fusion_candidates={dual.get('fusion_candidates', [])[:2]}")
                    except Exception as exc:
                        pred["scenario_error"] = str(exc)
                        print(f"[dual] error: {exc}")
                if label and conf >= confidence_threshold:
                    pred["label"] = label
                    pred["below_threshold"] = False
                else:
                    pred["label"] = None
                    pred["below_threshold"] = True
                    pred["status"] = "수어 구간은 잡혔지만 신뢰도가 낮습니다."
            elif seg_frames:
                pred.update({"segment_finalized": True, "segment_frames": len(seg_frames),
                             "status": "수어 구간이 너무 짧아 예측하지 않았습니다."})

        prediction = dict(base_pred)

        if force_finalize:
            seg = list(window); window.clear()
            set_session_misses(client_id, 0); memory_predictions.pop(client_id, None)
            prediction.update({"window_progress": 0, "segmenting": False, "missing_frames": 0,
                               "has_hand": None, "has_pose": None,
                               "landmarks": {"left_hand": [], "right_hand": [], "pose": []},
                               "segment_frames": None, "input_size": None, "processed_size": None})
            _run_finalize(seg, prediction)
        else:
            frame_file = request.files["frame"]
            img_pil = Image.open(io.BytesIO(frame_file.read())).convert("RGB")
            img_rgb = np.array(img_pil)
            h, w = img_rgb.shape[:2]
            scale = min(640 / w, 360 / h, 1)
            if scale < 1:
                img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ph, pw = img_rgb.shape[:2]

            with mp_holistic_lock:
                results = model_state.mp_holistic_instance.process(img_rgb)

            has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            has_pose = bool(results.pose_landmarks)
            landmarks = {"left_hand": [], "right_hand": [], "pose": []}
            if results.left_hand_landmarks:
                landmarks["left_hand"] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            if results.right_hand_landmarks:
                landmarks["right_hand"] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            if results.pose_landmarks:
                landmarks["pose"] = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

            prediction.update({"has_hand": has_hand, "has_pose": has_pose, "landmarks": landmarks,
                               "input_size": {"width": w, "height": h},
                               "processed_size": {"width": pw, "height": ph}})

            can_collect = mediapipe_landmarks_to_frame is not None
            if has_hand and can_collect:
                set_session_misses(client_id, 0)
                window.append(mediapipe_landmarks_to_frame(results))
                prediction.update({"missing_frames": 0, "window_progress": len(window),
                                   "window_filled": False, "segmenting": True,
                                   "status": "수어 단어 구간 수집 중"})
            elif has_hand:
                set_session_misses(client_id, 0)
                prediction.update({"missing_frames": 0, "window_progress": len(window),
                                   "window_filled": False, "segmenting": True,
                                   "status": "MediaPipe 랜드마크 감지 중"})
            else:
                misses = get_session_misses(client_id) + 1
                set_session_misses(client_id, misses)
                prediction["missing_frames"] = misses
                if misses > max_missing_frames:
                    seg = list(window); window.clear()
                    memory_predictions.pop(client_id, None)
                    prediction.update({"window_progress": 0, "segmenting": False})
                    _run_finalize(seg, prediction)
                else:
                    prediction["window_progress"] = len(window)

        prediction["process_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if prediction.get("segment_finalized"):
            print(
                f"[predict] finalized"
                f" label={prediction.get('label')}"
                f" raw_label={prediction.get('raw_label')}"
                f" confidence={prediction.get('confidence', 0):.3f}"
                f" frames={prediction.get('segment_frames')}"
                f" scenario_mode={scenario_mode}"
                f" scenario={prediction.get('scenario', {}).get('lookup_key') if prediction.get('scenario') else None}"
                f" top={prediction.get('top_predictions', [])[:2]}"
            )
        return jsonify({"prediction": prediction, "frame_id": frame_id}), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@inference_bp.route("/api/predict_landmarks", methods=["POST"])
def predict_landmarks():
    t0 = time.perf_counter()
    try:
        ensure_models_loaded()
        if np is None or torch is None or sequence_to_tensor is None:
            return jsonify({"error": "Model dependencies not installed"}), 503

        data           = request.get_json(silent=True) or {}
        force_finalize = str(data.get("force_finalize", "false")).lower() in {"true", "1", "yes"}
        landmarks      = data.get("landmarks") or {}
        if not force_finalize and not isinstance(landmarks, dict):
            return jsonify({"error": "landmarks must be an object"}), 400

        model_type     = normalize_model_type(str(data.get("model_type", "sequence")))
        landmark_layout = str(data.get("landmark_layout", "mediapipe_xyz"))
        if landmark_layout != "mediapipe_xyz":
            return jsonify({"error": f"Unsupported landmark_layout: {landmark_layout}"}), 400

        client_id            = str(data.get("client_id", "default"))
        frame_id             = str(data.get("frame_id", ""))
        seq_len              = int(_config["data"]["sequence_length"])
        confidence_threshold = float(data.get("confidence_threshold", _config.get("realtime", {}).get("confidence_threshold", 0.35)))
        window_size          = max(8, min(int(data.get("window_size", _config.get("realtime", {}).get("window_size", seq_len))), seq_len))
        stable_min_count     = int(data.get("stable_min_count", _config.get("realtime", {}).get("stable_min_count", 2)))
        max_missing_frames   = int(data.get("max_missing_frames", _config.get("realtime", {}).get("max_missing_frames", 3)))
        min_segment_frames   = int(data.get("min_segment_frames", _config.get("realtime", {}).get("min_segment_frames", 8)))
        temperature          = float(data.get("temperature", _config.get("realtime", {}).get("temperature", 0.9)))
        use_tta              = str(data.get("tta_enabled", _config.get("realtime", {}).get("tta_enabled", True))).lower() not in {"false", "0", "no"}
        run_model            = str(data.get("run_model", "true")).lower() != "false"
        scenario_mode        = str(data.get("scenario_mode", "false")).lower() in {"true", "1", "yes", "resident"}
        client_mediapipe_ms  = float(data.get("client_mediapipe_ms") or 0.0)
        client_payload_bytes = int(float(data.get("upload_bytes") or data.get("client_payload_bytes") or 0))
        demo_video_time_sec   = data.get("demo_video_time_sec")
        demo_segment_start_sec = data.get("demo_segment_start_sec")
        demo_finalize_reason  = data.get("demo_finalize_reason") or data.get("live_finalize_reason")

        window = get_session_window(client_id)

        prediction: dict = {
            "label": None, "confidence": 0.0,
            "has_hand": None if force_finalize else False,
            "has_pose": None if force_finalize else False,
            "landmarks": landmarks if isinstance(landmarks, dict) else {"left_hand": [], "right_hand": [], "pose": []},
            "window_progress": len(window), "window_size": window_size,
            "missing_frames": get_session_misses(client_id),
            "max_missing_frames": max_missing_frames,
            "min_segment_frames": min_segment_frames,
            "sequence_length": seq_len, "temperature": temperature,
            "tta_enabled": use_tta, "run_model": run_model,
            "force_finalize": force_finalize, "top_predictions": [],
            "frame_id": frame_id, "landmark_layout": landmark_layout,
            "model_type": model_type, "scenario_mode": scenario_mode,
            "processing_mode": "client_mediapipe",
            "client_mediapipe_ms": client_mediapipe_ms,
            "upload_bytes": client_payload_bytes,
            "stable_min_count": stable_min_count,
        }

        def _run_finalize(seg_frames: list, pred: dict) -> None:
            if len(seg_frames) >= min_segment_frames:
                label, conf, top, tta_n = predict_sequence_frames(model_type, seg_frames, seq_len, temperature, use_tta)
                pred.update({"top_predictions": top, "raw_label": label,
                             "display_label": display_label_for(label),
                             "raw_confidence": conf, "tta_count": tta_n,
                             "confidence": conf, "window_filled": True,
                             "segment_finalized": True, "segment_frames": len(seg_frames)})
                if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                    try:
                        dual = predict_dual_scenario(seg_frames, seq_len, temperature, use_tta)
                        pred["scenario"] = dual
                        if dual.get("scenario_text"):
                            pred["scenario_text"] = dual["scenario_text"]
                    except Exception as exc:
                        pred["scenario_error"] = str(exc)
                if label and conf >= confidence_threshold:
                    pred["label"] = label; pred["below_threshold"] = False
                else:
                    pred["label"] = None; pred["below_threshold"] = True
                    pred["status"] = "수어 구간은 잡혔지만 신뢰도가 낮습니다."
            elif seg_frames:
                pred.update({"segment_finalized": True, "segment_frames": len(seg_frames),
                             "status": "수어 구간이 너무 짧아 예측하지 않았습니다."})

        if force_finalize:
            prediction.update({"demo_video_time_sec": demo_video_time_sec,
                               "demo_segment_start_sec": demo_segment_start_sec,
                               "demo_finalize_reason": demo_finalize_reason})
            seg = list(window); window.clear()
            set_session_misses(client_id, 0); memory_predictions.pop(client_id, None)
            prediction.update({"window_progress": 0, "segmenting": False, "missing_frames": 0})
            _run_finalize(seg, prediction)
        else:
            has_hand = landmarks_have_points(landmarks.get("left_hand")) or landmarks_have_points(landmarks.get("right_hand"))
            has_pose = landmarks_have_points(landmarks.get("pose"))
            prediction.update({"has_hand": has_hand, "has_pose": has_pose})
            if has_hand:
                set_session_misses(client_id, 0)
                window.append(landmarks_payload_to_frame(landmarks))
                prediction.update({"missing_frames": 0, "window_progress": len(window),
                                   "window_filled": False, "segmenting": True,
                                   "status": "클라이언트 MediaPipe 랜드마크 수집 중"})
            else:
                misses = get_session_misses(client_id) + 1
                set_session_misses(client_id, misses)
                prediction["missing_frames"] = misses
                if misses > max_missing_frames:
                    seg = list(window); window.clear()
                    memory_predictions.pop(client_id, None)
                    prediction.update({"window_progress": 0, "segmenting": False})
                    _run_finalize(seg, prediction)
                else:
                    prediction["window_progress"] = len(window)

        prediction["process_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({"frame_id": frame_id, "prediction": prediction}), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@inference_bp.route("/api/kakao/login", methods=["GET"])
def kakao_login():
    from urllib.parse import urlencode
    from flask import redirect as flask_redirect

    redirect_uri = request.args.get("redirect_uri", "")
    if not Config.KAKAO_REST_API_KEY:
        return jsonify({"error": "KAKAO_REST_API_KEY가 설정되어 있지 않습니다."}), 500
    if not redirect_uri:
        return jsonify({"error": "redirect_uri가 필요합니다."}), 400
    params = urlencode({"client_id": Config.KAKAO_REST_API_KEY,
                        "redirect_uri": redirect_uri, "response_type": "code"})
    return flask_redirect(f"https://kauth.kakao.com/oauth/authorize?{params}")


@inference_bp.route("/api/kakao/token", methods=["POST"])
def kakao_token():
    import requests as _requests
    data = request.get_json(silent=True) or {}
    code         = str(data.get("code", "")).strip()
    redirect_uri = str(data.get("redirect_uri", "")).strip()
    if not Config.KAKAO_REST_API_KEY:
        return jsonify({"error": "KAKAO_REST_API_KEY가 설정되어 있지 않습니다."}), 500
    if not code:
        return jsonify({"error": "카카오 인가 코드(code)가 필요합니다."}), 400
    if not redirect_uri:
        return jsonify({"error": "redirect_uri가 필요합니다."}), 400
    try:
        payload = {"grant_type": "authorization_code", "client_id": Config.KAKAO_REST_API_KEY,
                   "redirect_uri": redirect_uri, "code": code}
        if Config.KAKAO_CLIENT_SECRET:
            payload["client_secret"] = Config.KAKAO_CLIENT_SECRET
        resp = _requests.post("https://kauth.kakao.com/oauth/token",
                              headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
                              data=payload, timeout=10)
        result = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": result.get("error_description") or str(result)}), resp.status_code
        return jsonify({"access_token": result.get("access_token"),
                        "refresh_token": result.get("refresh_token"),
                        "expires_in": result.get("expires_in")}), 200
    except Exception as exc:
        return jsonify({"error": f"카카오 토큰 발급 실패: {exc}"}), 502
