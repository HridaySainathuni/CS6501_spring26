"""Topic 6 Exercise 2: Video-surveillance agent.
Extract frames every 2 seconds, ask LLaVA if there is a person in the scene.
Writes times at which a person enters/exits. Requires: pip install ollama opencv-python
Run: ollama pull llava
     python video_surveillance_agent.py <video.mp4> [--verbose] [--max-image-size 384]
Frames are resized (longest side = max-image-size) before sending to reduce GPU use and avoid 500 errors.
"""
import sys
import os
import re

_llava_fallback_printed = [False]

def extract_frames(video_path: str, interval_sec: float = 2.0):
    """Yield (frame_index, frame_bgr, timestamp_sec) every interval_sec."""
    try:
        import cv2
    except ImportError:
        print("Install opencv-python: pip install opencv-python")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval_frames = max(1, int(fps * interval_sec))
    frame_num = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval_frames == 0:
            t_sec = frame_num / fps
            yield idx, frame, t_sec
            idx += 1
        frame_num += 1
    cap.release()

def _resize_frame_if_large(frame_bgr, max_size=384):
    """Resize frame so longest side is max_size to reduce GPU memory and avoid Ollama 500 errors."""
    import cv2
    h, w = frame_bgr.shape[:2]
    if max(h, w) <= max_size:
        return frame_bgr
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _call_vision_model(frame_bgr, model, max_image_size, verbose):
    """Single attempt: resize, write temp image, call Ollama. Returns (text, None) or (None, error_str)."""
    import tempfile
    import ollama
    import cv2
    frame_bgr = _resize_frame_if_large(frame_bgr, max_image_size)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.close(fd)
        cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
        abs_path = os.path.abspath(path)
        # Describe-style prompt works for both llava and moondream (strict yes/no can return empty on moondream)
        prompt = "Describe what you see in one short sentence. If there are people or a person, say so."
        r = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [abs_path],
            }],
            options={"num_ctx": 512},
        )
        raw = r.get("message") or {}
        text = (raw.get("content") or "").strip().lower()
        return text, None
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def ask_llava_has_person(frame_bgr, model="llava", verbose=False, max_image_size=384):
    """Send frame to vision model; return (bool, raw_response_str). Retries with smaller image or moondream on failure."""
    text, err = _call_vision_model(frame_bgr, model, max_image_size, verbose)
    if err and max_image_size > 128:
        text, err = _call_vision_model(frame_bgr, model, max(max_image_size // 2, 128), verbose)
    # If llava failed, try smaller moondream (1.8B) which often works when llava OOMs
    if err and model == "llava":
        if not _llava_fallback_printed[0]:
            _llava_fallback_printed[0] = True
            print("    [llava failed, trying moondream (run: ollama pull moondream)]")
        text2, err2 = _call_vision_model(frame_bgr, "moondream", max_image_size, verbose)
        if not err2:
            text, err = text2, None
    if err:
        if verbose:
            print(f"    [Vision error] {err}")
        return False, err
    if not text:
        return False, text
    if verbose:
        print(f"    [raw] {repr(text)}")
    if "error" in text or "500" in text or "unexpectedly stopped" in text or "resource" in text:
        return False, "[Ollama error] " + text[:80]
    if "no person" in text or "no people" in text or "nobody" in text or "no one " in text or "no one." in text:
        return False, text
    if text.startswith("no") and "no one" not in text and "nobody" not in text:
        return False, text
    if "yes" in text:
        return True, text
    for word in ("person", "people", "human", "man", "men", "woman", "women", "someone", "individual", "figure"):
        if word in text:
            return True, text
    return False, text

def main():
    args = [a for a in sys.argv[1:] if a and not a.startswith("-")]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    max_size = 384
    model = "llava"
    max_frames = None  # None = process all frames
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--max-image-size" and i + 1 < len(sys.argv):
            try:
                max_size = int(sys.argv[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            i += 2
            continue
        if sys.argv[i] == "--max-frames" and i + 1 < len(sys.argv):
            try:
                max_frames = int(sys.argv[i + 1])
            except ValueError:
                max_frames = None
            i += 2
            continue
        i += 1
    video_path = args[0] if args else None
    if not video_path or not os.path.isfile(video_path):
        print("Usage: python video_surveillance_agent.py <video.mp4> [--verbose] [--max-image-size 384] [--model llava|moondream]")
        return
    try:
        import ollama
        import cv2
    except ImportError as e:
        print("Install ollama and opencv-python:", e)
        return
    person_times = []  # (timestamp_sec, has_person)
    print("Extracting frames every 2 seconds and asking LLaVA if person is in scene...")
    for idx, frame, t_sec in extract_frames(video_path, 2.0):
        if max_frames is not None and idx >= max_frames:
            break
        has_person, raw = ask_llava_has_person(frame, model=model, verbose=verbose, max_image_size=max_size)
        person_times.append((t_sec, has_person))
        raw_short = (raw[:60] + "…") if len(raw) > 60 else raw
        print(f"  t={t_sec:.1f}s  person={has_person}  (LLaVA: {repr(raw_short)})")
    # Compute enter/exit times
    in_scene = False
    enter_times = []
    exit_times = []
    for t, has in person_times:
        if has and not in_scene:
            enter_times.append(t)
            in_scene = True
        elif not has and in_scene:
            exit_times.append(t)
            in_scene = False
    if in_scene and person_times:
        exit_times.append(person_times[-1][0])
    print("\n--- Results ---")
    print("Person enters at (seconds):", enter_times)
    print("Person exits at (seconds):", exit_times)

if __name__ == "__main__":
    main()
