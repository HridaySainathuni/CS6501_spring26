# Topic 6: Vision-Language Models (VLM)

## Table of Contents

1. [Exercise 1: Vision-language chat agent](#exercise-1)
2. [Exercise 2: Video-surveillance agent](#exercise-2)
3. [Setup](#setup)

## Setup

- Install Ollama and run: `ollama pull llava`
- Pip: `pip install -r requirements.txt` (ollama, opencv-python)

## Exercise 1: Vision-Language LangGraph-style chat agent

**vlma_chat_agent.py** – Multi-turn conversation about one uploaded image. Uses Ollama LLaVA; no separate LangGraph dependency (same pattern: maintain message history, send image with first user message).

```bash
ollama pull llava
python vlma_chat_agent.py path/to/photo.jpg
```

Then ask questions about the image (e.g. "Describe this image", "What color is the car?").

## Exercise 2: Video-surveillance agent

**video_surveillance_agent.py** – Extracts frames every 2 seconds from a video, sends each frame to a vision model with the prompt "Is there a person in this image?", and reports times at which a person enters and exits the scene.

```bash
python video_surveillance_agent.py video.mp4
```

If LLaVA returns 500 (e.g. "model runner stopped" on limited GPU), the script will retry with a smaller image and, if you have it, **moondream** (lighter vision model). Install it with: `ollama pull moondream`. You can also force moondream: `python video_surveillance_agent.py video.mp4 --model moondream`. Use `--max-image-size 256` to reduce memory further.

Output: list of (time_sec, person_detected) and computed enter/exit times in seconds.

Optional (per assignment): Use a 2-minute clip of an empty space where a person enters and exits; or connect to webcam with `cv2.VideoCapture(0)` and sample at intervals.

## Terminal output (logs)

- `log_vlma_chat_agent.txt` – Multi-turn vision-language chat (image)
- `log_video_surveillance_agent.txt` – Video surveillance run (frames, enter/exit times)

From repo root: `py -3 run_all_logs.py --topic 6`

## File index

| File | Purpose |
|------|--------|
| vlma_chat_agent.py | Multi-turn chat about an image (Ollama LLaVA/moondream) |
| video_surveillance_agent.py | Frames every 2s, person in scene, enter/exit times |
| README.md | This file |
