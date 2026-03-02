"""Run scripts for all topics and save terminal logs to topic directories.
Usage: py -3 run_all_logs.py [--topic 2|3|4|5|6|llm]   (default: all)
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd, cwd=None, stdin_text=None, log_path=None, timeout=600):
    """Run command; optionally pipe stdin and write stdout+stderr to log_path."""
    cwd = cwd or REPO_ROOT
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    use_shell = sys.platform == "win32"  # py -3 needs shell on Windows
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            input=stdin_text.encode("utf-8") if stdin_text else None,
            capture_output=True,
            timeout=timeout,
            env=env,
            shell=use_shell,
        )
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        combined = out + ("\n" + err if err else "")
    except subprocess.TimeoutExpired:
        combined = "[TIMEOUT after %s s]\n" % timeout
    except Exception as e:
        combined = "[ERROR] %s\n" % e
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(combined)
        print("  -> %s" % log_path)
    return combined

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topic", choices=["2", "3", "4", "5", "6", "llm"], default=None, help="Run only this topic (default: all)")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()
    topics = [args.topic] if args.topic else ["2", "3", "4", "5", "6", "llm"]

    py = "py -3" if sys.platform == "win32" else "python3"

    for topic in topics:
        if topic == "llm":
            # Running an LLM: quick MMLU (2 subjects), no chat (interactive)
            d = os.path.join(REPO_ROOT, "Running an LLM")
            if not os.path.isdir(d):
                continue
            log = os.path.join(d, "llama_mmlu_eval_log.txt")
            cmd = "%s llama_mmlu_eval.py --max-subjects 2" % py
            if args.dry_run:
                print("[LLM] %s > %s" % (cmd, log))
            else:
                run_cmd(cmd, cwd=d, log_path=log, timeout=900)
            continue

        if topic == "2":
            # Topic 2: LangGraph scripts (interactive - pipe a few lines then quit)
            d = os.path.join(REPO_ROOT, "Topic2Frameworks")
            inputs = {
                "task1_verbose_quiet_tracing": "verbose\nWhat is 2+2?\nquiet\nquit",
                "task2_empty_input_handling": "\n\nhello\nquit",
                "task3_parallel_models": "Tell me a short joke\nquit",
                "task4_conditional_model_routing": "What is the capital of France?\nquit",
                "task5_chat_history": "Hello\nHi again\nquit",
                "task6_chat_history_with_model_switching": "What is the best ice cream?\nHey Qwen, what do you think?\nquit",
                "task7_checkpointing_crash_recovery": "Hello\nquit",
            }
            # Also base agent
            for name, stdin in [
                ("langgraph_simple_agent", "Hello\nquit"),
                *list(inputs.items()),
            ]:
                script = name + ".py"
                if not os.path.isfile(os.path.join(d, script)):
                    continue
                log = os.path.join(d, "log_%s.txt" % name)
                cmd = "%s %s" % (py, script)
                if args.dry_run:
                    print("[T2] %s < stdin > %s" % (cmd, log))
                else:
                    run_cmd(cmd, cwd=d, stdin_text=stdin, log_path=log, timeout=420)

        elif topic == "3":
            # Topic 3: non-interactive (task2, task3, task4, task5) + optional task1 (Ollama)
            d = os.path.join(REPO_ROOT, "Topic3Tools")
            for script, extra, logname in [
                ("task2_openai_gpt4o_mini_test.py", "", "log_task2_openai_test.txt"),
                ("task3_manual_tool_calculator.py", "", "log_task3_manual_calculator.txt"),
                ("task4_langgraph_tools.py", '--query "How many s in Mississippi?"', "log_task4_langgraph_tools.txt"),
                ("task5_conversation_checkpointing.py", "", "log_task5_checkpointing.txt"),
            ]:
                path = os.path.join(d, script)
                if not os.path.isfile(path):
                    continue
                log = os.path.join(d, logname)
                cmd = "%s %s %s" % (py, script, extra)
                if args.dry_run:
                    print("[T3] %s > %s" % (cmd.strip(), log))
                else:
                    run_cmd(cmd, cwd=d, log_path=log, timeout=120)

        elif topic == "4":
            # Topic 4: interactive - one query then exit
            d = os.path.join(REPO_ROOT, "Topic4Exploring")
            for script, stdin in [
                ("toolnode_example.py", "What is 15 + 27?\nexit"),
                ("react_agent_example.py", "What is 15 + 27?\nexit"),
                ("two_hour_agent_project.py", "What time is it?\nexit"),
            ]:
                path = os.path.join(d, script)
                if not os.path.isfile(path):
                    continue
                log = os.path.join(d, "log_%s.txt" % os.path.splitext(script)[0])
                cmd = "%s %s" % (py, script)
                if args.dry_run:
                    print("[T4] %s < stdin > %s" % (cmd, log))
                else:
                    run_cmd(cmd, cwd=d, stdin_text=stdin, log_path=log, timeout=120)

        elif topic == "5":
            # Topic 5: RAG - non-interactive
            d = os.path.join(REPO_ROOT, "Topic5RAG")
            for script, extra, logname in [
                ("rag_pipeline.py", "--query 'What is the correct spark plug gap for a Model T?'", "log_rag_pipeline.txt"),
                ("exercise1_no_rag_vs_rag.py", "", "log_exercise1_no_rag_vs_rag.txt"),
                ("exercise4_topk.py", "", "log_exercise4_topk.txt"),
            ]:
                path = os.path.join(d, script)
                if not os.path.isfile(path):
                    continue
                log = os.path.join(d, logname)
                cmd = "%s %s %s" % (py, script, extra)
                if args.dry_run:
                    print("[T5] %s > %s" % (cmd.strip(), log))
                else:
                    run_cmd(cmd, cwd=d, log_path=log, timeout=180)

        elif topic == "6":
            # Topic 6: VLM - chat agent (stdin) + video (with --max-frames 2 for quick log)
            d = os.path.join(REPO_ROOT, "Topic6VLM")
            # Find an image for chat (use first frame from video or any image)
            img = os.path.join(d, "sample_image.jpg")
            if not os.path.isfile(img):
                # Create a tiny placeholder image with OpenCV if available
                try:
                    import cv2
                    import numpy as np
                    arr = np.zeros((50, 50, 3), dtype=np.uint8)
                    arr[:] = (200, 200, 200)
                    cv2.imwrite(img, arr)
                except Exception:
                    img = None
            if img and os.path.isfile(img):
                log = os.path.join(d, "log_vlma_chat_agent.txt")
                cmd = '%s vlma_chat_agent.py "%s" --model moondream' % (py, img)
                if args.dry_run:
                    print("[T6] %s < stdin > %s" % (cmd, log))
                else:
                    run_cmd(cmd, cwd=d, stdin_text="Describe this image.\nquit", log_path=log, timeout=120)
            # Video surveillance: use --max-frames 2 and existing .mov if present
            video = None
            for v in ["vecteezy_travellers-and-people-meeting-them-in-arrivals-hall_29108841.mov", "video.mp4", "test.mov"]:
                vpath = os.path.join(d, v)
                if os.path.isfile(vpath):
                    video = vpath
                    break
            if video:
                log = os.path.join(d, "log_video_surveillance_agent.txt")
                cmd = '%s video_surveillance_agent.py "%s" --max-frames 2 --model moondream' % (py, os.path.basename(video))
                if args.dry_run:
                    print("[T6] %s > %s" % (cmd, log))
                else:
                    run_cmd(cmd, cwd=d, log_path=log, timeout=180)

    print("Done.")

if __name__ == "__main__":
    main()
