"""Scales H3 videos up 2x with NVIDIA RTX Video Super Resolution (nvidia-vfx), in place.

H3Video.exe --upscale starts this with ComfyUI's Python as a resident helper, which keeps torch and the
model loaded (about 450 MB of VRAM) so every video after the first costs ~2 s instead of ~10 s:
    python -I RtxVideo.py --serve port parentPid stampFile
POST /upscale (body: mp4 path) queues a video, POST /result (same body) answers busy, the summary line
or "error: ...". It ends when the parent process is gone, after 15 idle minutes or when this file changes.
    python -I RtxVideo.py video.mp4    upscales one video and exits
ffmpeg decodes the frames, nvvfx upscales them on the GPU, ffmpeg encodes them like H3Video (libx264
crf 18) and copies the audio, then the result replaces the mp4. Its comment tag keeps the generated size
(480x832 upscaled with ...), StrictGen extends at that size. On a failure the video stays as it was.
"""
import os
import subprocess
import sys
import time

Encoding = ["-c:v", "libx264", "-crf", "18", "-preset", "faster", "-pix_fmt", "yuv420p"]
Tag = " upscaled with RTX Video Super Resolution"
IdleSeconds = 900


def loaded():
    import torch
    import nvvfx
    if not torch.cuda.is_available() or "RTX" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("RTX Video Super Resolution needs an NVIDIA RTX card")
    effect = nvvfx.VideoSuperRes(quality=nvvfx.QualityLevel.ULTRA)
    effect.output_width, effect.output_height = 1280, 720
    effect.load()
    return effect


def probe(path):
    probed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    width, height, rate = probed.stdout.strip().split(",")[:3]
    return int(width), int(height), rate


def upscaled(effect, path, scale=2):
    import torch
    started = time.perf_counter()
    width, height, rate = probe(path)
    wide, high = max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)
    effect.output_width, effect.output_height = wide, high
    output = path + ".rtx.mp4"
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
        f"{wide}x{high}", "-r", rate, "-i", "-", "-i", path, "-map", "0:v", "-map", "1:a?"] + Encoding +
        ["-c:a", "copy", "-metadata", f"comment={width}x{height}{Tag}", output], stdin=subprocess.PIPE)
    frame = width * height * 3
    frames = 0
    try:
        with torch.inference_mode():
            while True:
                data = decoder.stdout.read(frame)
                if len(data) < frame:
                    break
                pixels = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(height, width, 3)
                seen = pixels.cuda().permute(2, 0, 1).float().div_(255).contiguous()
                result = torch.from_dlpack(effect.run(seen).image)
                encoder.stdin.write(result.mul(255).round_().clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy().tobytes())
                frames += 1
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
        torch.cuda.empty_cache()
    if decoder.wait() != 0 or encoder.wait() != 0 or frames == 0:
        if os.path.exists(output):
            os.remove(output)
        raise RuntimeError("ffmpeg could not upscale " + path)
    os.replace(output, path)
    return (f"RTX Video Super Resolution {width}x{height} to {wide}x{high}, {frames} frames in "
        f"{time.perf_counter() - started:.1f} s")


def alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return code.value == 259


def serve(port, parent, stamp):
    import queue
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    jobs, results, lock, used = queue.Queue(), {}, threading.Lock(), [time.time()]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            path = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            with lock:
                if self.path == "/upscale":
                    results[path] = "busy"
                    jobs.put(path)
                    answer = "queued"
                elif self.path == "/result":
                    answer = results.get(path, "error: no such video was queued")
                    if answer != "busy":
                        results.pop(path, None)
                else:
                    answer = "error: unknown request " + self.path
            data = answer.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    with open(stamp, "w") as file:
        file.write(str(os.getpid()))
    loading = []

    def work():
        try:
            loading.append(loaded())
        except Exception as error:
            loading.append(error)
        while True:
            path = jobs.get()
            used[0] = time.time()
            try:
                if isinstance(loading[0], Exception):
                    raise loading[0]
                message = upscaled(loading[0], path)
            except Exception as error:
                message = "error: " + str(error).strip().splitlines()[-1]
            with lock:
                results[path] = message
            used[0] = time.time()

    def watch():
        written = os.path.getmtime(__file__)
        while alive(parent) and time.time() - used[0] < IdleSeconds and os.path.getmtime(__file__) == written:
            time.sleep(2)
        server.shutdown()

    threading.Thread(target=work, daemon=True).start()
    threading.Thread(target=watch, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        if os.path.exists(stamp):
            os.remove(stamp)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--serve":
        serve(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) == 2:
        try:
            print(upscaled(loaded(), sys.argv[1]))
        except Exception as error:
            print("RTX Video Super Resolution failed: " + str(error), file=sys.stderr)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(1)
