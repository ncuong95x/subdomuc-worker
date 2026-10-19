#!/usr/bin/env python3
"""
YouTube Game Cloud Worker (Máy ảo 2 - Dự án Phân tích Game)
Runs on GitHub Actions runner (Ubuntu) to perform:
1. Video download (if URL provided)
2. VieNeu TTS synthesis with old voice 'Hải Đăng' (v3turbo ONNX)
3. 1.3x speed voice alignment & master audio mixing
4. Single-pass FFmpeg compound render:
   - Dynamic 16:9 Landscape or 9:16 Shorts detection
   - Optional watermark blur
   - Audio mix (5% original game sound/BGM + 100% Vietnamese dub)
   - Compound slowdown 0.8x (Net: Video 0.8x, Voice 1.04x)
   - Faststart MP4 optimized for YouTube
5. Synchronized SRT subtitle generation
Stealth Mode: Che bớt tiến trình, không in kịch bản/nội dung thoại ra màn hình console.
"""

import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import soundfile as sf

try:
    from vieneu import Vieneu
except ImportError:
    Vieneu = None


def get_video_info(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    stream = data.get("streams", [{}])[0]
    format_info = data.get("format", {})

    w = int(stream.get("width", 1920))
    h = int(stream.get("height", 1080))
    dur_str = stream.get("duration") or format_info.get("duration", "0")
    duration = float(dur_str)

    return {
        "width": w,
        "height": h,
        "is_shorts": h > w,
        "duration": duration
    }


def download_video_yt_dlp(url: str, output_path: Path):
    print("[Download] Đang tải video từ nguồn trực tuyến...")
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"[Download] Tải video thành công ({output_path.stat().st_size / 1024 / 1024:.1f} MB).")


def main():
    parser = argparse.ArgumentParser(description="YouTube Game Cloud Worker (Máy ảo 2)")
    parser.add_argument("--job-id", type=str, required=True, help="Unique job ID")
    parser.add_argument("--video-url", type=str, default="", help="Video download URL")
    parser.add_argument("--speed-voice", type=float, default=1.3, help="Voice speed factor (default 1.3)")
    parser.add_argument("--slowdown", type=float, default=0.8, help="Final slowdown factor (default 0.8)")
    parser.add_argument("--orig-vol", type=float, default=0.05, help="Original audio volume (default 0.05)")
    parser.add_argument("--blur-box", type=str, default="", help="Optional watermark blur box x,y,w,h")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 65)
    print("🚀 [MÁY ẢO 2] CLOUD RENDER - DỰ ÁN PHÂN TÍCH GAME (YOUTUBE)")
    print(f"👉 Giọng đọc: 'Hải Đăng' | Thoại {args.speed_voice}x | Chậm {args.slowdown}x | Âm nền {int(args.orig_vol*100)}%")
    print("=" * 65)

    work_dir = Path("workspace").resolve()
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    temp_dir = work_dir / "temp"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate or download video
    video_path = None
    mp4_candidates = list(input_dir.glob("*.mp4"))
    if mp4_candidates:
        video_path = mp4_candidates[0]
        print("[Video] Đã nhận video từ gói dữ liệu.")
    elif args.video_url:
        video_path = input_dir / "source_video.mp4"
        download_video_yt_dlp(args.video_url, video_path)
    else:
        any_vids = list(work_dir.rglob("*.mp4"))
        if any_vids:
            video_path = any_vids[0]
            print("[Video] Đã tìm thấy video nguồn.")
        else:
            raise FileNotFoundError("Không tìm thấy file video và không có URL được cung cấp!")

    vinfo = get_video_info(video_path)
    total_video_dur = vinfo["duration"]
    is_shorts = vinfo["is_shorts"]
    print(f"[Video] Định dạng: {'9:16 Shorts' if is_shorts else '16:9 Chuẩn'} | Thời lượng: {total_video_dur:.1f}s ({total_video_dur/60:.2f} phút).")

    # 2. Locate segments_vi.json
    seg_candidates = list(input_dir.glob("*segments*.json")) + list(work_dir.rglob("*segments_vi.json"))
    if not seg_candidates:
        raise FileNotFoundError("Không tìm thấy dữ liệu thoại!")
    segments_path = seg_candidates[0]
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    print(f"[Segments] Tổng số câu thoại: {len(segments)} câu.")

    # 3. VieNeu TTS Synthesis (Hải Đăng)
    audio_clips_dir = temp_dir / "audio_clips_hai_dang"
    audio_clips_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 48000

    print("\n[TTS] Bắt đầu tổng hợp giọng đọc AI 'Hải Đăng'...")
    t_tts_start = time.time()
    tts = Vieneu(mode="v3turbo", backend="onnx", device="cpu")

    missing = []
    for s in segments:
        clip = audio_clips_dir / f"clip_{s['id']:04d}.wav"
        if not (clip.exists() and clip.stat().st_size > 2000):
            missing.append(s)
        else:
            s["audio_duration"] = sf.info(str(clip)).duration

    if missing:
        batch_size = 16
        for b_start in range(0, len(missing), batch_size):
            batch = missing[b_start : b_start + batch_size]
            texts = [s["vi"] for s in batch]
            audios = tts.infer_batch(
                texts,
                voice="Hải Đăng",
                batch_size=len(texts),
                max_new_frames=600,
                max_chars=350,
                denoise=True,
                apply_watermark=False
            )
            for s, audio_arr in zip(batch, audios):
                clip = audio_clips_dir / f"clip_{s['id']:04d}.wav"
                arr = np.asarray(audio_arr, dtype=np.float32)
                sf.write(str(clip), arr, sample_rate, subtype="PCM_16")
                s["audio_duration"] = len(arr) / sample_rate

            cur_done = min(b_start + batch_size, len(missing))
            pct = int((cur_done / len(missing)) * 100)
            print(f"[TTS] Tiến độ: {cur_done}/{len(missing)} câu ({pct}%)...", flush=True)

    print(f"[TTS] ✅ Hoàn tất giọng đọc trong {time.time() - t_tts_start:.1f}s.")

    # 4. Schedule timeline & atempo stretch
    sped_dir = temp_dir / "sped_clips"
    sped_dir.mkdir(parents=True, exist_ok=True)

    placements = []
    for s in segments:
        start_time = float(s["start"])
        orig_dur = float(s.get("audio_duration", 1.0))
        tempo = args.speed_voice
        actual_dur = orig_dur / tempo
        actual_end = start_time + actual_dur

        placements.append({
            "id": s["id"],
            "start_1_0": start_time,
            "dur_1_0": actual_dur,
            "end_1_0": actual_end,
            "tempo": tempo,
            "vi": s["vi"],
            "clip_file": audio_clips_dir / f"clip_{s['id']:04d}.wav"
        })

    def process_clip(item):
        in_file = item["clip_file"]
        tempo = item["tempo"]
        out_file = sped_dir / f"sped_{item['id']:04d}.wav"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(in_file),
            "-filter:a", f"atempo={tempo:.4f}",
            "-ar", str(sample_rate),
            str(out_file)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return item["id"]

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(process_clip, item) for item in placements]
        for f in as_completed(futures):
            f.result()

    total_samples = int(np.ceil(total_video_dur * sample_rate)) + sample_rate
    master_audio = np.zeros(total_samples, dtype=np.float32)

    for item in placements:
        sped_file = sped_dir / f"sped_{item['id']:04d}.wav"
        if not sped_file.exists():
            continue
        data, _ = sf.read(str(sped_file))
        if len(data.shape) > 1:
            data = data[:, 0]

        start_idx = int(round(item["start_1_0"] * sample_rate))
        end_idx = min(total_samples, start_idx + len(data))
        clip_len = end_idx - start_idx
        if clip_len > 0:
            master_audio[start_idx:end_idx] += data[:clip_len].astype(np.float32)

    final_sample_count = int(round(total_video_dur * sample_rate))
    master_audio = master_audio[:final_sample_count]

    peak = float(np.max(np.abs(master_audio))) if len(master_audio) > 0 else 0.0
    if peak > 0.98:
        master_audio = master_audio * (0.95 / peak)

    master_wav = temp_dir / "master_voice_1_0.wav"
    sf.write(str(master_wav), master_audio, sample_rate, subtype="PCM_16")

    # 5. Export synchronized SRT subtitle
    def fmt_time(sec: float) -> str:
        ms_tot = max(0, round(sec * 1000))
        h, rem = divmod(ms_tot, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    final_srt = output_dir / "video_dubbed_game.srt"
    srt_rows = []
    for idx, item in enumerate(placements, start=1):
        f_s = item["start_1_0"] / args.slowdown
        f_e = item["end_1_0"] / args.slowdown
        srt_rows.append(str(idx))
        srt_rows.append(f"{fmt_time(f_s)} --> {fmt_time(f_e)}")
        srt_rows.append(item["vi"])
        srt_rows.append("")
    final_srt.write_text("\n".join(srt_rows), encoding="utf-8")

    # 6. Single-Pass FFmpeg Render (YouTube High Quality)
    print("\n[Render] Đang kết xuất video thành phẩm (FFmpeg YouTube)...")

    # Video filter chain
    if args.blur_box and args.blur_box.strip() and args.blur_box.strip().lower() != "none":
        try:
            bx, by, bw, bh = [int(x.strip()) for x in args.blur_box.split(",")]
            blur_r = min(10, max(2, int(bh // 4)))
            vfilter = (
                f"[0:v]split=2[base][wm];"
                f"[wm]crop={bw}:{bh}:{bx}:{by},boxblur={blur_r}:2[wmblur];"
                f"[base][wmblur]overlay={bx}:{by}[v_blurred];"
                f"[v_blurred]setpts=PTS/{args.slowdown},fps=30[vout];"
            )
        except Exception:
            vfilter = f"[0:v]setpts=PTS/{args.slowdown},fps=30[vout];"
    else:
        vfilter = f"[0:v]setpts=PTS/{args.slowdown},fps=30[vout];"

    afilter = (
        f"[0:a]aresample=48000,volume={args.orig_vol:.3f}[a_orig];"
        f"[1:a]aresample=48000,volume=1.0[a_dub];"
        f"[a_orig][a_dub]amix=inputs=2:duration=first:dropout_transition=0[a_mix];"
        f"[a_mix]atempo={args.slowdown}[aout]"
    )

    filter_complex = vfilter + afilter
    final_mp4 = output_dir / "video_dubbed_game.mp4"
    t_render_start = time.time()
    cmd_render = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "0", "-t", f"{total_video_dur}",
        "-i", str(video_path),
        "-i", str(master_wav),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", "3500k",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-movflags", "+faststart",
        str(final_mp4)
    ]
    subprocess.run(cmd_render, check=True)
    render_time = time.time() - t_render_start
    print(f"[Render] ✅ Render hoàn thành trong {render_time:.1f}s ({final_mp4.stat().st_size / 1024 / 1024:.1f} MB).")

    # 7. Summary
    summary = {
        "job_id": args.job_id,
        "video_duration_orig": total_video_dur,
        "video_duration_final": total_video_dur / args.slowdown,
        "segments_count": len(segments),
        "total_time_seconds": time.time() - t_start,
        "video_file": final_mp4.name,
        "srt_file": final_srt.name
    }
    (output_dir / "job_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"🎉 Hoàn thành Máy ảo 2 (Game) trong {(time.time() - t_start)/60:.2f} phút!")


if __name__ == "__main__":
    main()
