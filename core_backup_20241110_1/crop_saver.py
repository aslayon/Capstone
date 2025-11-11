# 상단 import 보강
import os, cv2, time, datetime
from pathlib import Path
import re

def _sanitize_name(name: str) -> str:
    # 윈도우 금지문자 및 경로 구분자 제거/치환
    return re.sub(r'[\\/:*?"<>|]+', '_', name).strip()

def _project_root() -> Path:
    # 이 파일 기준으로 프로젝트 루트 계산: <repo>/core/crop_saver.py → parents[1] = <repo>
    return Path(__file__).resolve().parents[1]

class CropSaver:
    def __init__(self, save_root: str = "reid_crops", save_every: int = 3,
                 pad: int = 2, print_interval_sec: float = 1.0, on_saved=None, verbose=True):
        self.pad = int(pad)
        self.save_every = max(1, int(save_every))
        self.print_interval = float(print_interval_sec)
        self.on_saved = on_saved
        self.verbose = verbose

        # === 절대 경로 강제 ===
        root = Path(save_root)
        if not root.is_absolute():
            root = _project_root() / root
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

        self._counts = {"L": 0, "C": 0, "R": 0}
        self._last_print_t = 0.0
        self._cam_name = None

        if self.verbose:
            print(f"[CropSaver] CWD={os.getcwd()}  ROOT={self.root}")

    def new_camera(self, cam_name: str, reset_counts: bool = True):
        self._cam_name = cam_name
        if reset_counts:
            self._counts = {"L": 0, "C": 0, "R": 0}

    def _safe_crop(self, img, box):
        if img is None:
            if self.verbose: print("[CropSaver] WARN: seg_frame is None")
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - self.pad); y1 = max(0, y1 - self.pad)
        x2 = min(w - 1, x2 + self.pad); y2 = min(h - 1, y2 + self.pad)
        if x2 <= x1 or y2 <= y1:
            if self.verbose: print(f"[CropSaver] WARN: invalid box after pad: {(x1,y1,x2,y2)} in {w}x{h}")
            return None
        return img[y1:y2, x1:x2]

    def _save_seg_crop(self, seg: str, seg_frame, box, cam_name: str, shown_id: int) -> bool:
        crop = self._safe_crop(seg_frame, box)
        if crop is None:
            return False

        # === 카메라명 sanitize ===
        cam_dir = _sanitize_name(cam_name)
        day = datetime.datetime.now().strftime("%Y%m%d")
        out_dir = self.root / cam_dir / f"seg_{seg}" / day
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%H%M%S_%f")
        x1, y1, x2, y2 = map(int, box)
        fname = out_dir / f"{ts}_id{shown_id}_{x1}_{y1}_{x2}_{y2}.jpg"

        # === imencode + open('wb')로 저장 (유니코드 경로 안전) ===
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            if self.verbose: print("[CropSaver] ERROR: cv2.imencode failed")
            return False
        try:
            with open(fname, "wb") as f:
                f.write(buf.tobytes())
            if self.on_saved: 
                try: self.on_saved(str(fname), seg, cam_name, shown_id, box)
                except Exception as e:
                    if self.verbose: print(f"[CropSaver] on_saved error: {e}")
            return True
        except Exception as e:
            if self.verbose: print(f"[CropSaver] ERROR writing {fname}: {e}")
            return False

    def process_tri(self, frame_idx, selected_id, seg_frames, seg_tracks, seg_remap, cam_name,
                    now_ts=None, one_per_segment=True, debug_save_first_if_no_match=False):
        if selected_id is None and not debug_save_first_if_no_match:
            self._maybe_print(now_ts); return

        if frame_idx % self.save_every != 0:
            self._maybe_print(now_ts); return

        if self._cam_name != cam_name:
            self.new_camera(cam_name, reset_counts=False)

        # 저장 로직
        matched_any = False
        for seg in ("L", "C", "R"):
            frame = seg_frames.get(seg)
            tracks = seg_tracks.get(seg, [])
            remap = seg_remap.get(seg, {})

            saved_here = 0
            # 1) 먼저 selected_id 일치 저장 시도
            if selected_id is not None:
                for tid, x1, y1, x2, y2 in tracks:
                    shown_id = remap.get(tid, tid)
                    if shown_id == selected_id:
                        ok = self._save_seg_crop(seg, frame, (x1,y1,x2,y2), cam_name, shown_id)
                        if ok:
                            self._counts[seg] += 1; saved_here += 1; matched_any = True
                            if one_per_segment: break

            # 2) 매칭이 전혀 안 될 때 디버그 모드로 첫 트랙 저장(폴더/경로 테스트 용)
            if not matched_any and debug_save_first_if_no_match and len(tracks) > 0 and saved_here == 0:
                tid, x1, y1, x2, y2 = tracks[0]
                ok = self._save_seg_crop(seg, frame, (x1,y1,x2,y2), cam_name, remap.get(tid, tid))
                if ok: self._counts[seg] += 1; saved_here += 1

        self._maybe_print(now_ts)

    def draw_hud(self, tri_frame):
        total = self._counts["L"] + self._counts["C"] + self._counts["R"]
        if total <= 0 or tri_frame is None: return
        text = f"SAVED L{self._counts['L']} C{self._counts['C']} R{self._counts['R']} | TOTAL {total}"
        cv2.putText(tri_frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def _maybe_print(self, now_ts):
        now_ts = time.time() if now_ts is None else now_ts
        if now_ts - self._last_print_t >= self.print_interval:
            self._last_print_t = now_ts
            total = self._counts["L"] + self._counts["C"] + self._counts["R"]
            if total > 0 and self._cam_name:
                print(f"[CROP] {self._cam_name} 누적: L={self._counts['L']} C={self._counts['C']} R={self._counts['R']} (총 {total})")
