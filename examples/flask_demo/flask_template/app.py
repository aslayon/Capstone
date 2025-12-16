"""
CCTV 차량 추적 시스템 웹 인터페이스 (개선 버전)
Pipeline과 완전 통합 - 키 입력, 클릭, 로그 실시간 연동
"""
from core.pipeline import run_detect
from core.frame_bus import BUS
from datetime import datetime
from pathlib import Path
import time
import json
import re
import click
import cv2
import numpy as np
import threading
import os
import queue
from collections import deque
from typing import Dict, Any, Optional
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
    UserMixin,
)
from werkzeug.security import generate_password_hash, check_password_hash

GRAPH_CONNECTIONS_CACHE: Optional[Dict[str, Any]] = None


def load_graph_connections_cache() -> Dict[str, Any]:
    """config/cctv_graph_connections.json을 로딩/캐싱"""
    global GRAPH_CONNECTIONS_CACHE
    if GRAPH_CONNECTIONS_CACHE is not None:
        return GRAPH_CONNECTIONS_CACHE
    try:
        project_root = Path(__file__).resolve().parents[3]
    except IndexError:
        project_root = Path(__file__).resolve().parent
    graph_path = project_root / "config" / "cctv_graph_connections.json"
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[GRAPH] 연결 그래프 로드 실패: {e}")
        GRAPH_CONNECTIONS_CACHE = {}
        return GRAPH_CONNECTIONS_CACHE
    connections: Dict[str, Any] = {}
    for entry in data:
        name = entry.get("cctvname")
        if not name:
            continue
        connections[name] = entry.get("connections", [])
    GRAPH_CONNECTIONS_CACHE = connections
    return GRAPH_CONNECTIONS_CACHE

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key-change-me"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ============================================================
# 전역 상태 관리 - Pipeline과 웹을 연결하는 브릿지
# ============================================================
class PipelineState:
    """Pipeline 실행 상태를 웹과 공유"""
    def __init__(self):
        self.key_queue = queue.Queue()  # 웹→pipeline 키 입력
        self.click_queue = queue.Queue()  # 웹→pipeline 클릭 좌표
        self.fps = 0.0
        self.total_tracks = 0
        self.selected_id = None
        self.mode = 'single'  # 'single' or 'tri'
        self.current_camera = None
        self.lock = threading.Lock()
        self.camera_route: list[str] = []
        self.tri_active_cameras: list[str] = []
        self.max_route_length = 20
        self.graph_connections = load_graph_connections_cache()
        self.route_start_camera: Optional[str] = None
        
        # 통계 자동 업데이트를 위한 변수
        self.frame_count = 0
        self.last_time = time.time()

    def _append_camera_route(self, camera_name: Optional[str]):
        if not camera_name:
            return
        name = str(camera_name).strip()
        if not name:
            return
        if self.camera_route and self.camera_route[-1] == name:
            return
        self.camera_route.append(name)
        if len(self.camera_route) > self.max_route_length:
            self.camera_route = self.camera_route[-self.max_route_length:]

    def _update_tri_active(self, cameras: Optional[list]):
        cleaned = [str(cam).strip() for cam in (cameras or []) if cam]
        if not cleaned:
            self.tri_active_cameras = []
            return
        self.tri_active_cameras = cleaned

    def update_stats(self, **kwargs):
        """통계 업데이트"""
        with self.lock:
            tri_list = kwargs.pop('tri_cameras', None)
            if tri_list is not None:
                self._update_tri_active(tri_list)
            prev_selected = self.selected_id
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    if key == "current_camera":
                        self._append_camera_route(value)
            if 'selected_id' in kwargs:
                new_selected = kwargs.get('selected_id')
                if new_selected and new_selected != prev_selected:
                    self.route_start_camera = self.current_camera
                    self.camera_route = []
                    self._append_camera_route(self.current_camera)
                elif not new_selected:
                    self.route_start_camera = None
    
    def get_stats(self):
        """현재 통계 반환 (JSON 직렬화 안전)"""
        def _to_native(value):
            try:
                import numpy as np  # type: ignore
                if isinstance(value, np.generic):
                    return value.item()
            except Exception:
                pass
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    return value
            return value
        
        with self.lock:
            return {
                'fps': round(float(self.fps), 1),
                'total_tracks': _to_native(self.total_tracks),
                'selected_id': _to_native(self.selected_id),
                'mode': self.mode,
                'camera': self.current_camera
            }
    
    def put_key(self, key):
        """키 입력 큐에 추가"""
        try:
            self.key_queue.put_nowait(key)
            return True
        except queue.Full:
            return False
    
    def get_key(self, block=False, timeout=None):
        """키 입력 가져오기 (pipeline에서 호출)"""
        try:
            return self.key_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def put_click(self, x, y):
        """클릭 좌표 큐에 추가"""
        try:
            self.click_queue.put_nowait((x, y))
            return True
        except queue.Full:
            return False
    
    def get_click(self, block=False, timeout=None):
        """클릭 좌표 가져오기 (pipeline에서 호출)"""
        try:
            return self.click_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def calculate_fps(self):
        """FPS 자동 계산"""
        now = time.time()
        elapsed = now - self.last_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            with self.lock:
                self.fps = fps
            self.frame_count = 0
            self.last_time = now
        
        self.frame_count += 1
    
    def get_camera_route_summary(self) -> Dict[str, Any]:
        with self.lock:
            route_snapshot = list(self.camera_route)
            start = self.route_start_camera or (route_snapshot[0] if route_snapshot else None)
            current = route_snapshot[-1] if route_snapshot else None
            neighbors = self.graph_connections.get(current, []) if current else []
            tri_active = list(self.tri_active_cameras)
        return {
            'start_camera': start,
            'current_camera': current,
            'route': route_snapshot,
            'neighbors': neighbors,
            'tri_active_cameras': tri_active,
        }

# 전역 pipeline 상태
pipeline_state = PipelineState()

# ============================================================
# 로그 파일 모니터링
# ============================================================
class VehicleTrackingLogMonitor:
    """vehicle_track.txt 로그 파일 실시간 모니터링"""
    
    def __init__(self, log_root="tracking_logs"):
        self.log_root = Path(log_root)
        self.last_positions = {}  # 파일별 마지막 읽은 위치
        self.graph_connections = load_graph_connections_cache()
        
    def get_latest_logs(self, max_lines=100):
        """최신 로그 가져오기"""
        if not self.log_root.exists():
            return []
        
        # 오늘 날짜 디렉터리
        today = datetime.now().strftime('%Y-%m-%d')
        today_dir = self.log_root / today
        
        if not today_dir.exists():
            return []
        
        all_logs = []
        
        # 모든 세션 로그 파일 읽기
        for log_file in sorted(today_dir.glob("session_*_vehicle_track.txt")):
            try:
                file_key = str(log_file)
                last_pos = self.last_positions.get(file_key, 0)
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    # 마지막 위치부터 읽기
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    self.last_positions[file_key] = f.tell()
                    
                    # 로그 파싱
                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        log_entry = self._parse_log_line(line)
                        if log_entry:
                            all_logs.append(log_entry)
                            
            except Exception as e:
                print(f"[LOG_MONITOR] 로그 읽기 오류: {e}")
                continue
        
        # 시간순 정렬 (최신순)
        all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_logs[:max_lines]
    
    def _parse_log_line(self, line):
        """로그 라인 파싱"""
        try:
            # 형식: "2025-11-04 10:12:08.126 | [이벤트] 내용..."
            parts = line.split(' | ', 1)
            if len(parts) != 2:
                return None
            
            timestamp = parts[0].strip()
            message = parts[1].strip()
            
            # 이벤트 타입 추출
            event_type = "INFO"
            if "[추적 시작]" in message:
                event_type = "START"
            elif "[동일 차량 발견]" in message or "[MATCH" in message:
                event_type = "MATCH"
            elif "[카메라 전환]" in message:
                event_type = "SWITCH"
            
            # 카메라명 추출
            camera = "알 수 없음"
            if "카메라:" in message:
                try:
                    camera = message.split("카메라:")[1].split(",")[0].strip()
                except:
                    pass
            elif "→" in message:
                # "A → B" 형식
                try:
                    camera_part = message.split("→")[0].split("(")[0].strip()
                    camera = camera_part.split("]")[-1].strip()
                except:
                    pass
            
            # 차량 ID 추출
            vehicle_id = "-"
            id_match = re.search(r'ID\s*[:=]?\s*([A-Za-z0-9_-]+)', message)
            if id_match:
                vehicle_id = id_match.group(1)
            
            # 신뢰도 추출
            confidence = "-"
            if "신뢰도:" in message:
                try:
                    confidence = message.split("신뢰도:")[1].split()[0].strip()
                except:
                    pass
            elif "유사도:" in message:
                try:
                    similarity = message.split("유사도:")[1].split()[0].strip()
                    confidence = f"d={similarity}"
                except:
                    pass
            
            return {
                'timestamp': timestamp,
                'event_type': event_type,
                'camera': camera,
                'vehicle_id': vehicle_id,
                'confidence': confidence,
                'message': message,
                'raw': line
            }
            
        except Exception as e:
            print(f"[LOG_PARSE] 파싱 오류: {e}, line: {line[:100]}")
            return None
    
    def get_all_logs_from_file(self, max_lines=200):
        """파일에서 모든 로그 읽기 (전체 로그 페이지용)"""
        if not self.log_root.exists():
            return []
        
        today = datetime.now().strftime('%Y-%m-%d')
        today_dir = self.log_root / today
        
        if not today_dir.exists():
            return []
        
        all_logs = []
        
        for log_file in sorted(today_dir.glob("session_*_vehicle_track.txt"), reverse=True):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    log_entry = self._parse_log_line(line)
                    if log_entry:
                        all_logs.append(log_entry)
                        
            except Exception as e:
                print(f"[LOG_MONITOR] 전체 로그 읽기 오류: {e}")
                continue
        
        # 최신순 정렬
        all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_logs[:max_lines]

    def _extract_numeric_score(self, confidence_text):
        if not confidence_text:
            return None
        match = re.search(r"(\d+\.\d+)", confidence_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _is_reliable_camera_update(self, log_entry):
        event_type = (log_entry.get('event_type') or '').upper()
        confidence = (log_entry.get('confidence') or '').upper()
        if event_type in ('START', 'SWITCH'):
            return True
        if event_type == 'MATCH':
            if any(level in confidence for level in ('HIGH', 'MEDIUM')):
                return True
            score = self._extract_numeric_score(confidence)
            return score is not None and score >= 0.65
        return False

    def build_vehicle_states(self, logs):
        """�α׷κ��� �� ������ ����/���� ī�޶� ����"""
        if not logs:
            return []
        try:
            sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))
        except Exception:
            sorted_logs = logs
        states = {}
        for log in sorted_logs:
            vehicle_id = (log.get('vehicle_id') or '').strip()
            camera = (log.get('camera') or '').strip()
            if not vehicle_id or vehicle_id == '-' or not camera or camera == '�� �� ����':
                continue
            state = states.setdefault(vehicle_id, {
                'start_camera': None,
                'current_camera': None,
                'last_event': None,
                'last_timestamp': None,
            })
            if state['start_camera'] is None and log.get('event_type') == 'START':
                state['start_camera'] = camera
            if self._is_reliable_camera_update(log):
                if state['start_camera'] is None:
                    state['start_camera'] = camera
                state['current_camera'] = camera
                state['last_event'] = log.get('event_type')
                state['last_timestamp'] = log.get('timestamp')
        summaries = []
        for vid, info in states.items():
            current_camera = info.get('current_camera') or info.get('start_camera')
            neighbors = self.graph_connections.get(current_camera, []) if current_camera else []
            summaries.append({
                'vehicle_id': vid,
                'start_camera': info.get('start_camera'),
                'current_camera': current_camera,
                'last_event': info.get('last_event'),
                'last_timestamp': info.get('last_timestamp'),
                'neighbors': neighbors,
            })
        summaries.sort(key=lambda x: x.get('last_timestamp') or '', reverse=True)
        return summaries

    def get_vehicle_state_summary(self, max_lines=400):
        logs = self.get_all_logs_from_file(max_lines=max_lines)
        return self.build_vehicle_states(logs)
log_monitor = VehicleTrackingLogMonitor()

# ============================================================
# CCTV 데이터 로드
# ============================================================
def load_cctv_data():
    """JSON 파일에서 CCTV 데이터를 로드"""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        r"D:\MDO\CCTV\cctv_api_response.json",
    )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cctv_data = data.get("response", {}).get("data", [])

        # 슬롯에 CCTV 매핑
        hardcoded_cameras = {}

        # 서울 구역
        seoul_slots = [
            "SEOUL-A1", "SEOUL-A2", "SEOUL-B1", "SEOUL-B2",
            "SEOUL-C1", "SEOUL-C2", "SEOUL-D1", "SEOUL-D2",
        ]
        # 경기 구역
        gyeonggi_slots = [
            "GYEONGGI-A1", "GYEONGGI-A2", "GYEONGGI-B1", "GYEONGGI-B2",
            "GYEONGGI-C1", "GYEONGGI-C2", "GYEONGGI-D1", "GYEONGGI-D2",
        ]

        # CCTV 데이터를 슬롯에 할당
        for i, slot in enumerate(seoul_slots):
            if i < len(cctv_data):
                cctv = cctv_data[i]
                hardcoded_cameras[slot] = {
                    "name": cctv.get("cctvname", f"CCTV {i+1}"),
                    "url": cctv.get("cctvurl", "0"),
                }
            else:
                hardcoded_cameras[slot] = {"name": "미사용", "url": "0"}

        for i, slot in enumerate(gyeonggi_slots):
            if i < len(cctv_data):
                cctv = cctv_data[i]
                hardcoded_cameras[slot] = {
                    "name": cctv.get("cctvname", f"CCTV {i+1}"),
                    "url": cctv.get("cctvurl", "0"),
                }
            else:
                hardcoded_cameras[slot] = {"name": "미사용", "url": "0"}

        # 웹캠 테스트용 슬롯 추가
        hardcoded_cameras["SEOUL-C2"] = {"name": "웹캠 테스트", "url": "0"}
        hardcoded_cameras["GYEONGGI-C2"] = {"name": "웹캠 테스트", "url": "0"}

        return hardcoded_cameras

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"JSON 파일 로드 실패: {e}")
        # 기본값 반환 - 모든 슬롯 포함!
        return {
            "SEOUL-A1": {"name": "[세종포천선] [포천]고삼터널(포천)", "url": "0"},
            "SEOUL-A2": {"name": "[경인선] 서운분기점2", "url": "0"},
            "SEOUL-B1": {"name": "[경인선] 부천", "url": "0"},
            "SEOUL-B2": {"name": "[인천국제공항선] 공항 30.1K", "url": "0"},
            "SEOUL-C1": {"name": "[인천국제공항선] 김포나들목", "url": "0"},
            "SEOUL-C2": {"name": "웹캠 테스트", "url": "0"},
            "SEOUL-D1": {"name": "연결 대기", "url": "0"},
            "SEOUL-D2": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-A1": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-A2": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-B1": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-B2": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-C1": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-C2": {"name": "웹캠 테스트", "url": "0"},
            "GYEONGGI-D1": {"name": "연결 대기", "url": "0"},
            "GYEONGGI-D2": {"name": "연결 대기", "url": "0"},
        }

# ============================================================
# 데이터베이스 모델
# ============================================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    logs = db.relationship("DetectionLog", backref="user", lazy=True)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    stream_url = db.Column(db.String(255), nullable=False)
    slot = db.Column(db.String(32), nullable=True, index=True)
    logs = db.relationship("DetectionLog", backref="camera", lazy=True)


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id"), nullable=False)
    label = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    ts = db.Column(db.DateTime, default=datetime.utcnow, index=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ============================================================
# 인증 라우트
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("로그인에 성공했습니다.", "success")
            next_url = request.args.get("next") or url_for("cctv_select")
            return redirect(next_url)
        flash("아이디 또는 비밀번호가 올바르지 않습니다.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("모든 필드를 입력해 주세요.", "warning")
            return render_template("signup.html")

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("이미 사용 중인 아이디 또는 이메일입니다.", "danger")
            return render_template("signup.html")

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("회원가입이 완료되었습니다. 로그인해 주세요.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("로그아웃되었습니다.", "info")
    return redirect(url_for("login"))

# ============================================================
# CCTV 관련 라우트
# ============================================================
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("cctv_select"))
    return redirect(url_for("login"))


@app.route("/cctv")
@login_required
def cctv_select():
    hardcoded_cameras = load_cctv_data()
    return render_template("cctv_select.html", hardcoded_cameras=hardcoded_cameras)


@app.route("/cctv/<slot>")
@login_required
def cctv_page(slot: str):
    hardcoded_cameras = load_cctv_data()

    if slot not in hardcoded_cameras:
        flash("존재하지 않는 카메라입니다.", "warning")
        return redirect(url_for("cctv_select"))

    camera_info = hardcoded_cameras[slot]
    camera_info["slot"] = slot
    return render_template("stream.html", camera=camera_info)


@app.route("/video_feed")
@login_required
def video_feed():
    """
    실시간 비디오 스트리밍 (MJPEG)
    BUS에서 프레임을 가져와 웹으로 스트리밍
    """
    def generate():
        frame_count = 0
        no_frame_count = 0
        
        while True:
            try:
                # BUS에서 최신 프레임 가져오기
                frame = BUS.latest()
                
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count % 30 == 1:
                        print(f"[VIDEO_FEED] ⚠️  프레임 없음 ({no_frame_count}회)")
                    time.sleep(0.033)
                    continue
                
                # 프레임 복구 로그
                if no_frame_count > 0:
                    if no_frame_count > 10:
                        print(f"[VIDEO_FEED] ✅ 복구! (누락: {no_frame_count})")
                    no_frame_count = 0
                
                # FPS 계산
                pipeline_state.calculate_fps()
                
                # JPEG 인코딩
                ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                
                frame_count += 1
                
                # MJPEG 형식으로 전송
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpg.tobytes() + 
                       b'\r\n')
                
            except GeneratorExit:
                print(f"[VIDEO_FEED] 🔌 연결 종료")
                break
            except Exception as e:
                print(f"[VIDEO_FEED] ❌ 에러: {e}")
                time.sleep(0.1)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ============================================================
# 로그 페이지
# ============================================================
@app.route("/logs")
@login_required
def logs():
    """차량 추적 로그 페이지"""
    # 로그 파일에서 로그 가져오기
    vehicle_logs = log_monitor.get_all_logs_from_file(max_lines=200)
    vehicle_states = log_monitor.build_vehicle_states(vehicle_logs)
    camera_route = pipeline_state.get_camera_route_summary()
    
    return render_template(
        "logs.html",
        logs=vehicle_logs,
        vehicle_states=vehicle_states,
        camera_route=camera_route,
    )

# ============================================================
# API 엔드포인트 - 웹 ↔ Pipeline 통신
# ============================================================
@app.route('/api/stats')
def api_stats():
    """현재 시스템 통계 API"""
    return jsonify(pipeline_state.get_stats())


@app.route('/api/logs')
def api_logs():
    """실시간 로그 API"""
    logs = log_monitor.get_latest_logs(max_lines=50)
    vehicle_states = log_monitor.get_vehicle_state_summary()
    camera_route = pipeline_state.get_camera_route_summary()
    return jsonify({
        'logs': logs,
        'vehicle_states': vehicle_states,
        'camera_route': camera_route,
    })


@app.route('/api/send_key', methods=['POST'])
def api_send_key():
    """키 입력 전송 API"""
    data = request.get_json()
    key = data.get('key', '')
    
    if not key:
        return jsonify({'success': False, 'message': '잘못된 키'}), 400
    
    # pipeline에 키 전송
    success = pipeline_state.put_key(key)
    
    if success:
        # 키에 따른 메시지
        messages = {
            'p': '🔄 Tri 모드 전환',
            'a': '⬅️ 좌측 카메라로 이동',
            's': '⬇️ 아래 카메라로 이동',
            'd': '➡️ 우측 카메라로 이동',
            'w': '⬆️ 위 카메라로 이동'
        }
        
        return jsonify({
            'success': True,
            'message': messages.get(key, f'키 전송: {key}')
        })
    else:
        return jsonify({'success': False, 'message': '키 전송 실패 (큐 가득참)'}), 500


@app.route('/api/click', methods=['POST'])
def api_click():
    """마우스 클릭 좌표 전송 API"""
    data = request.get_json()
    x = data.get('x', 0)
    y = data.get('y', 0)
    
    # pipeline에 클릭 좌표 전송
    success = pipeline_state.put_click(x, y)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'클릭: ({x}, {y})',
            'selected_id': pipeline_state.selected_id
        })
    else:
        return jsonify({'success': False, 'message': '클릭 전송 실패'}), 500


@app.route('/api/clear_selection', methods=['POST'])
def api_clear_selection():
    """차량 선택 해제 API"""
    pipeline_state.selected_id = None
    pipeline_state.put_key('clear')
    
    return jsonify({
        'success': True,
        'message': '✅ 차량 선택 해제됨'
    })

# ============================================================
# CLI 명령어
# ============================================================
@app.cli.command("init-db")
def init_db_command():
    """데이터베이스 테이블 생성 및 기본 카메라 시드"""
    db.create_all()
    ensure_slot_column()
    if Camera.query.count() == 0:
        db.session.add(Camera(name="웹캠(로컬)", stream_url="0"))
        db.session.commit()
        print("DB 초기화 및 카메라 시드 완료")
    else:
        print("DB가 이미 초기화되어 있습니다.")


def ensure_slot_column():
    """DB에 slot 컬럼이 없으면 추가"""
    try:
        from sqlalchemy import inspect
        engine = db.get_engine()
        inspector = inspect(engine)
        cols = [c["name"] for c in inspector.get_columns("camera")]
        if "slot" not in cols:
            with engine.connect() as conn:
                conn.execute("ALTER TABLE camera ADD COLUMN slot VARCHAR(32)")
                conn.commit()
                print("slot 컬럼이 추가되었습니다.")
    except Exception as e:
        print(f"slot 컬럼 추가 실패: {e}")

# ============================================================
# Pipeline 통합
# ============================================================
def start_pipeline_background():
    """파이프라인을 별도 스레드로 실행"""
    print("[INIT] 🚀 Starting pipeline thread...")
    t = threading.Thread(target=run_detect, daemon=True)
    t.start()
    print("[INIT] ✅ Pipeline thread started")


def run():
    """애플리케이션 실행"""
    # DB 초기화
    with app.app_context():
        db.create_all()
        ensure_slot_column()
        if Camera.query.count() == 0:
            db.session.add(Camera(name="웹캠(로컬)", stream_url="0"))
            db.session.commit()
    
    # Pipeline 시작
    start_pipeline_background()
    
    # Flask 앱 실행
    print("[INIT] 🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    run()
