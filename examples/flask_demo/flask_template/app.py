"""
JSON파일에서 URL을 읽어와서 설정 -> 현재 딜레이가 발생하는 문제
버퍼? 뭐 쓰면 해결된다고 했던거 같은데 확인점

3344


876 1163
app.py 실행 후 실행안되면
터미널에서
cd ~~~/DongOh/CCTV/CCTV_web/flask-template로 옮기고
Remove-Item instance\app.db -ErrorAction SilentlyContinue
입력후
python app.py실행 ㄱㄱ
"""
from core.pipeline import run_detect
from core.frame_bus import BUS
from datetime import datetime
import time
import json
import click
import cv2
import numpy as np
import threading
import os
import pathlib
from typing import Dict, Any, Optional
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    stream_with_context,
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

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key-change-me"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# CCTV 정보를 JSON 파일에서 로드하는 함수
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
            "SEOUL-A1",
            "SEOUL-A2",
            "SEOUL-B1",
            "SEOUL-B2",
            "SEOUL-C1",
            "SEOUL-C2",
            "SEOUL-D1",
            "SEOUL-D2",
        ]
        # 경기 구역
        gyeonggi_slots = [
            "GYEONGGI-A1",
            "GYEONGGI-A2",
            "GYEONGGI-B1",
            "GYEONGGI-B2",
            "GYEONGGI-C1",
            "GYEONGGI-C2",
            "GYEONGGI-D1",
            "GYEONGGI-D2",
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
        # 기본값 반환
        return {
            "SEOUL-A1": {"name": "[경인선] 도당육교"},
            "SEOUL-A2": {"name": "[경인선] 서운분기점2"},
            "SEOUL-B1": {"name": "[경인선] 부천"},
            "SEOUL-B2": {"name": "[인천국제공항선] 공항 30.1K"},
            "SEOUL-C1": {"name": "[인천국제공항선] 김포나들목"},
            "SEOUL-C2": {"name": "웹캠 테스트"},
            "SEOUL-D1": {"name": "연결 실패"},
            "SEOUL-D2": {"name": "연결 실패"},
            "GYEONGGI-A1": {"name": "연결 실패"},
            "GYEONGGI-A2": {"name": "연결 실패"},
            "GYEONGGI-B1": {"name": "연결 실패"},
            "GYEONGGI-B2": {"name": "연결 실패"},
            "GYEONGGI-C1": {"name": "연결 실패"},
            "GYEONGGI-C2": {"name": "웹캠 테스트"},
            "GYEONGGI-D1": {"name": "연결 실패"},
            "GYEONGGI-D2": {"name": "연결 실패"},
        }


# ---------- 모델 ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(
        db.String(64), unique=True, nullable=False, index=True
    )  # 로그인 ID
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    logs = db.relationship("DetectionLog", backref="user", lazy=True)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)  # 예: 정문, 로비 등
    stream_url = db.Column(
        db.String(255), nullable=False
    )  # 0(웹캠), 파일경로, rtsp/http 등
    slot = db.Column(
        db.String(32), nullable=True, index=True
    )  # 슬롯 위치 (예: SEOUL-A1)
    logs = db.relationship("DetectionLog", backref="camera", lazy=True)


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id"), nullable=False)
    label = db.Column(db.String(64), nullable=False)  # 예: person
    confidence = db.Column(db.Float, nullable=False)  # 0.0 ~ 1.0
    ts = db.Column(db.DateTime, default=datetime.utcnow, index=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------- 라우트: 인증 ----------
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

        if User.query.filter(
            (User.username == username) | (User.email == email)
        ).first():
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


# ---------- 라우트: CCTV 선택/스트림 ----------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("cctv_select"))
    return redirect(url_for("login"))


@app.route("/cctv")
@login_required
def cctv_select():
    # JSON 파일에서 CCTV 정보 동적 로드
    hardcoded_cameras = load_cctv_data()
    return render_template("cctv_select.html", hardcoded_cameras=hardcoded_cameras)


@app.route("/cctv/<slot>")
@login_required
def cctv_page(slot: str):
    # JSON 파일에서 CCTV 정보 동적 로드
    hardcoded_cameras = load_cctv_data()

    if slot not in hardcoded_cameras:
        flash("존재하지 않는 카메라입니다.", "warning")
        return redirect(url_for("cctv_select"))

    camera_info = hardcoded_cameras[slot]
    camera_info["slot"] = slot
    return render_template("stream.html", camera=camera_info)




@app.route("/cctv/<slot>/video_feed")
@login_required
def video_feed(slot: str):
    """
    실시간 비디오 스트리밍 (MJPEG)
    ✅ BUS에서 프레임만 가져옴 (cv2.VideoCapture 사용 안 함)
    """
    from core.frame_bus import BUS
    import cv2
    import time
    
    print(f"[VIDEO_FEED] 🎥 스트림 시작: slot={slot}")
    
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
                    print(f"[VIDEO_FEED] ✅ 복구! (누락: {no_frame_count})")
                    no_frame_count = 0
                
                # JPEG 인코딩
                ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[VIDEO_FEED] 📊 {frame_count}프레임 전송 완료")
                
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


# ---------- 라우트: 로그 조회 ----------
@app.route("/logs")
@login_required
def logs():
    camera_id = request.args.get("camera_id", type=int)
    q = DetectionLog.query.filter_by(user_id=current_user.id)
    if camera_id:
        q = q.filter_by(camera_id=camera_id)
    q = q.order_by(DetectionLog.ts.desc()).limit(200)
    cams = Camera.query.order_by(Camera.id.asc()).all()
    return render_template(
        "logs.html", logs=q.all(), cameras=cams, selected_camera_id=camera_id
    )


# ---------- CLI: DB 초기화/시드 ----------
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


@app.cli.command("import-cctv-json")
@click.argument("path")
def import_cctv_json(path: str):
    """외부 API로 받은 CCTV JSON을 DB에 등록합니다."""
    with app.app_context():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"파일을 열 수 없습니다: {e}")
            return

        items = []
        if isinstance(payload, dict):
            items = payload.get("response", {}).get("data", [])
        if not isinstance(items, list):
            print("JSON에서 항목을 찾을 수 없습니다.")
            return

        created = 0
        skipped = 0
        for it in items:
            name = it.get("cctvname") or it.get("name") or "API-CAM"
            url = it.get("cctvurl") or it.get("stream_url") or it.get("url")
            if not url:
                skipped += 1
                continue
            if Camera.query.filter_by(stream_url=url).first():
                skipped += 1
                continue
            cam = Camera(name=name, stream_url=url)
            db.session.add(cam)
            created += 1

        db.session.commit()
        print(f"완료: 생성 {created}, 스킵 {skipped}")


# --------- API: 카메라 등록 ---------
@app.route("/api/register_camera", methods=["POST"])
@login_required
def api_register_camera():
    payload = request.get_json() or {}
    name = payload.get("name") or "API-CAM"
    url = payload.get("stream_url") or payload.get("url")
    slot = payload.get("slot")

    if not url:
        return jsonify({"error": "missing url"}), 400

    # 중복 URL 검사
    if Camera.query.filter_by(stream_url=url).first():
        return jsonify({"status": "exists"})

    # 슬롯이 이미 사용중인지 검사
    if slot and Camera.query.filter_by(slot=slot).first():
        return jsonify({"error": "slot already occupied"}), 400

    cam = Camera(name=name, stream_url=url, slot=slot)
    db.session.add(cam)
    db.session.commit()
    return jsonify({"status": "created", "id": cam.id})


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
        
def start_pipeline_background():
    """파이프라인을 별도 스레드로 실행"""
    print("[INIT] Starting pipeline thread...")
    t = threading.Thread(target=run_detect, daemon=True)
    t.start()

def run():
    # 개발 편의를 위해 직접 실행도 가능
    with app.app_context():
        db.create_all()
        ensure_slot_column()  # slot 컬럼 확인/추가
        if Camera.query.count() == 0:
            db.session.add(Camera(name="웹캠(로컬)", stream_url="0"))
            db.session.commit()
    start_pipeline_background()
    app.run(debug=False, threaded=True)



import queue
command_queue = queue.Queue()  # 웹 → pipeline 명령 전달

tracking_stats = {
    "mode": "single",
    "selected_id": None,
    "total_tracks": 0,
    "fps": 0.0,
    "last_update": time.time(),
}


# ===== API: 키 입력 전송 =====
@app.route("/api/send_key", methods=["POST"])
@login_required
def api_send_key():
    """
    웹에서 키 입력을 pipeline으로 전달
    
    Body: {"key": "p"}
    """
    data = request.get_json() or {}
    key = data.get("key", "").lower()
    
    if not key:
        return jsonify({"status": "error", "message": "키 없음"}), 400
    
    # 명령 큐에 추가
    command_queue.put({"type": "key", "key": key})
    
    print(f"[API] 키 입력: {key}")
    
    messages = {
        "p": "Tri 모드 토글",
        "a": "왼쪽 카메라로 이동",
        "d": "오른쪽 카메라로 이동",
        "w": "위쪽 카메라로 이동",
        "s": "아래쪽 카메라로 이동",
    }
    
    return jsonify({
        "status": "ok",
        "message": messages.get(key, f"키 '{key}' 전송 완료")
    })


# ===== API: 클릭 좌표 전송 =====
@app.route("/api/click", methods=["POST"])
@login_required
def api_click():
    """
    웹에서 클릭 좌표를 pipeline으로 전달
    
    Body: {"x": 320, "y": 240}
    """
    data = request.get_json() or {}
    x = data.get("x", 0)
    y = data.get("y", 0)
    
    # 명령 큐에 추가
    command_queue.put({"type": "click", "x": x, "y": y})
    
    print(f"[API] 클릭: ({x}, {y})")
    
    return jsonify({
        "status": "ok",
        "selected_id": tracking_stats.get("selected_id")
    })


# ===== API: 선택 해제 =====
@app.route("/api/clear_selection", methods=["POST"])
@login_required
def api_clear_selection():
    """선택 해제"""
    command_queue.put({"type": "clear_selection"})
    
    print(f"[API] 선택 해제")
    
    return jsonify({
        "status": "ok",
        "message": "선택 해제됨"
    })


# ===== API: 통계 조회 =====
@app.route("/api/stats")
@login_required
def api_stats():
    """실시간 통계 반환"""
    return jsonify(tracking_stats)


# ===== API: 통계 업데이트 (pipeline에서 호출) =====
def update_tracking_stats(stats_dict):
    """
    pipeline.py에서 호출하는 함수
    
    예: update_tracking_stats({"fps": 28.5, "total_tracks": 5})
    """
    global tracking_stats
    tracking_stats.update(stats_dict)
    tracking_stats["last_update"] = time.time()
