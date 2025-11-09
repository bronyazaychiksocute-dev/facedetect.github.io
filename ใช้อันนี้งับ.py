from flask import Flask, Response, request, jsonify
import cv2
import base64
import numpy as np
import qrcode
import socket
import os
import threading
import time
import mediapipe as mp
import math
from collections import deque

recent_percentages = []

# --- ฟังก์ชันช่วยเหลือ (จากโค้ด MediaPipe) ---

def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - c[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude = math.hypot(*ba) * math.hypot(*bc)
    if magnitude == 0:
        return 0
    angle_rad = math.acos(min(1, max(-1, dot / magnitude)))
    return math.degrees(angle_rad)


def calculate_signed_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[0])
    bc = (c[0] - b[0], c[1] - b[1])
    angle1 = math.atan2(ba[1], ba[0])
    angle2 = math.atan2(bc[1], bc[0])
    angle = math.degrees(angle2 - angle1)
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def moving_average_point(points_deque):
    n = len(points_deque)
    if n == 0:
        return None
    x = sum(p[0] for p in points_deque) / n
    y = sum(p[1] for p in points_deque) / n
    return (int(x), int(y))


def smooth_point(prev, current, alpha=0.4):
    if prev is None:
        return current
    x = int(prev[0] * (1 - alpha) + current[0] * alpha)
    y = int(prev[1] * (1 - alpha) + current[1] * alpha)
    return (x, y)


def summarize_risk(history_list):
    if not history_list:
        return 0.0, "Unknown"
    total = len(history_list)
    risk_score_map = {
        "Normal": 0,
        "Low Risk": 1,
        "Moderate Risk": 2,
        "High Risk": 3,
        "Forward": 2,
        "Backward": 1
    }
    score_total = sum(risk_score_map.get(status, 0) for status in history_list)
    avg_score = score_total / total
    percent = (avg_score / 3) * 100
    if avg_score <= 0.75:
        status = "Normal"
    elif avg_score <= 1.5:
        status = "Low Risk"
    elif avg_score <= 2.3:
        status = "Moderate Risk"
    else:
        status = "High Risk"
    return percent, status


def calculate_OfficeSyndrome_overall_risk(neck_summary, back3_summary, back1_summary):
    # ปรับปรุง: ใช้ neck, back1, back3 ตามชื่อตัวแปรที่ส่งมา
    office_risk = (neck_summary * 0.30) + (back1_summary * 0.10) + (back3_summary * 0.60)
    if office_risk < 25:
        office_status = "Normal"
    elif office_risk < 50:
        office_status = "Low Risk"
    elif office_risk < 75:
        office_status = "Moderate Risk"
    else:
        office_status = "High Risk"
    return office_risk, office_status


def calculate_HerniatedDisc_overall_risk(hip_summary, back1_summary):
    # ปรับปรุง: ใช้ hip_summary และ back1_summary ตามชื่อตัวแปรที่ส่งมา
    Disc_risk = (back1_summary * 0.30) + (hip_summary * 0.70)
    if Disc_risk < 25:
        Disc_status = "Normal"
    elif Disc_risk < 50:
        Disc_status = "Low Risk"
    elif Disc_risk < 75:
        Disc_status = "Moderate Risk"
    else:
        Disc_status = "High Risk"
    return Disc_risk, Disc_status


def calculate_TextNeck_overall_risk(neck_summary):
    TNeck_risk = (neck_summary * 1)
    if TNeck_risk < 25:
        TNeck_status = "Normal"
    elif TNeck_risk < 50:
        TNeck_status = "Low Risk"
    elif TNeck_risk < 75:
        TNeck_status = "Moderate Risk"
    else:
        TNeck_status = "High Risk"
    return TNeck_risk, TNeck_status


def calculate_PRFM_overall_risk(hip_summary, lumbar_summary):
    PRFM_risk = (hip_summary * 0.6) + (lumbar_summary * 0.4)
    if PRFM_risk < 25:
        PRFM_status = "Normal"
    elif PRFM_risk < 50:
        PRFM_status = "Low Risk"
    elif PRFM_risk < 75:
        PRFM_status = "Moderate Risk"
    else:
        PRFM_status = "High Risk"
    return PRFM_risk, PRFM_status

# --- การตั้งค่า MediaPipe และตัวแปร Global (รวมจากทั้งสองโค้ด) ---

# Flask Setup
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


app = Flask(__name__)
local_ip = get_local_ip()
port = 5000
url = f"http://{local_ip}:{port}/camera"


# QR code setup
os.makedirs("static", exist_ok=True)
qr = qrcode.make(url)
qr_path = os.path.join("static", "qrcode.png")
qr.save(qr_path)
print(f"QR Code saved at: {qr_path}")
print(f"Scan this URL on your phone: {url}")


# Global variables สำหรับเฟรมและล็อค
global risk_saved
latest_raw_frame = None
latest_processed_frame = None
frame_lock = threading.Lock()
# --- ตัวแปร Global ใหม่สำหรับส่งสถานะความเสี่ยงไปยังเว็บ ---
latest_risk_summary = None 
risk_summary_lock = threading.Lock()
# -----------------------------------------------------------

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      smooth_landmarks=True,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

# Buffers
buffer_len = 10
ear_buffer = deque(maxlen=buffer_len)
shoulder_buffer = deque(maxlen=buffer_len)
hip_buffer = deque(maxlen=buffer_len)
knee_buffer = deque(maxlen=buffer_len)

# Calibration and Risk Variables
baseline_values = {
    "neck": None,
    "back1": None,
    "back3": None,
    "hip": None,
    "lumbar": None
}

calibration_buffers = {
    "neck": deque(),
    "back1": deque(),
    "back3": deque(),
    "hip": deque(),
    "lumbar": deque()
}

risk_logging = False
risk_start_time = None
risk_duration = 60
risk_history = {
    "neck": [],
    "back1": [],
    "back3": [],
    "hip": [],
    "lumbar": []
}
risk_score_map = {
    "Normal": 0,
    "Low Risk": 1,
    "Moderate Risk": 2,
    "High Risk": 3,
    "Forward": 2,
    "Backward": 1
}

risk_saved = False 

is_calibrating = False
calibrate_start_time = None
calibrate_duration = 3
smoothed_hip = None
smoothed_knee = None
flip_camera = True # ตั้งค่าเป็น True เพื่อกลับภาพให้เหมือนในโค้ดเดิม

# --- Worker thread ประมวลผลภาพ (รวมโค้ด MediaPipe) ---


def process_frames():
    global is_calibrating, calibrate_start_time, risk_logging, risk_start_time
    global smoothed_hip, smoothed_knee, latest_risk_summary, risk_saved
    global latest_raw_frame, latest_processed_frame, is_calibrating, calibrate_start_time, risk_logging, risk_start_time, smoothed_hip, smoothed_knee, latest_risk_summary
    
    while True:
        frame_to_process = None
        with frame_lock:
            # รับเฟรมล่าสุดจากกล้องมือถือ
            if latest_raw_frame is not None:
                frame_to_process = latest_raw_frame.copy()
                latest_raw_frame = None # เคลียร์เฟรมที่ถูกใช้แล้ว

        if frame_to_process is not None:
            frame = frame_to_process

            if flip_camera:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            output_frame = np.zeros((h, w + 2 * 600, 3), dtype=np.uint8) # สร้าง output_frame
            x_offset = 600
            output_frame[:, x_offset:x_offset + w, :] = frame # คัดลอกภาพไปตรงกลาง

            text_x_left = 20
            text_x_right = w + 620
            y_start = 40
            line_height = 30
            
            # Reset Status/Score
            neck_percent, neck_status = 0.0, "Unknown"
            back3_percent, back3_status = 0.0, "Unknown"
            back1_angle, back1_status = 0.0, "Unknown"
            hip_percent, hip_status = 0.0, "Unknown"
            lumbar_angle, lambar_status = 0.0, "Unknown"
            
            color_map = {
                "Normal": (0, 255, 0),
                "Low Risk": (0, 255, 255),
                "Moderate Risk": (0, 128, 255),
                "High Risk": (0, 0, 255),
                "Forward": (96, 234, 119),
                "Backward": (0, 50, 0),
                "Unknown": (150, 150, 150)
            }
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_point(landmark): return (int(landmark.x * w), int(landmark.y * h))

                # ดึงจุดต่างๆ
                left_ear = get_point(landmarks[mp_pose.PoseLandmark.LEFT_EAR])
                right_ear = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])
                left_shoulder = get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
                right_shoulder = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                left_hip = get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                right_hip = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                left_knee = get_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
                right_knee = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])

                # จุดกึ่งกลาง
                ear_center = ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
                hip_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
                knee_center = ((left_knee[0] + right_knee[0]) // 2, (left_knee[1] + right_knee[1]) // 2)

                # Moving Average
                ear_buffer.append(ear_center)
                shoulder_buffer.append(shoulder_center)
                hip_buffer.append(hip_center)
                knee_buffer.append(knee_center)

                ear = moving_average_point(ear_buffer)
                shoulder = moving_average_point(shoulder_buffer)
                hip_raw = moving_average_point(hip_buffer)
                knee_raw = moving_average_point(knee_buffer)

                if None in [ear, shoulder, hip_raw, knee_raw]:
                    with frame_lock:
                        latest_processed_frame = output_frame
                    continue

                smoothed_hip = smooth_point(smoothed_hip, hip_raw)
                smoothed_knee = smooth_point(smoothed_knee, knee_raw)

                # คำนวณมุมและระยะทาง
                dx = ear[0] - shoulder[0]
                dy = ear[1] - shoulder[1]
                back_length = math.hypot(shoulder[0] - smoothed_hip[0], shoulder[1] - smoothed_hip[1])

                neck_angle = math.degrees(math.atan2(abs(dx), abs(dy)))
                back_vertical_angle = calculate_signed_angle(shoulder, smoothed_hip, (smoothed_hip[0], smoothed_hip[1] - 100))
                Lumbar_Angle = calculate_signed_angle(shoulder, smoothed_hip, (smoothed_hip[0], smoothed_hip[1] - 100))
                hip_tilt_angle = calculate_signed_angle(shoulder, smoothed_hip, smoothed_knee)

                neck_percent = neck_angle
                back1_angle = back_vertical_angle
                hip_percent = abs(hip_tilt_angle)
                lumbar_angle = Lumbar_Angle
                
                # Calibration
                if is_calibrating:
                    elapsed = time.time() - calibrate_start_time
                    cv2.putText(output_frame, f"Calibrating... {elapsed:.1f}s", (x_offset + 30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    calibration_buffers["back3"].append(back_length)

                    if elapsed >= calibrate_duration:
                        for key in baseline_values:
                            if calibration_buffers[key]:
                                if key == "back3": # เฉพาะ back3 ที่ใช้ baseline จากการ Calibrate
                                    baseline_values[key] = sum(calibration_buffers[key]) / len(calibration_buffers[key])
                        is_calibrating = False
                        risk_logging = True
                        risk_start_time = time.time()
                        for key in risk_history:
                            risk_history[key].clear()
                        risk_saved = False
                
                # คำนวณ back3_percent
                if baseline_values["back3"] is not None and baseline_values["back3"] != 0:
                    back3_percent = (back_length / baseline_values["back3"]) * 100
                    back3_percent = min(100.0, back3_percent)
                else:
                    back3_percent = 100
                
                # การจัดระดับความเสี่ยง (Real-time Status)
                if neck_percent <= 20:
                    neck_status = "Normal"
                elif neck_percent <= 25:
                    neck_status = "Low Risk"
                elif neck_percent <= 30:
                    neck_status = "Moderate Risk"
                else:
                    neck_status = "High Risk"

                if back1_angle > 7:
                    back1_status = "Backward"
                elif back1_angle < -7:
                    back1_status = "Forward"
                else:
                    back1_status = "Normal"
                    
                if lumbar_angle > 15:
                    lambar_status = "Backward"
                elif lumbar_angle < -10:
                    lambar_status = "Forward"
                else:
                    lambar_status = "Normal"

                if back3_percent <= 85:
                    back3_status = "High Risk"
                elif back3_percent <= 90:
                    back3_status = "Moderate Risk"
                elif back3_percent <= 95:
                    back3_status = "Low Risk"
                else:
                    back3_status = "Normal"

                if hip_percent > 120:
                    hip_status = "Forward"
                elif hip_percent < 90:
                    hip_status = "Backward"
                else:
                    hip_status = "Normal"

                neck_color = color_map[neck_status]
                back1_color = color_map[back1_status]
                back3_color = color_map[back3_status]
                hip_color = color_map[hip_status]
                lumbar_color = color_map[lambar_status]
                
                # Risk Logging
                if risk_logging:
                    elapsed_risk = time.time() - risk_start_time
                    if elapsed_risk <= risk_duration:
                        risk_history["neck"].append(neck_status)
                        risk_history["back1"].append(back1_status)
                        risk_history["back3"].append(back3_status)
                        risk_history["hip"].append(hip_status)
                        risk_history["lumbar"].append(lambar_status) # ใช้ lambar_status แทน lumbar_angle
                        
                        # อัพเดตสถานะความเสี่ยงปัจจุบันสำหรับมือถือ
                        with risk_summary_lock:
                            latest_risk_summary = {
                                "status": "logging",
                                "time_remaining": int(risk_duration - elapsed_risk)
                            }
                    else:
                        risk_logging = False
                # คำนวณสรุปความเสี่ยง (เมื่อเสร็จสิ้นการ Log)
            

                if not risk_logging and any(risk_history["neck"]) and not risk_saved:
                    summary_neck, status_neck = summarize_risk(risk_history["neck"])
                    summary_back1, status_back1 = summarize_risk(risk_history["back1"])
                    summary_back3, status_back3 = summarize_risk(risk_history["back3"])
                    summary_hip, status_hip = summarize_risk(risk_history["hip"])
                    summary_lumbar, status_lumbar = summarize_risk(risk_history["lumbar"])

                    office_risk, office_status = calculate_OfficeSyndrome_overall_risk(
                        summary_neck, summary_back1, summary_back3
                    )
                    # แก้ไขการเรียกใช้ฟังก์ชัน: สลับ argument ตามที่ฟังก์ชันต้องการ (hip_summary, back1_summary)
                    # ในโค้ดเดิมเรียกผิด: calculate_HerniatedDisc_overall_risk(summary_back1, summary_hip)
                    # ในโค้ดใหม่เรียกถูก: calculate_HerniatedDisc_overall_risk(summary_hip, summary_back1)
                    Disc_risk, Disc_status = calculate_HerniatedDisc_overall_risk(
                        summary_hip, summary_back1
                    )
                    TNeck_risk, TNeck_status = calculate_TextNeck_overall_risk(summary_neck)
                    PRFM_risk, PRFM_status = calculate_PRFM_overall_risk(
                        summary_hip, summary_lumbar
                    )


                    # ✅ เก็บเปอร์เซ็นต์ล่าสุดไว้ใน deque (บันทึกย้อนหลัง 10 ครั้ง)
                    if not risk_saved:
                     recent_percentages.append({
                        "Office Syndrome": office_risk,
                        "Herniated Disc": Disc_risk,
                        "Text Neck": TNeck_risk,
                        "Piriformis": PRFM_risk
                     })
                    risk_saved = True



                    # --- อัพเดตตัวแปร Global สำหรับส่ง JSON ไปยังมือถือ ---
                    with risk_summary_lock:
                        latest_risk_summary = {
                            "status": "ready",
                            "results": [
                                {"name": "ออฟฟิศซินโดรม(Office Syndrome)", "percent": f"{office_risk:.1f}%", "risk": office_status},
                                {"name": "หมอนรองกระดูกทับเส้นประสาท(Herniated Disc)", "percent": f"{Disc_risk:.1f}%", "risk": Disc_status},
                                {"name": "ภาวะคอเสื่อม(Text Neck Syndrome)", "percent": f"{TNeck_risk:.1f}%", "risk": TNeck_status},
                                {"name": "สลักเพรชจม(Piriformis Syndrome)", "percent": f"{PRFM_risk:.1f}%", "risk": PRFM_status},
                            ]
                        }
                    # ---------------------------------------------------
                elif not is_calibrating and baseline_values["back3"] is None:
                     with risk_summary_lock:
                        latest_risk_summary = {"status": "uncalibrated"}
                elif not risk_logging and baseline_values["back3"] is not None and not any(risk_history.values()):
                    with risk_summary_lock:
                        latest_risk_summary = {"status": "idle"}
                

                # ส่วนแสดงผลบนหน้าจอ PC (OpenCV)
                
                # แสดงผล Real-time
                cv2.putText(output_frame, "---- Real-time Posture Risk ----", (text_x_left, y_start),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"[Neck] : {neck_percent:.1f} ({neck_status})", (text_x_left, y_start + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, neck_color, 2)
                cv2.putText(output_frame, f"[Back Length] : {back3_percent:.1f}% ({back3_status})", (text_x_left, y_start + 2 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, back3_color, 2)
                cv2.putText(output_frame, f"[Back Tilt] : {back1_angle:.1f} ({back1_status})", (text_x_left, y_start + 3 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, back1_color, 2)
                cv2.putText(output_frame, f"[Hip Tilt] : {hip_percent:.1f} ({hip_status})", (text_x_left, y_start + 4 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, hip_color, 2)
                cv2.putText(output_frame, f"[Lumbar Angle] : {lumbar_angle:.1f} ({lambar_status})", (text_x_left, y_start + 5 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, lumbar_color, 2)
                
                # วาดจุดและเส้น
                cv2.circle(output_frame, (ear[0] + x_offset, ear[1]), 5, neck_color, -1)
                cv2.circle(output_frame, (shoulder[0] + x_offset, shoulder[1]), 5, back1_color, -1)
                cv2.circle(output_frame, (smoothed_hip[0] + x_offset, smoothed_hip[1]), 5, back3_color, -1)
                cv2.circle(output_frame, (smoothed_knee[0] + x_offset, smoothed_knee[1]), 5, back3_color, -1)
                cv2.line(output_frame, (ear[0] + x_offset, ear[1]), (shoulder[0] + x_offset, shoulder[1]), neck_color, 2)
                cv2.line(output_frame, (shoulder[0] + x_offset, shoulder[1]), (smoothed_hip[0] + x_offset, smoothed_hip[1]), back1_color, 2)
                cv2.line(output_frame, (smoothed_hip[0] + x_offset, smoothed_hip[1]), (smoothed_knee[0] + x_offset, smoothed_knee[1]), back3_color, 2)
                cv2.line(output_frame, (smoothed_hip[0] + x_offset, smoothed_hip[1]), (smoothed_hip[0] + x_offset, smoothed_hip[1] - 100), (100, 255, 100), 1)
                
            else: # ถ้าไม่เจอคน
                cv2.putText(output_frame, "No person detected. Please adjust camera.", (x_offset + 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # สรุปความเสี่ยง (Risk Summary) - บน PC
            cv2.putText(output_frame, "---- Risk Summary (1min) ----", (text_x_right, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if risk_logging:
                remaining_time = max(0, risk_duration - int(time.time() - risk_start_time))
                cv2.putText(output_frame, f"Timer : {remaining_time}s (Calibrated)", (text_x_right, y_start + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            elif not is_calibrating and baseline_values["back3"] is None:
                cv2.putText(output_frame, "Press 't' to Calibrate First", (text_x_right, y_start + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif not is_calibrating and baseline_values["back3"] is not None and not any(risk_history.values()):
                cv2.putText(output_frame, "Calibration Done, Risk Logging Idle", (text_x_right, y_start + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            if not risk_logging and any(risk_history["neck"]):
                # คำนวณสรุปความเสี่ยง (ซ้ำกับด้านบน แต่เพื่อใช้แสดงผลบน PC)
                summary_neck, status_neck = summarize_risk(risk_history["neck"])
                summary_back1, status_back1 = summarize_risk(risk_history["back1"])
                summary_back3, status_back3 = summarize_risk(risk_history["back3"])
                summary_hip, status_hip = summarize_risk(risk_history["hip"])
                summary_lumbar, status_lumbar = summarize_risk(risk_history["lumbar"])

                cv2.putText(output_frame, f"Neck : {summary_neck:.1f}% ", (text_x_right, y_start + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(status_neck, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Back (Length) : {summary_back3:.1f}% ", (text_x_right, y_start + 2 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(status_back3, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Back (Tilt) : {summary_back1:.1f}% ", (text_x_right, y_start + 3 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(status_back1, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Hip (Tilt) : {summary_hip:.1f}% ", (text_x_right, y_start + 4 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(status_hip, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Lumbar Angle : {summary_lumbar:.1f}% ", (text_x_right, y_start + 5 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(status_lumbar, (255, 255, 255)), 2)

                office_risk, office_status = calculate_OfficeSyndrome_overall_risk(summary_neck, summary_back1, summary_back3)
                Disc_risk, Disc_status = calculate_HerniatedDisc_overall_risk(summary_hip, summary_back1) # แก้ไขการเรียกใช้ฟังก์ชัน
                TNeck_risk, TNeck_status = calculate_TextNeck_overall_risk(summary_neck)
                PRFM_risk, PRFM_status = calculate_PRFM_overall_risk(summary_hip, summary_lumbar)

                cv2.putText(output_frame, "---- Overall Risk Result ----", (text_x_left, y_start + 9 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Office Syndrome Risk: {office_risk:.1f}% | {office_status}", (text_x_left, y_start + 10 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(office_status, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Herniated Disc Risk: {Disc_risk:.1f}% | {Disc_status}", (text_x_left, y_start + 11 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(Disc_status, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Text Neck Syndrome Risk: {TNeck_risk:.1f}% | {TNeck_status}", (text_x_left, y_start + 12 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(TNeck_status, (255, 255, 255)), 2)
                cv2.putText(output_frame, f"Piriformis Syndrome Risk: {PRFM_risk:.1f}% | {PRFM_status}", (text_x_left, y_start + 13 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(PRFM_status, (255, 255, 255)), 2)
            else:
                 cv2.putText(output_frame, "No summary yet.", (text_x_right, y_start + 2 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)


            with frame_lock:
                latest_processed_frame = output_frame
        else:
            time.sleep(0.001)


# --- Thread แสดงผลบน PC (ปรับปรุงให้รองรับการกด 't' สำหรับ Calibrate) ---
def show_frames():
    global latest_processed_frame, is_calibrating, calibrate_start_time, risk_logging
    
    while True:
        frame_to_show = None
        with frame_lock:
            if latest_processed_frame is not None:
                frame_to_show = latest_processed_frame.copy()
                
        if frame_to_show is not None:
            cv2.imshow('Analysis of Work Posture', frame_to_show)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting display...")
                break
            elif key == ord('t'):
                # รีเซ็ตค่า Calibrate และเริ่มใหม่
                for key in calibration_buffers:
                    calibration_buffers[key].clear()
                is_calibrating = True
                calibrate_start_time = time.time()
                risk_logging = False # หยุดการ Log ความเสี่ยงชั่วคราวขณะ Calibrate
                # รีเซ็ตผลลัพธ์บนมือถือ
                with risk_summary_lock:
                     latest_risk_summary = {"status": "calibrating"}
                print("Starting Calibration...")
        else:
            time.sleep(0.01)
            
    cv2.destroyAllWindows()


# --- Flask Routes ที่แก้ไข ---

@app.route('/risk_status')
def risk_status():
    """Route สำหรับส่งสถานะความเสี่ยงในรูปแบบ JSON ไปยังหน้าเว็บมือถือ"""
    global latest_risk_summary
    with risk_summary_lock:
        # ส่งข้อมูลความเสี่ยงล่าสุด + ข้อมูลย้อนหลัง 10 ครั้ง
        if latest_risk_summary:
            data = latest_risk_summary.copy()
            data["recent"] = list(recent_percentages)
            return jsonify(data)
        else:
            # สถานะเริ่มต้นก่อนการ Calibrate
            return jsonify({"status": "waiting", "recent": list(recent_percentages)})



@app.route('/')
def index():
    # local_ip และ port ต้องถูกกำหนดไว้ภายนอกฟังก์ชันนี้
    stream_url = f"http://{local_ip}:{port}/video_feed" 
    
    # URL สำหรับมือถือ (สมมติว่า 'url' มาจากตัวแปรที่ถูกกำหนดไว้แล้ว เช่น f"http://{local_ip}:{port}")
    # หาก 'url' ไม่ได้กำหนดไว้ในโค้ดเดิม ให้ใช้ตัวแปรเดียวกับ stream_url หรือ URL หลักของเซิร์ฟเวอร์
    # ในตัวอย่างนี้ ผมจะสมมติว่าตัวแปร {url} มีค่าถูกต้องอยู่แล้วตามโค้ดเดิม
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Mobile Camera CV Stream Setup</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Prompt:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --accent: #ff2e2e;
            --accent2: #ff5e5e;
            --text: #ffffff;
            --overlay: rgba(0, 0, 0, 0.55);
            --glass: rgba(255, 255, 255, 0.08);
            --border-shadow: 0 0 20px rgba(255, 0, 0, 0.6);
        }}

        body {{
            margin: 0;
            font-family: "Prompt", sans-serif;
            color: var(--text);
            text-align: center;
            background: url("https://images.unsplash.com/photo-1602524811689-bb2c57b3b45a?auto=format&fit=crop&w=1920&q=80")
                no-repeat center center fixed;
            background-size: cover;
            background-blend-mode: overlay;
            background-color: #1a0000;
        }}

        .overlay {{
            position: fixed;
            inset: 0;
            background: var(--overlay);
            z-index: 1;
        }}

        .content {{
            position: relative;
            z-index: 2;
            padding: 40px 20px;
            max-width: 960px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2.5em;
            font-weight: 300;
            color: var(--accent);
            text-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
            margin-bottom: 25px;
        }}

        p {{
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #eee;
        }}

        a {{
            color: var(--accent2);
            text-decoration: none;
            font-weight: bold;
            transition: color 0.3s;
        }}

        a:hover {{
            color: var(--accent);
            text-shadow: 0 0 5px var(--accent2);
        }}

        img {{
            border-radius: 12px;
            box-shadow: var(--border-shadow);
        }}

        .qr-code {{
            border: 5px solid var(--accent);
            margin-bottom: 20px;
        }}

        .stream-img {{
            border: 3px solid var(--accent);
            margin-top: 20px;
        }}

        hr {{
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(255, 46, 46, 0), rgba(255, 46, 46, 0.75), rgba(255, 46, 46, 0));
            margin: 30px 0;
        }}

        .note {{
            background: var(--glass);
            padding: 15px;
            border-radius: 10px;
            margin: 25px auto;
            max-width: 450px;
            backdrop-filter: blur(6px);
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.3);
        }}

        .key-hint {{
            font-size: smaller;
            color: #ccc;
            margin-top: 10px;
        }}

        b {{
            color: var(--accent2);
        }}

    </style>
</head>
<body>
    <div class="overlay"></div>
    <div class="content">
        <h1>Mobile Camera CV Stream</h1>
        
        <p>สแกน QR code หรือเข้า URL บนมือถือ:</p>
        <img src="/static/qrcode.png" width="200" class="qr-code">
        <p><a href="{url}">{url}</a></p>
        
        <hr>
        
        <p>ดูสตรีมบนเว็บ:</p>
        <p><a href="{stream_url}">{stream_url}</a></p>
        <img src="{stream_url}" width="480" class="stream-img">
        
        <hr>
        
        <div class="note">
            <p>ผลลัพธ์จะโชว์บนหน้าต่าง OpenCV ของ PC และ <b>ด้านล่างของกล้องในหน้ามือถือ</b></p>
            <p class="key-hint">กด **'t'** บนหน้าต่าง OpenCV เพื่อเริ่ม Calibrate | กด **'q'** เพื่อออกจากโปรแกรม</p>
        </div>
    </div>
</body>
</html>
    """

@app.route('/camera')
def camera():
    """
    HTML/JS สำหรับหน้ากล้องมือถือ
    * แก้ไขให้แสดงผลเป็นแท่งบาร์และเปลี่ยนสีตามระดับความเสี่ยง
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Camera Client Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Prompt:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --main-bg: linear-gradient(135deg, #1a0000 0%, #330000 50%, #660000 100%);
      --glass: rgba(255, 255, 255, 0.08);
      --accent: #ff2e2e;
      --accent2: #ff5e5e;
      --text: #ffffff;
      --overlay: rgba(0, 0, 0, 0.55);
      --sidebar-width: 250px; /* กำหนดความกว้าง Sidebar */
      --sidebar-bg: rgba(51, 0, 0, 0.9); /* พื้นหลัง Sidebar */
    }

    body {
      margin: 0;
      font-family: "Prompt", sans-serif;
      color: var(--text);
      /* ลบ background-image ออกจาก body เพื่อให้ Sidebar แสดงผลได้ดีขึ้น */
      background-color: #1a0000;
      min-height: 100vh;
      display: flex; /* ใช้ Flexbox สำหรับ Body เพื่อจัด Sidebar และ Main Content */
    }

    /* **โครงสร้างหลัก** */
    .app-wrapper {
      position: fixed;
      inset: 0;
      background: url("https://images.unsplash.com/photo-1602524811689-bb2c57b3b45a?auto=format&fit=crop&w=1920&q=80")
        no-repeat center center fixed;
      background-size: cover;
      background-blend-mode: overlay;
      background-color: #1a0000;
      z-index: 0;
    }
    .overlay {
      position: fixed;
      inset: 0;
      background: var(--overlay);
      z-index: 1;
    }

    /* **Sidebar Style** */
    .sidebar {
      width: var(--sidebar-width);
      background-color: var(--sidebar-bg);
      backdrop-filter: blur(8px);
      box-shadow: 2px 0 15px rgba(0, 0, 0, 0.5);
      position: fixed;
      top: 0;
      left: 0;
      bottom: 0;
      z-index: 10;
      padding-top: 20px;
    }
    .sidebar-header {
      text-align: center;
      padding: 10px 0 30px;
      color: var(--accent);
      font-size: 1.2em;
      font-weight: 600;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      margin-bottom: 20px;
    }
    .nav-item {
      padding: 15px 20px;
      cursor: pointer;
      transition: background-color 0.3s, color 0.3s;
      font-weight: 400;
      font-size: 1.05em;
      display: flex;
      align-items: center;
    }
    .nav-item:hover, .nav-item.active {
      background-color: var(--accent);
      color: var(--text);
      box-shadow: inset 5px 0 0 #fff;
    }
    .nav-item i {
      margin-right: 15px;
      font-size: 1.2em;
    }

    /* **Main Content Area** */
    .main-content-area {
      margin-left: var(--sidebar-width); /* ขยับเนื้อหาหลักให้พ้น Sidebar */
      flex-grow: 1;
      position: relative;
      z-index: 2;
      padding: 40px 20px;
      text-align: center;
      max-width: calc(100% - var(--sidebar-width));
    }

    .content-page {
        display: none; /* ซ่อนทุกหน้าไว้ก่อน */
        max-width: 1000px;
        margin: 0 auto;
    }
    .content-page.active {
        display: block; /* แสดงเฉพาะหน้าที่ถูกเลือก */
    }


    /* **Styles สำหรับ Dashboard Page (เหมือนเดิม)** */
    h1 {
      font-size: 2.3em;
      font-weight: 700;
      color: var(--accent);
      text-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
    }

    h2 {
      font-weight: 400;
      font-size: 1.1em;
      color: #eee;
      opacity: 0.9;
      margin-bottom: 25px;
    }

    .camera-section {
      text-align: center;
      margin-bottom: 20px;
    }

    #video {
      width: 90%;
      max-width: 420px;
      border-radius: 12px;
      border: 3px solid var(--accent);
      box-shadow: 0 0 25px rgba(255, 0, 0, 0.6);
      display: block;
      margin: 0 auto;
    }

    #message {
      margin-top: 15px;
      font-weight: 600;
    }
    
    /* **Styles สำหรับ Progress Bar และ Risk Colors** */

/* ภาชนะบรรจุแต่ละ Risk Item */
.risk-item {
  margin-bottom: 15px;
  padding-bottom: 5px;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
}

.risk-item:last-child {
  border-bottom: none;
}

.risk-name {
  display: block;
  font-weight: 400;
  font-size: 1em;
  margin-bottom: 5px;
}

/* ภาชนะบรรจุ Progress Bar */
.progress-bar-container {
  width: 100%;
  height: 18px;
  background-color: rgba(0, 0, 0, 0.4);
  border-radius: 9px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5);
}

/* แถบสีเติม Progress Bar */
.progress-bar-fill {
  height: 100%;
  line-height: 18px;
  text-align: right;
  padding-right: 8px;
  color: #000;
  font-weight: 700;
  font-size: 0.75em;
  transition: width 0.4s ease-out;
}

.color-normal {
  background: linear-gradient(to right, #4CAF50, #8BC34A); 
  color: #000;
}
.color-low {
  background: linear-gradient(to right, #FFEB3B, #FFC107);
  color: #000;
}
.color-moderate {
  background: linear-gradient(to right, #FF9800, #FF5722);
  color: #000;
}
.color-high {
  background: linear-gradient(to right, #F44336, #D32F2F);
  color: #fff; 
}

    #risk-container {
      margin: 30px auto;
      padding: 20px;
      width: 90%;
      max-width: 500px;
      background: var(--glass);
      border-radius: 12px;
      text-align: left;
      box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
      backdrop-filter: blur(6px);
    }
    
    #risk-container h3 {
      text-align: center;
      color: var(--accent2);
      margin-bottom: 12px;
      text-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
    }

    /* **Styles สำหรับ History Page** */
    .history-page h3 {
        font-size: 2em;
        color: var(--accent2);
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
        margin-bottom: 30px;
    }

    .history-card {
        padding: 20px;
        background: var(--glass);
        border-radius: 12px;
        text-align: left;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
        backdrop-filter: blur(6px);
        margin-top: 20px;
    }
    
    #history-chart .bar-content {
        width: 100%; /* ทำให้ยาวเต็มพื้นที่ */
    }

/* ปรับปรุง .entry ให้ใช้ Grid */
.history-card .entry {
    display: grid;
    grid-template-columns: 100px 1fr; /* คอลัมน์สำหรับ Label และ Bar */
    align-items: center;
    gap: 5px 15px; /* ระยะห่างระหว่างแถวและคอลัมน์ */
    margin-bottom: 15px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.history-card .entry:last-child { border-bottom: none; }
.history-card .entry .bar-label {
    text-align: right; /* จัด label ชิดขวาในคอลัมน์แรก */
    font-weight: 600;
    margin-bottom: 0;
    grid-column: 1 / 2; /* กำหนดให้อยู่คอลัมน์แรก */
}
.history-card .entry .bar { 
    grid-column: 2 / 3; /* กำหนดให้อยู่คอลัมน์ที่สอง */
    width: 100%;
}


/* ภาชนะบรรจุแถบ History */
.bar {
  background-color: rgba(255, 255, 255, 0.15); 
  border-radius: 4px;
  height: 15px; 
  /* width: 100%; (กำหนดแล้วใน grid) */
  margin-bottom: 5px;
  overflow: hidden; 
}

/* แถบสีเติม History */
.bar .fill {
  height: 100%;
  transition: width 0.5s ease-out;
  text-align: right;
  padding-right: 5px;
  color: #000;
  font-size: 0.7em;
  font-weight: 600;
  line-height: 15px;
}

/* คลาสสีเฉพาะโรคใน History */
.fill.office { background-color: #4CAF50; } /* เขียว */
.fill.disc { background-color: #FFC107; } /* เหลือง */
.fill.neck { background-color: #2196F3; } /* ฟ้า */
.fill.piriformis { background-color: #E91E63; } /* ชมพู */
    
    /* Style เพิ่มเติมสำหรับ Header ของแต่ละรอบ */
.history-card .entry-header {
    border-bottom: none; /* ลบเส้นคั่นใต้ชื่อรอบ */
    padding-bottom: 0px; 
    margin-bottom: 5px; /* ลดระยะห่างด้านล่าง */
}
.history-card .entry-header .bar-label {
    text-align: left !important; /* จัดชิดซ้าย */
    grid-column: 1 / 3; /* ให้ข้อความครอบคลุม 2 คอลัมน์ */
    color: var(--accent2); /* ใช้สีที่เด่นขึ้น */
    font-weight: 700;
    text-shadow: 0 0 5px rgba(255, 0, 0, 0.5);
}

    .entry {
      margin-bottom: 18px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      padding-bottom: 10px;
    }
    .entry:last-child { border-bottom: none; }
    
    #calibrate-btn {
        /* ขนาดและรูปร่าง */
        padding: 10px 25px; /* เพิ่มขนาด padding ให้ปุ่มใหญ่ขึ้น */
        border-radius: 30px; /* ทำให้ขอบโค้งมนยิ่งขึ้น */
        
        /* สีและพื้นหลัง */
        background: linear-gradient(135deg, var(--accent) 0%, #a00000 100%); /* เปลี่ยนเป็น Gradient สีแดงเข้ม */
        border: 2px solid rgba(255, 255, 255, 0.2); /* เพิ่มเส้นขอบบางๆ */
        color: #fff;
        font-weight: 900; /* ตัวหนาขึ้น */
        font-size: 0.8em; /* ขยายขนาดตัวอักษร */
        letter-spacing: 0.5px; /* เพิ่มระยะห่างตัวอักษรเล็กน้อย */
        
        /* เอฟเฟกต์ */
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(255, 0, 0, 0.8), 0 0 10px rgba(255, 255, 255, 0.1) inset; /* ทำให้เงาสีแดงเด่นชัด */
        transition: all 0.3s ease; /* เพิ่ม transition สำหรับทุกการเปลี่ยนแปลง */
        margin-top: 30px; /* เพิ่มระยะห่างด้านบน */
        text-transform: uppercase; /* เปลี่ยนเป็นตัวพิมพ์ใหญ่ */
    }

    #calibrate-btn:hover {
        transform: scale(1.00) translateY(-2px); /* ปุ่มขยายและลอยขึ้นเล็กน้อย */
        box-shadow: 0 8px 30px rgba(255, 50, 50, 1), 0 0 15px rgba(255, 255, 255, 0.2) inset; /* เงาเข้มขึ้นและกระจายตัว */
        background: linear-gradient(135deg, var(--accent2) 0%, #c00000 100%); /* เปลี่ยนสี Gradient เล็กน้อย */
    }

    #calibrate-btn:active {
        transform: scale(1.05); /* ปุ่มยุบตัวเมื่อกด */
        box-shadow: 0 2px 10px rgba(255, 0, 0, 0.5); /* ลดเงาให้ดูจมลง */
    }

    /* [เหลือสไตล์ย่อยอื่น ๆ ที่เกี่ยวข้องกับ progress bar/risk status] */
    
    /* Media Queries สำหรับมือถือ/จอเล็ก */
    @media (max-width: 767px) {
        .sidebar {
            width: 100%;
            height: auto;
            position: relative;
            box-shadow: none;
            padding-top: 0;
            display: none; /* ซ่อน Sidebar ในมุมมองมือถือ */
        }
        .main-content-area {
            margin-left: 0;
            padding: 20px 10px;
            max-width: 100%;
        }
        body {
            flex-direction: column;
        }
    }
  </style>
</head>
<body>
  <div class="app-wrapper">
    <div class="overlay"></div>
  </div>

  <div class="sidebar">
    <div class="sidebar-header">Posture Monitor</div>
    <div id="nav-dashboard" class="nav-item active" data-page="dashboard">
      <i class="fas fa-tachometer-alt"></i> Dashboard
    </div>
    <div id="nav-history" class="nav-item" data-page="history">
      <i class="fas fa-history"></i> History
    </div>
  </div>

  <div class="main-content-area">
    
    <div id="dashboard-page" class="content-page active">
      <h1>Camera Client Dashboard</h1>
      <h2>Monitor your posture risk levels in real time</h2>
      
      <div class="camera-section">
        <video id="video" autoplay playsinline muted></video>
        <div id="message">Waiting for camera...</div>
      </div>

      <div id="risk-container">
        <h3>📊 Overall Risk Result</h3>
        <div id="risk-display">Waiting for calibration...</div>
      </div>

      <button id="calibrate-btn">Start Calibration (t)</button>
    </div>
    
<div id="history-page" class="content-page">
      <div class="history-card">
        <h3>📈 Risk Trend (Line Chart)</h3>
          <div style="width: 95%; margin: auto; height: 300px; margin-bottom: 20px;"> 
              <canvas id="risk-line-chart"></canvas>
          </div>         <h3>🕒 History</h3>
          <div id="history-chart">
            <p>No history yet.</p>
          </div>
      </div>
    </div>
    
    <footer>
        </footer>
  </div>

  <script>
    // ... [โค้ด JavaScript ส่วนบน (ตัวแปร, riskColorMap, sendCalibrateSignal)] ...
    const video = document.getElementById("video");
    const messageDiv = document.getElementById("message");
    const riskDisplayDiv = document.getElementById("risk-display");
    const historyChart = document.getElementById("history-chart");
    const calibrateBtn = document.getElementById("calibrate-btn");
    const serverUrl = window.location.protocol + "//" + window.location.host;
    const uploadUrl = serverUrl + "/upload_frame";
    const statusUrl = serverUrl + "/risk_status";
    let isSending = false;
    let myLineChart = null;

    const riskColorMap = {
      "Normal": "color-normal",
      "Low Risk": "color-low",
      "Moderate Risk": "color-moderate",
      "High Risk": "color-high",
      Unknown: "",
    };

    async function sendCalibrateSignal() {
      try {
        const response = await fetch(serverUrl + "/start_calibration", { method: "POST" });
        if (response.ok) {
          messageDiv.textContent = "Calibration started! Please assume a correct posture.";
          riskDisplayDiv.innerHTML = "<p>Calibrating... Stand still.</p>";
        } else {
          messageDiv.textContent = "Error starting calibration on PC.";
        }
      } catch {
        messageDiv.textContent = "Network error during calibration request.";
      }
    }
    calibrateBtn.addEventListener("click", sendCalibrateSignal);

    function startCamera() {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then((stream) => {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            messageDiv.textContent = "Camera started! Sending frames to PC...";
            setInterval(sendFrame, 1000 / 30);
            setInterval(fetchRiskStatus, 1000);
          };
        })
        .catch((err) => {
          messageDiv.textContent = `Error: ${err.name}. Please allow camera.`;
        });
    }

    async function sendFrame() {
      if (!video.srcObject || isSending || video.readyState < video.HAVE_ENOUGH_DATA) return;
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataURL = canvas.toDataURL("image/jpeg", 0.7);
      isSending = true;
      try {
        await fetch(uploadUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataURL }),
        });
      } catch (error) {
        messageDiv.textContent = `Network Error: ${error.message}`;
      } finally {
        isSending = false;
      }
    }
    
    function renderLineChart(data) {
      const historyChartCanvas = document.getElementById("risk-line-chart");
      
      if (!data || data.length === 0) {
        if (myLineChart) {
          myLineChart.destroy();
          myLineChart = null;
        }
        return;
      }
      
      // เรียงข้อมูลจากเก่าไปใหม่ (Round #1 -> Round #10)
      const chartData = data.slice(); 
      const labels = chartData.map((_, index) => `Round #${index + 1}`);

      const datasets = [
        {
          label: 'Office Syndrome',
          data: chartData.map(d => d['Office Syndrome']),
          borderColor: '#4CAF50', // เขียว
          backgroundColor: 'rgba(76, 175, 80, 0.4)',
          fill: false,
          tension: 0.3
        },
        {
          label: 'Herniated Disc',
          data: chartData.map(d => d['Herniated Disc']),
          borderColor: '#FFC107', // เหลือง
          backgroundColor: 'rgba(255, 193, 7, 0.4)',
          fill: false,
          tension: 0.3
        },
        {
          label: 'Text Neck',
          data: chartData.map(d => d['Text Neck']),
          borderColor: '#2196F3', // ฟ้า
          backgroundColor: 'rgba(33, 150, 243, 0.4)',
          fill: false,
          tension: 0.3
        },
        {
          label: 'Piriformis',
          data: chartData.map(d => d['Piriformis']),
          borderColor: '#E91E63', // ชมพู
          backgroundColor: 'rgba(233, 30, 99, 0.4)',
          fill: false,
          tension: 0.3
        }
      ];

      const ctx = historyChartCanvas.getContext('2d');
      
      if (myLineChart) {
        // อัพเดตกราฟที่มีอยู่
        myLineChart.data.labels = labels;
        myLineChart.data.datasets = datasets;
        myLineChart.update();
      } else {
        // สร้างกราฟใหม่
        myLineChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: labels,
            datasets: datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
              y: {
                title: { display: true, text: 'Risk Percentage (%)', color: '#fff' },
                min: 0,
                max: 100,
                ticks: { color: '#ccc' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' }
              },
              x: {
                title: { display: true, text: 'Measurement Round', color: '#fff' },
                ticks: { color: '#ccc' },
                grid: { display: false }
              }
            },
            plugins: {
              legend: {
                labels: { color: '#fff' }
              }
            }
          }
        });
      }
    }
    function renderHistoryChart(data) {
      if (!data || data.length === 0) {
        historyChart.innerHTML = "<p>No history yet.</p>";
        return;
      }
      let html = '<div class="bar-content">';
      const reversed = data.slice().reverse();
      
      reversed.forEach((entry, index) => {
        const round = reversed.length - index;
        

        html += `
          <div class="entry entry-header"> 
            <div class="bar-label" style="text-align: left; grid-column: 1 / 3; font-size: 1.1em; padding-bottom: 5px;">
              <strong>Round #${round}</strong>
            </div>
          </div>
          `;
        
        // --- 2. แถวสำหรับ Office Syndrome ---
        html += `
          <div class="entry">
            <div class="bar-label">Office ${entry["Office Syndrome"].toFixed(1)}%</div>
            <div class="bar"><div class="fill office" style="width:${entry["Office Syndrome"]}%"></div></div>
          </div>`;
          
        // --- 3. แถวสำหรับ Herniated Disc ---
        html += `
          <div class="entry">
            <div class="bar-label">Disc ${entry["Herniated Disc"].toFixed(1)}%</div>
            <div class="bar"><div class="fill disc" style="width:${entry["Herniated Disc"]}%"></div></div>
          </div>`;

        // --- 4. แถวสำหรับ Text Neck ---
        html += `
          <div class="entry">
            <div class="bar-label">Neck ${entry["Text Neck"].toFixed(1)}%</div>
            <div class="bar"><div class="fill neck" style="width:${entry["Text Neck"]}%"></div></div>
          </div>`;

        // --- 5. แถวสำหรับ Piriformis ---
        html += `
          <div class="entry">
            <div class="bar-label">Piriformis ${entry["Piriformis"].toFixed(1)}%</div>
            <div class="bar"><div class="fill piriformis" style="width:${entry["Piriformis"]}%"></div></div>
          </div>`;
      });
      
      historyChart.innerHTML = html;
    }

    async function fetchRiskStatus() {
      try {
        const response = await fetch(statusUrl);
        const data = await response.json();
        let htmlContent = "";

        if (data.status === "waiting") {
          htmlContent = "<p>Waiting for first frame...</p>";
        } else if (data.status === "uncalibrated") {
          htmlContent = "<p>Please click Start Calibration to begin.</p>";
        } else if (data.status === "calibrating") {
          htmlContent = "<p>Calibrating... Stand still.</p>";
        } else if (data.status === "logging") {
          htmlContent = `<p>Logging Risk... Time Remaining: ${data.time_remaining}s</p>`;
        } else if (data.status === "ready" && data.results) {
          htmlContent = "<div>";
          data.results.forEach((item) => {
            const percent = parseFloat(item.percent.replace("%", "")) || 0;
            const colorClass = riskColorMap[item.risk] || "color-low";
            htmlContent += `
              <div class="risk-item">
                <span class="risk-name">${item.name} | <span class="${colorClass}">${item.risk}</span></span>
                <div class="progress-bar-container">
                  <div class="progress-bar-fill ${colorClass}" style="width:${Math.min(percent,100)}%;">
                    ${item.percent}
                  </div>
                </div>
              </div>`;
          });
          htmlContent += "</div>";
        } else {
          htmlContent = `<p>Status: ${data.status}</p>`;
        }

       riskDisplayDiv.innerHTML = htmlContent;

         if (data.recent) {
            //  วาดกราฟเส้น 
            renderLineChart(data.recent);
            // 2. วาดกราฟแท่ง 
            renderHistoryChart(data.recent);
        } else {
            renderHistoryChart(null);
            renderLineChart(null); 
        }

    } catch (error) {
        riskDisplayDiv.innerHTML = "<p class='color-high'>Error fetching risk data.</p>";
      }
    }


 
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', function() {
            const targetPage = this.getAttribute('data-page');

   
            document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
            document.querySelectorAll('.content-page').forEach(page => page.classList.remove('active'));

 
            this.classList.add('active');
            document.getElementById(`${targetPage}-page`).classList.add('active');
            
            if (targetPage === 'dashboard') {
                if (video.srcObject && video.paused) video.play();
            } else {

            }
        });
    });


    window.onload = startCamera;
  </script>
</body>
</html>

    """


@app.route('/start_calibration', methods=['POST'])
def start_calibration_from_mobile():
    """Route สำหรับให้มือถือส่งสัญญาณเริ่ม Calibrate"""
    global is_calibrating, calibrate_start_time, risk_logging, latest_risk_summary
    
    with risk_summary_lock:
        latest_risk_summary = {"status": "calibrating"}
        
    for key in calibration_buffers:
        calibration_buffers[key].clear()
    is_calibrating = True
    calibrate_start_time = time.time()
    risk_logging = False
    print("[FLASK] Received mobile signal: Starting Calibration...")
    return jsonify({'status': 'Calibration started'})


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_raw_frame
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'no image'}), 400
    try:
        if ',' in data:
            data = data.split(',')[1]
        frame_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with frame_lock:
            latest_raw_frame = img
        return jsonify({'status': 'ok'})
    except Exception as e:
        print(f"[SERVER] Error decoding frame: {e}")
        return jsonify({'error': str(e)}), 500


def generate_mjpeg():
    global latest_processed_frame
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                continue
            # เข้ารหัสภาพที่ผ่านการประมวลผลแล้ว
            ret, jpeg = cv2.imencode('.jpg', latest_processed_frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Main Execution ---


if __name__ == '__main__':
    # เริ่ม thread ประมวลผลและแสดงผล
    processor_thread = threading.Thread(target=process_frames)
    processor_thread.daemon = True
    processor_thread.start()


    display_thread = threading.Thread(target=show_frames)
    display_thread.daemon = True
    display_thread.start()


    # Run Flask
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)