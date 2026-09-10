from flask import Flask, render_template, Response, request, jsonify
import cv2, time, threading, os, tempfile, re
import pytesseract
import sounddevice as sd
import numpy as np
from ultralytics import YOLO

app=Flask(__name__)
model=YOLO("yolo11n.pt")
cap=None
source_lock=threading.Lock()
events=[]
offline_mode=False
zone=(0.62,0.15,0.98,0.88)  # normalized x1,y1,x2,y2
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
anpr_seen={}
track_history={}

# Audio anomaly detection
audio_state={"level":0.0,"status":"STARTING","last_alert":0.0}
audio_lock=threading.Lock()
AUDIO_THRESHOLD=0.18
AUDIO_COOLDOWN=5.0

def audio_callback(indata, frames, callback_time, status):
    if status:
        pass
    level=float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
    now=time.time()
    with audio_lock:
        audio_state["level"]=round(min(level*5.0,1.0),3)
        if level >= AUDIO_THRESHOLD:
            audio_state["status"]="ANOMALY"
            if now-audio_state["last_alert"] >= AUDIO_COOLDOWN:
                add_event("Audio Anomaly","MEDIUM",
                          f"Sudden sound spike detected (level {level:.2f})")
                audio_state["last_alert"]=now
        else:
            audio_state["status"]="NORMAL"

def start_audio_monitor():
    try:
        stream=sd.InputStream(channels=1,samplerate=16000,
                              blocksize=1024,callback=audio_callback)
        stream.start()
        return stream
    except Exception as e:
        with audio_lock:
            audio_state["status"]="UNAVAILABLE"
        print("Audio monitor unavailable:",e)
        return None

audio_stream=start_audio_monitor()

VEHICLES={"car","motorcycle","bus","truck","bicycle"}
ANIMALS={"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}

def add_event(kind, severity, detail):
    now=time.strftime("%H:%M:%S")
    events.insert(0, {"time":now,"kind":kind,"severity":severity,"detail":detail})
    del events[30:]

def point_in_zone(x,y,w,h):
    x1,y1,x2,y2=zone
    return x1*w <= x <= x2*w and y1*h <= y <= y2*h


def clean_plate_text(text):
    return re.sub(r"[^A-Z0-9]","",text.upper())

def read_plate(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size==0:
        return None,0
    gray=cv2.cvtColor(vehicle_crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    gray=cv2.bilateralFilter(gray,9,75,75)
    variants=[
        cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY,31,11)
    ]
    candidates=[]
    gh,gw=gray.shape[:2]
    for binary in variants:
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            ratio=w/float(h) if h else 0
            if 2.0<=ratio<=6.5 and w*h>=max(120,gw*gh*0.002):
                candidates.append(gray[max(0,y-2):min(gh,y+h+2),
                                       max(0,x-2):min(gw,x+w+2)])
    candidates.append(gray)
    best=None
    best_score=0
    for candidate in candidates[:25]:
        if candidate.size==0:
            continue
        candidate=cv2.resize(candidate,None,fx=1.5,fy=1.5,
                             interpolation=cv2.INTER_CUBIC)
        candidate=cv2.GaussianBlur(candidate,(3,3),0)
        _,candidate=cv2.threshold(candidate,0,255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            raw=pytesseract.image_to_string(candidate,config=config)
        except Exception:
            return None,0
        plate=clean_plate_text(raw)
        if 5<=len(plate)<=12:
            score=min(100,len(plate)*9)
            if any(ch.isdigit() for ch in plate): score+=15
            if any(ch.isalpha() for ch in plate): score+=15
            if score>best_score:
                best_score=score
                best=plate
    return best,best_score

def frames():
    global cap
    if cap is None:
        cap=cv2.VideoCapture(0)
    while True:
        ok,frame=cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ok,frame=cap.read()
            if not ok:
                time.sleep(.1); continue
        h,w=frame.shape[:2]
        results=model.track(frame,conf=.45,verbose=False,device="cpu",persist=True,tracker="bytetrack.yaml")
        person_intrusion=False
        detections=0
        for r in results:
            for b in r.boxes:
                cls=int(b.cls[0])
                name=model.names[cls]
                conf=float(b.conf[0])
                track_id=int(b.id[0]) if b.id is not None else None
                x1,y1,x2,y2=map(int,b.xyxy[0])
                detections+=1
                if track_id is not None:
                    hist=track_history.setdefault(track_id,[])
                    hist.append({"time":time.strftime("%H:%M:%S"),"class":name,
                                 "camera":"CAM-01","x":(x1+x2)//2,"y":(y1+y2)//2})
                    if len(hist)>50: del hist[:-50]
                if name=="person":
                    cx=(x1+x2)//2; cy=(y1+y2)//2
                    inside=point_in_zone(cx,cy,w,h)
                    label=f"PERSON #{track_id} {conf*100:.0f}%" if track_id is not None else f"PERSON {conf*100:.0f}%"
                    color=(0,0,255) if inside else (255,190,0)
                    if inside: person_intrusion=True
                elif name in VEHICLES:
                    label=f"{name.upper()} #{track_id} {conf*100:.0f}%" if track_id is not None else f"{name.upper()} {conf*100:.0f}%"; color=(0,220,100)
                    vehicle_crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
                    plate,plate_conf=read_plate(vehicle_crop)
                    if plate and plate_conf>=45:
                        label=f"{name.upper()} | PLATE {plate} {plate_conf}%"
                        cv2.putText(frame,f"ANPR: {plate}",
                                    (x1,max(42,y1-28)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                        now=time.strftime("%H:%M:%S")
                        if anpr_seen.get(plate) != now:
                            add_event("ANPR Detection","MEDIUM",
                                      f"{plate} detected on {name}")
                            anpr_seen[plate]=now
                elif name in ANIMALS:
                    label=f"{name.upper()} {conf*100:.0f}%"; color=(255,170,0)
                else:
                    continue
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        zx1,zy1,zx2,zy2=int(zone[0]*w),int(zone[1]*h),int(zone[2]*w),int(zone[3]*h)
        cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),(0,0,255),2)
        cv2.putText(frame,"RESTRICTED ZONE",(zx1,zy1-8),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
        if person_intrusion:
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),5)
            cv2.putText(frame,"INTRUSION ALERT",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if not events or events[0]["kind"]!="Restricted Zone Intrusion" or events[0]["time"]!=time.strftime("%H:%M:%S"):
                add_event("Restricted Zone Intrusion","HIGH","Person detected inside virtual fence")
        cv2.putText(frame,f"IBVAP | AI detections: {detections}",(15,h-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,235,250),2)
        ok,buf=cv2.imencode(".jpg",frame)
        if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"

@app.get("/entities")
def entities():
    return jsonify({"entities":[{"track_id":tid,"class":hist[-1]["class"],
                                 "observations":hist[-20:]} for tid,hist in track_history.items() if hist]})

@app.get("/audio-status")
def audio_status():
    with audio_lock:
        return jsonify(audio_state)

@app.get("/anpr-status")
def anpr_status():
    return jsonify({"tesseract":os.path.exists(TESSERACT_PATH),
                    "path":TESSERACT_PATH})

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video")
def video(): return Response(frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload")
def upload():
    global cap
    f=request.files.get("video")
    if not f: return jsonify({"ok":False,"error":"No video selected"}),400
    path=os.path.join(tempfile.gettempdir(),"ibvap_video.mp4")
    f.save(path)
    with source_lock:
        if cap: cap.release()
        cap=cv2.VideoCapture(path)
    add_event("Video Source Changed","LOW","Uploaded video is now the active CCTV simulation")
    return jsonify({"ok":True})

@app.get("/events")
def get_events(): return jsonify({"events":events,"offline":offline_mode})

@app.post("/offline")
def offline():
    global offline_mode
    offline_mode=not offline_mode
    add_event("Network State","MEDIUM" if offline_mode else "LOW",
              "Offline edge mode active" if offline_mode else "Network restored; synchronization available")
    return jsonify({"offline":offline_mode})

@app.post("/clear")
def clear():
    events.clear(); return jsonify({"ok":True})

if __name__=="__main__":
    app.run(host="127.0.0.1",port=5000,debug=False,threaded=True)
