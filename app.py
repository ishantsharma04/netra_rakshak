from flask import Flask, render_template, Response, request, jsonify
import cv2, time, threading, os, tempfile
from ultralytics import YOLO

app=Flask(__name__)
model=YOLO("yolo11n.pt")
cap=None
source_lock=threading.Lock()
events=[]
offline_mode=False
zone=(0.62,0.15,0.98,0.88)  # normalized x1,y1,x2,y2

VEHICLES={"car","motorcycle","bus","truck","bicycle"}
ANIMALS={"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}

def add_event(kind, severity, detail):
    now=time.strftime("%H:%M:%S")
    events.insert(0, {"time":now,"kind":kind,"severity":severity,"detail":detail})
    del events[30:]

def point_in_zone(x,y,w,h):
    x1,y1,x2,y2=zone
    return x1*w <= x <= x2*w and y1*h <= y <= y2*h

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
        results=model.predict(frame,conf=.45,verbose=False,device="cpu")
        person_intrusion=False
        detections=0
        for r in results:
            for b in r.boxes:
                cls=int(b.cls[0])
                name=model.names[cls]
                conf=float(b.conf[0])
                x1,y1,x2,y2=map(int,b.xyxy[0])
                detections+=1
                if name=="person":
                    cx=(x1+x2)//2; cy=(y1+y2)//2
                    inside=point_in_zone(cx,cy,w,h)
                    label=f"PERSON {conf*100:.0f}%"
                    color=(0,0,255) if inside else (255,190,0)
                    if inside: person_intrusion=True
                elif name in VEHICLES:
                    label=f"{name.upper()} {conf*100:.0f}%"; color=(0,220,100)
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
