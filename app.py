from flask import Flask, render_template, jsonify
import threading
import time
import live_detection_v3 as live

app = Flask(__name__)

running = False
emergency_detected = False
phase = "Idle"
logs = []
message = "No Emergency Detected"

# phase timer
phase_start_time = time.time()
PHASE_DURATION = 4


def detection_loop():

    global running
    global emergency_detected
    global phase
    global logs
    global message
    global phase_start_time

    while running:

        # =========================
        # DISTRESS PHASE
        # =========================

        phase = "distress"
        phase_start_time = time.time()

        result = live.detect("distress")

        message = result["message"]

        new_logs = result.get("logs", [])

        for log in new_logs:
            logs.append(log)

        if len(logs) > 50:
            logs[:] = logs[-50:]

        if result["status"] == "emergency":
            emergency_detected = True
        else:
            emergency_detected = False

        if not running:
            break


        # =========================
        # KEYWORD PHASE
        # =========================

        phase = "keyword"
        phase_start_time = time.time()

        result = live.detect("keyword")

        message = result["message"]

        new_logs = result.get("logs", [])

        for log in new_logs:
            logs.append(log)

        if len(logs) > 50:
            logs[:] = logs[-50:]

        if result["status"] == "emergency":
            emergency_detected = True
        else:
            emergency_detected = False


@app.route('/')
def home():

    return render_template("index.html")


@app.route('/start')
def start():

    global running

    if not running:

        running = True

        thread = threading.Thread(target=detection_loop)
        thread.daemon = True
        thread.start()

    return jsonify({"status":"started"})


@app.route('/stop')
def stop():

    global running

    running = False

    return jsonify({"status":"stopped"})


@app.route('/status')
def status():

    remaining = max(0, PHASE_DURATION - int(time.time() - phase_start_time))

    return jsonify({
        "emergency": emergency_detected,
        "phase": phase,
        "logs": logs,
        "message": message,
        "timer": remaining
    })


if __name__ == "__main__":

    print("Starting Server...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )