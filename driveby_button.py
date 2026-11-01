#!/usr/bin/env python3
import tkinter as tk, subprocess, signal, os, time

REC  = ["python3", "-u", "/home/debo/record.py"]
LOG  = "/home/debo/record.log"
SIZE = 260
last_tap = 0

def rec_pid():
    r = subprocess.run(["pgrep", "-f", "python3 -u /home/debo/record.py"],
                       capture_output=True, text=True)
    p = r.stdout.split()
    return int(p[0]) if p else None

def elapsed(pid):
    r = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True)
    s = int(r.stdout.strip() or 0)
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

def toggle(e=None):
    global last_tap
    if time.time() - last_tap < 2: return      # ignore double taps
    last_tap = time.time()
    pid = rec_pid()
    if pid: subprocess.run(["xset", "s", "reset"])
    if pid:
        os.kill(pid, signal.SIGINT)
        subprocess.run(["sudo", "-n", "systemctl", "stop", "audio-log"])
        lbl.config(text="Stopping…", bg="#555555", font=("DejaVu Sans", 24, "bold"))
    else:
        f = open(LOG, "a")
        subprocess.Popen(REC, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        subprocess.run(["sudo", "-n", "systemctl", "start", "audio-log"])
        lbl.config(text="Starting…", bg="#555555", font=("DejaVu Sans", 24, "bold"))

def refresh():
    pid = rec_pid()
    if pid: subprocess.run(["xset", "s", "reset"])
    if time.time() - last_tap > 2:
        if pid: lbl.config(text=f"STOP\n{elapsed(pid)}", bg="#c62828", font=("DejaVu Sans", 34, "bold"))
        else:   lbl.config(text="DriveBy", bg="#2e7d32", font=("DejaVu Sans", 34, "bold"))
    root.after(1000, refresh)

root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)
W, H = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{SIZE}x{SIZE}+{W-SIZE-40}+{H-SIZE-80}")
lbl = tk.Label(root, text="DriveBy", fg="white", bg="#2e7d32",
               font=("DejaVu Sans", 34, "bold"))
lbl.pack(fill="both", expand=True)
lbl.bind("<Button-1>", toggle)
refresh()
root.mainloop()
