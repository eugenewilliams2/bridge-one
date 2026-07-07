#!/usr/bin/env python3
"""BRIDGE ONE — Deep Split engine.

Local stem-separation server wrapping Demucs (htdemucs). Runs entirely on
this machine; nothing is uploaded anywhere. The BRIDGE ONE web console
auto-detects it on http://127.0.0.1:8765 and shows a DEEP SPLIT button.
Also serves its own minimal drop-a-file UI at / for standalone use.

Env overrides:
  BRIDGESPLIT_MODEL   demucs model name (default: htdemucs; try htdemucs_ft
                      for max quality at ~4x the time)
  BRIDGESPLIT_DEVICE  cpu | mps   (default: cpu — safest on 8 GB machines)
  BRIDGESPLIT_PORT    port (default: 8765)
"""
import json, os, platform, re, shutil, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def _ensure_native_arch():
    """If we were launched under Rosetta (x86_64 on an Apple Silicon Mac),
    re-exec as arm64 — the venv's compiled packages (numpy/torch) are arm64
    and will fail to import otherwise."""
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return
    try:
        translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                                    capture_output=True, text=True).stdout.strip()
    except Exception:
        translated = "0"
    if translated == "1":
        os.execvp("arch", ["arch", "-arm64", sys.executable] + sys.argv)

_ensure_native_arch()

MODEL  = os.environ.get("BRIDGESPLIT_MODEL", "htdemucs")
DEVICE = os.environ.get("BRIDGESPLIT_DEVICE", "cpu")
PORT   = int(os.environ.get("BRIDGESPLIT_PORT", "8765"))
STEMS  = ["vocals", "drums", "bass", "other"]
MAX_UPLOAD = 300 * 1024 * 1024

JOB = {"id": None, "state": "idle", "pct": 0, "msg": "", "dir": None, "err": "",
       "type": None, "out": None, "report": None}
LOCK = threading.Lock()

def run_enhance(job_id, src_path, workdir, genre, target_lufs, linphase):
    out_path = os.path.join(workdir, "master.wav")
    with LOCK:
        JOB.update(state="running", pct=1, msg="starting engine")
    try:
        import engine  # heavy import (torch) — deferred to the worker thread
        def prog(pct, msg):
            with LOCK:
                if JOB["id"] == job_id:
                    JOB.update(pct=int(pct), msg=msg)
        report = engine.enhance(src_path, out_path, genre=genre,
                                target_lufs=target_lufs, device=DEVICE,
                                model=MODEL, linphase=linphase, progress=prog)
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="done", pct=100, msg="master ready",
                           out=out_path, report=report,
                           master_orig=out_path, genre=report.get("genre", "hiphop"))
    except Exception as e:  # noqa: BLE001 — job errors go to the client
        print(f"enhance job failed: {e}", file=sys.stderr)
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="error", err=str(e)[:240], msg=str(e)[:240])

def run_aimix(job_id, src_path, workdir, genre, mix_lufs):
    stem_dir = os.path.join(workdir, "aimix")
    mix_path = os.path.join(workdir, "mix.wav")
    with LOCK:
        JOB.update(state="running", pct=1, msg="starting engine")
    try:
        import engine  # heavy import (torch) — deferred to the worker thread
        def prog(pct, msg):
            with LOCK:
                if JOB["id"] == job_id:
                    JOB.update(pct=int(pct), msg=msg)
        report = engine.aimix(src_path, stem_dir, mix_path, genre=genre,
                              mix_target_lufs=mix_lufs, device=DEVICE,
                              model=MODEL, progress=prog)
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="done", pct=100, msg="mix ready",
                           dir=stem_dir, out=mix_path, report=report)
    except Exception as e:  # noqa: BLE001 — job errors go to the client
        print(f"aimix job failed: {e}", file=sys.stderr)
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="error", err=str(e)[:240], msg=str(e)[:240])

def run_job(job_id, src_path, workdir):
    out = os.path.join(workdir, "out")
    cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "-d", DEVICE,
           "--segment", "7", "-o", out, "--filename", "{stem}.{ext}", src_path]
    with LOCK:
        JOB.update(state="running", pct=1, msg="loading model (first run downloads weights)")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            m = re.findall(r"(\d{1,3})%\|", line)
            if m:
                pct = min(99, int(m[-1]))
                with LOCK:
                    if JOB["id"] == job_id:
                        JOB.update(pct=max(JOB["pct"], pct), msg="separating stems")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"demucs exited {proc.returncode}")
        stemdir = os.path.join(out, MODEL)
        missing = [s for s in STEMS if not os.path.exists(os.path.join(stemdir, s + ".wav"))]
        if missing:
            raise RuntimeError("missing stems: " + ", ".join(missing))
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="done", pct=100, msg="stems ready", dir=stemdir)
    except Exception as e:  # noqa: BLE001 — job errors go to the client
        print(f"split job failed: {e}", file=sys.stderr)
        with LOCK:
            if JOB["id"] == job_id:
                JOB.update(state="error", err=str(e)[:240], msg=str(e)[:240])

INDEX_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRIDGE ONE — Deep Split</title><style>
body{background:#07080B;color:#EFEADE;font-family:ui-monospace,Menlo,monospace;
 max-width:560px;margin:40px auto;padding:0 16px;font-size:14px;line-height:1.7}
h1{color:#E8B45C;letter-spacing:3px;font-size:20px}
.drop{border:1.5px dashed #3A424F;border-radius:10px;padding:34px 16px;text-align:center;
 color:#8B94A3;cursor:pointer;margin:18px 0}
.drop.armed{border-color:#E8B45C;color:#E8B45C}
.bar{height:9px;background:#04060A;border:1px solid #262C36;border-radius:5px;overflow:hidden;display:none}
.fill{height:100%;width:0;background:linear-gradient(90deg,#8A6A2F,#E8B45C)}
#msg{color:#8B94A3;font-size:12px;min-height:20px}
a{color:#57D9A3;display:block;margin:4px 0}
small{color:#4C5462}</style></head><body>
<h1>BRIDGE ONE · DEEP ENGINE</h1>
<div>AI stem separation + auto-enhance mastering (Demucs · __MODEL__ · __DEVICE__). Local only —
nothing leaves this Mac. Drop a song here, or leave this window running and use the DEEP SPLIT /
AI ENHANCE buttons inside BRIDGE ONE.</div>
<div style="margin:14px 0 4px">
  <label><input type="radio" name="mode" value="split" checked> SPLIT to stems</label> &nbsp;
  <label><input type="radio" name="mode" value="enhance"> ENHANCE (auto-master)</label> &nbsp;
  <select id="genre" style="background:#04060A;color:#EFEADE;border:1px solid #39414E;border-radius:6px;padding:6px">
    <option value="hiphop">hip-hop / trap</option><option value="rnb">r&amp;b</option><option value="pop">pop</option>
  </select>
</div>
<div class="drop" id="d">DROP A SONG / CLICK TO BROWSE</div>
<div class="bar" id="b"><div class="fill" id="f"></div></div>
<div id="msg"></div><div id="links"></div>
<small>Tip: BRIDGESPLIT_MODEL=htdemucs_ft for maximum quality (~4× slower).</small>
<script>
const d=document.getElementById('d'),b=document.getElementById('b'),f=document.getElementById('f'),
msg=document.getElementById('msg'),links=document.getElementById('links');
const pick=document.createElement('input');pick.type='file';
d.onclick=()=>pick.click();
d.ondragover=e=>{e.preventDefault();d.classList.add('armed')};
d.ondragleave=()=>d.classList.remove('armed');
d.ondrop=e=>{e.preventDefault();d.classList.remove('armed');if(e.dataTransfer.files[0])go(e.dataTransfer.files[0])};
pick.onchange=()=>{if(pick.files[0])go(pick.files[0])};
async function go(file){
  const mode=document.querySelector('input[name=mode]:checked').value;
  const genre=document.getElementById('genre').value;
  const ep = mode==='enhance' ? '/enhance?genre='+genre : '/split';
  links.innerHTML='';b.style.display='block';msg.textContent='uploading…';
  const r=await fetch(ep,{method:'POST',headers:{'X-Filename':file.name},body:file});
  if(!r.ok){msg.textContent='engine busy or error — try again';return}
  const {id}=await r.json();
  const iv=setInterval(async()=>{
    const s=await (await fetch('/status?id='+id)).json();
    f.style.width=s.pct+'%';msg.textContent=s.msg+' · '+s.pct+'%';
    if(s.state==='done'){clearInterval(iv);
      if(mode==='enhance'){
        const rep=await (await fetch('/report')).json();
        msg.innerHTML='done — <b>'+rep.after.lufs+' LUFS · '+rep.after.true_peak_db+' dBTP</b><br>'+
          (rep.issues.length?('found: '+rep.issues.join('; ')+'<br>'):'')+
          rep.actions.map(a=>'› '+a).join('<br>');
        links.innerHTML='<a href="/result" download="'+file.name.replace(/\\.[^.]+$/,'')+'_AI_MASTER.wav">⬇ enhanced master (24-bit WAV)</a>';
      }else{
        msg.textContent='done — download your stems:';
        links.innerHTML=['vocals','drums','bass','other'].map(n=>'<a href="/stem?id='+id+'&name='+n+'" download="'+n+'.wav">⬇ '+n+'.wav</a>').join('');
      }}
    if(s.state==='error'){clearInterval(iv);msg.textContent='failed: '+s.msg;}
  },1500);
}
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "X-Filename, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the terminal clean; progress lives in /status

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            with LOCK:
                busy = JOB["state"] == "running"
            self._json(200, {"ok": True, "engine": "demucs", "model": MODEL,
                             "device": DEVICE, "busy": busy})
        elif u.path == "/status":
            with LOCK:
                self._json(200, {k: JOB[k] for k in ("id", "state", "pct", "msg")})
        elif u.path == "/report":
            with LOCK:
                rep = JOB.get("report") if JOB["state"] == "done" else None
            if rep is None:
                self._json(404, {"error": "report not available"})
            else:
                self._json(200, rep)
        elif u.path == "/result":
            with LOCK:
                path = JOB.get("out") if JOB["state"] == "done" else None
            if not path or not os.path.exists(path):
                self._json(404, {"error": "result not available"})
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as fh:
                shutil.copyfileobj(fh, self.wfile)
        elif u.path == "/stem":
            q = parse_qs(u.query)
            name = (q.get("name") or [""])[0]
            with LOCK:
                ok = JOB["state"] == "done" and JOB["dir"] and name in STEMS
                path = os.path.join(JOB["dir"], name + ".wav") if ok else None
            if not path or not os.path.exists(path):
                self._json(404, {"error": "stem not available"})
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as fh:
                shutil.copyfileobj(fh, self.wfile)
        elif u.path == "/app":
            # serve the BRIDGE ONE console itself — one origin, no CORS anywhere
            app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "index.html")
            if not os.path.exists(app_path):
                self._json(404, {"error": "index.html not found next to deepsplit/"})
                return
            with open(app_path, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/":
            body = (INDEX_HTML.replace("__MODEL__", MODEL)
                    .replace("__DEVICE__", DEVICE)).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/adjust":
            self._handle_adjust()
            return
        if u.path not in ("/split", "/enhance", "/aimix"):
            self._json(404, {"error": "not found"})
            return
        with LOCK:
            if JOB["state"] == "running":
                self._json(429, {"error": "engine busy — one song at a time"})
                return
            old = JOB.get("workdir")
        if old and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self._json(400, {"error": "bad upload size"})
            return
        name = os.path.basename(self.headers.get("X-Filename") or "track.wav")
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name) or "track.wav"
        workdir = tempfile.mkdtemp(prefix="bridgesplit_")
        src = os.path.join(workdir, name)
        remaining, chunk = length, 1 << 20
        with open(src, "wb") as fh:
            while remaining > 0:
                data = self.rfile.read(min(chunk, remaining))
                if not data:
                    break
                fh.write(data)
                remaining -= len(data)
        job_id = str(int(time.time() * 1000))
        with LOCK:
            JOB.update(id=job_id, state="queued", pct=0, msg="queued", dir=None,
                       err="", workdir=workdir, out=None, report=None,
                       type=u.path.lstrip("/"))
        q = parse_qs(u.query)
        genre = (q.get("genre") or ["hiphop"])[0]
        if u.path == "/enhance":
            try:
                tgt = float((q.get("lufs") or ["-9.5"])[0])
            except ValueError:
                tgt = -9.5
            tgt = min(-5.0, max(-20.0, tgt))
            linphase = (q.get("linphase") or ["0"])[0] in ("1", "true", "yes")
            threading.Thread(target=run_enhance,
                             args=(job_id, src, workdir, genre, tgt, linphase), daemon=True).start()
        elif u.path == "/aimix":
            try:
                mix_lufs = float((q.get("lufs") or ["-16"])[0])
            except ValueError:
                mix_lufs = -16.0
            mix_lufs = min(-10.0, max(-24.0, mix_lufs))
            threading.Thread(target=run_aimix,
                             args=(job_id, src, workdir, genre, mix_lufs), daemon=True).start()
        else:
            threading.Thread(target=run_job, args=(job_id, src, workdir), daemon=True).start()
        self._json(200, {"id": job_id})

    def _handle_adjust(self):
        """Synchronous cleanup: apply chat-derived ops to the current AI master
        and re-limit. Fast (EQ + limiter, no model) so the chat feels instant."""
        with LOCK:
            orig = JOB.get("master_orig")
            workdir = JOB.get("workdir")
            genre = JOB.get("genre", "hiphop")
            busy = JOB["state"] == "running"
        if busy:
            self._json(429, {"error": "engine busy — wait for the current job"})
            return
        if not orig or not os.path.exists(orig):
            self._json(409, {"error": "no master to adjust — run AI ENHANCE first"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body or b"{}")
        except Exception:
            self._json(400, {"error": "bad JSON body"})
            return
        ops = req.get("ops") or []
        if not isinstance(ops, list) or len(ops) > 32:
            self._json(400, {"error": "ops must be a list (≤32)"})
            return
        try:
            tgt = float(req.get("lufs", -9.5)); tgt = min(-5.0, max(-20.0, tgt))
            ceil = float(req.get("ceiling", -1.0)); ceil = min(-0.3, max(-2.0, ceil))
        except (TypeError, ValueError):
            tgt, ceil = -9.5, -1.0
        out = os.path.join(workdir or tempfile.gettempdir(), "adjusted.wav")
        try:
            import engine
            report = engine.adjust_master(orig, out, ops, genre=genre,
                                          target_lufs=tgt, ceiling_db=ceil)
        except Exception as e:  # noqa: BLE001
            print(f"adjust failed: {e}", file=sys.stderr)
            self._json(500, {"error": str(e)[:240]})
            return
        with LOCK:
            JOB["out"] = out           # /result now returns the adjusted master
        self._json(200, report)

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"BRIDGE ONE Deep Split · demucs/{MODEL} on {DEVICE}")
    print(f"listening on http://127.0.0.1:{PORT}  (BRIDGE ONE will auto-detect)")
    print("keep this window open while splitting · Ctrl+C to quit")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")

if __name__ == "__main__":
    main()
