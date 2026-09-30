"""Stand-in Ollama for tests: the endpoints hybrid-intent-router calls, with keyword answers."""
import http.server, json, os, sys
STATE = os.environ["STUB_STATE"]
PORT = int(sys.argv[1])

def models():
    import hashlib
    return [{"name": n.replace("__", ":"), "digest": hashlib.sha256(n.encode()).hexdigest()}
            for n in sorted(os.listdir(STATE)) if not n.startswith(".")] if os.path.isdir(STATE) else []

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/api/version": return self._send(200, {"version": os.environ.get("STUB_VERSION", "0.35.2")})
        if self.path == "/api/tags": return self._send(200, {"models": models()})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/systemone":
            msg = json.dumps(body.get("state", "")).lower()
            if "endpoint" in msg or "usage" in msg: probs = {"usage_reports_api": 0.995, "billing_api": 0.005}
            elif "okta" in msg: probs = {"sso_setup": 0.55, "user_management": 0.40, "other": 0.05}
            else: probs = {"other": 0.9, "feature_request": 0.1}
            crit = body["questions"]["intent"]["criteria"]
            full = {k: probs.get(k, 0.0) for k in crit}
            choice = max(full, key=full.get)
            return self._send(200, {"model": body.get("model", "tev1:0.8b"), "usage": {"input_tokens": 50, "output_tokens": 1},
                                    "answers": {"intent": {"type": "choice", "choice": choice, "confidence": full[choice], "probabilities": full}}})
        if self.path == "/api/chat":
            import time; time.sleep(float(os.environ.get("STUB_DELAY", "0")))
            q = body["messages"][-1]["content"].lower()
            h = "account_access_queue" if "okta" in q else "human_triage"
            return self._send(200, {"model": body["model"], "message": {"role": "assistant",
                                    "content": json.dumps({"handler": h, "reason": "Stub fallback reasoning about the message."})}})
        if self.path == "/api/embed": return self._send(404, {"error": "model 'clm-encoder' not found"})
        self._send(404, {"error": "not found"})
    def log_message(self, *a): pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
import subprocess; subprocess.Popen(["sleep", "1000"])  # a stand-in model runner child, spawned once bound
srv.serve_forever()
