"""Stand-in Laya for tests: same Router surface, keyword-driven calibrated-looking answers."""
import re
PINNED_REVISIONS = {}

class Router:
    def __init__(self, revisions=None, device=None, **kw):
        self.device = device
    def load(self, name):
        return self
    def predict(self, state, questions):
        text = " ".join(str(v) for v in state.values()).lower()
        out = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                if name == "exploit":
                    p = 0.93 if re.search(r"admin access|ignore previous rules|bypass", text) else 0.04
                else:
                    p = 0.84 if re.search(r"cancel|leave", text) else 0.1
                out[name] = {"noul": p}
            elif q["type"] == "choice":
                if re.search(r"crash|screen|uninstall", text): probs = {"billing": 0.03, "technical": 0.92, "sales": 0.05}
                elif re.search(r"bill|refund|invoice", text): probs = {"billing": 0.95, "technical": 0.03, "sales": 0.02}
                else: probs = {"billing": 0.2, "technical": 0.6, "sales": 0.2}
                c = max(probs, key=probs.get)
                out[name] = {"choice": c, "probabilities": probs, "confidence": probs[c]}
            else:
                out[name] = {"score": 1.2}
        return {"answers": out}
