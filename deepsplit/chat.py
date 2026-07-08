"""BRIDGE ONE — AI chat brain.

Optional LLM layer for MASTER CHAT. When an Anthropic API key is present, this
maps the user's plain-English mastering notes to the exact DSP op list the
engine already understands, using Claude (Opus 4.8) with tool use. If the SDK
or a key isn't available, server falls back to the app's local keyword parser —
the chat always works, the AI is a bonus.

Nothing but the user's text + the numeric analysis is sent to the API; the
audio never leaves the machine.
"""
import os

MODEL = os.environ.get("BRIDGESPLIT_CHAT_MODEL", "claude-opus-4-8")

# Same op vocabulary adjust_master() applies. The model returns the FULL desired
# set each turn (it's re-applied to the original AI master, so no generation loss).
_OPS_ENUM = ["lowshelf", "highshelf", "bell", "deess", "width", "saturate",
             "dereverb", "reverb", "tape", "vintagecomp", "vintageeq"]
TOOL = {
    "name": "set_master_adjustments",
    "description": "Set the complete list of processing adjustments to apply to the user's master, "
                   "plus a short friendly reply. Always include every adjustment that should be "
                   "active (not just new ones) — the list fully replaces the current chain.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string",
                      "description": "One or two sentences to the user, plain and warm, saying what you changed and why."},
            "target_lufs": {"type": ["number", "null"],
                            "description": "New integrated loudness target in LUFS (e.g. -8 louder, -12 more dynamic), or null to leave it."},
            "ops": {
                "type": "array",
                "description": "The complete adjustment chain to apply.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": _OPS_ENUM},
                        "f": {"type": "number", "description": "center/corner frequency in Hz (lowshelf/highshelf/bell)"},
                        "q": {"type": "number", "description": "bell Q (width), ~0.7 wide to ~3 narrow"},
                        "gain": {"type": "number", "description": "EQ gain in dB, roughly -6..+6"},
                        "amt": {"type": "number", "description": "0..1 strength (deess/dereverb/reverb/tape/vintagecomp/vintageeq)"},
                        "mult": {"type": "number", "description": "stereo width multiplier for 'width' (0.6 narrower .. 1.4 wider)"},
                        "drive": {"type": "number", "description": "saturation drive for 'saturate' (1.0..1.4)"},
                        "size": {"type": "number", "description": "reverb size 0.3..1.5"},
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["reply", "ops"],
    },
}

def available():
    """AI chat is usable only if the SDK imports AND some credential is resolvable."""
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))

def _system(genre, analysis, target, current_ops):
    def reg(a):  # already-region-averaged dict
        return a
    a = analysis
    lines = [
        "You are the mastering engineer inside BRIDGE ONE, a local audio app. The user tells you, "
        "in plain English, what they want changed about their already-AI-mastered track. You translate "
        "that into a precise chain of DSP adjustments and reply briefly.",
        "",
        "Guiding principles: be musical and restrained — small moves. Match the genre's real sound. "
        "Only change what the user asked for; keep anything already applied that they didn't ask to undo. "
        "You return the COMPLETE chain every time (it replaces the current one).",
        "",
        f"Genre / reference target: {genre}. Its tonal target (dB relative to mids): "
        f"sub {target['sub']:+.1f}, low {target['low']:+.1f}, low-mid {target['lowmid']:+.1f}, "
        f"presence {target['pres']:+.1f}, air {target['air']:+.1f}.",
        "",
        "The user's current master measures:",
        f"  loudness {a['lufs']} LUFS, true peak {a['tp']} dBTP, crest {a['crest']} dB, correlation {a['corr']} "
        f"(1.0 = mono, lower = wider).",
        f"  tone vs target (dB): sub {a['sub']:+.1f}, low {a['low']:+.1f}, low-mid {a['lowmid']:+.1f}, "
        f"presence {a['pres']:+.1f}, air {a['air']:+.1f}. Positive = more than the target there.",
        "",
        "Op guide: lowshelf/highshelf {f,gain} broad tone; bell {f,q,gain} surgical; deess {amt} sibilance; "
        "width {mult} stereo (1.0 = no change; keep >0.5 for mono safety); saturate {drive} density; "
        "tape {amt} analog warmth; vintagecomp {amt} opto glue; vintageeq {amt} Pultec tilt; "
        "dereverb {amt} dry it up; reverb {amt,size} add space. Typical EQ moves are 1–4 dB. "
        "For 'louder'/'quieter' set target_lufs (hip-hop sits ~-8 to -9, more dynamic ~-12 to -14).",
        "",
        "Frequency map: sub ~50 Hz, boom ~90 Hz, warmth ~150 Hz, mud ~300 Hz, box ~500 Hz, "
        "honk ~1 kHz, presence ~3.5 kHz, air 10 kHz shelf.",
        "",
        f"Currently applied ops: {current_ops if current_ops else 'none'}.",
        "Call set_master_adjustments with the full updated chain and your reply.",
    ]
    return "\n".join(lines)

def llm_chat(message, history, current_ops, genre, analysis, target):
    """Returns {'reply', 'ops', 'target_lufs'} or raises."""
    import anthropic
    client = anthropic.Anthropic()
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs.append({"role": "user", "content": message})
    resp = client.messages.create(
        model=MODEL, max_tokens=1500,
        system=_system(genre, analysis, target, current_ops),
        tools=[TOOL], tool_choice={"type": "tool", "name": "set_master_adjustments"},
        messages=msgs,
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "set_master_adjustments":
            out = block.input
            return {"reply": out.get("reply", "Done."),
                    "ops": _sanitize(out.get("ops") or []),
                    "target_lufs": out.get("target_lufs")}
    raise RuntimeError("model returned no adjustment")

def _sanitize(ops):
    """Whitelist + clamp whatever the model produced so the DSP stays safe."""
    clean = []
    for op in ops[:24]:
        t = op.get("type")
        if t not in _OPS_ENUM:
            continue
        o = {"type": t}
        if t in ("lowshelf", "highshelf", "bell"):
            o["f"] = float(min(20000, max(20, op.get("f", 1000))))
            o["gain"] = float(min(9, max(-9, op.get("gain", 0))))
            if t == "bell":
                o["q"] = float(min(8, max(0.3, op.get("q", 1.0))))
        elif t == "width":
            o["mult"] = float(min(1.5, max(0.4, op.get("mult", 1.0))))
        elif t == "saturate":
            o["drive"] = float(min(1.5, max(1.0, op.get("drive", 1.1))))
        elif t == "reverb":
            o["amt"] = float(min(1, max(0, op.get("amt", 0.4))))
            o["size"] = float(min(2, max(0.2, op.get("size", 1.0))))
        else:  # deess/dereverb/tape/vintagecomp/vintageeq
            o["amt"] = float(min(1, max(0, op.get("amt", 0.5))))
        clean.append(o)
    return clean
