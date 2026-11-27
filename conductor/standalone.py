"""Standalone mode (user 2026-11-26: "used for a D&D tabletop session"): no game, no telemetry. A local
web page with one button per music state - Calm, Tension, Fight, Horde, Boss - and Pause, which fades
the music out (Play fades it back in where it was). The same state director and blends as in game.

    python -m conductor standalone [--playlist NAME] [--port 8770] [--lan]

Open http://127.0.0.1:8770 (with --lan, any device on your network can open it, e.g. a phone/tablet
at the table).
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .profile import Intent

STATES = ("calm", "build", "fight", "horde", "boss")
LABEL = {"calm": "Calm", "build": "Tension", "fight": "Fight", "horde": "Horde", "boss": "Boss"}
# what the director reads for each state (state_of: mode/tier; tension via forced_state)
INTENT = {
    "calm": dict(energy=0.2, tension=0.0, mode="explore", tier=0),
    "build": dict(energy=0.45, tension=0.8, mode="explore", tier=0, building=True),
    "fight": dict(energy=0.7, tension=0.5, mode="combat", tier=1),
    "horde": dict(energy=0.85, tension=0.6, mode="combat", tier=2),
    "boss": dict(energy=1.0, tension=0.7, mode="combat", tier=3),
}
FADE_OUT, FADE_IN = 3.0, 2.0


def _style_name(style: str) -> str:
    import re
    m = re.match(r"(basscarry|rhyfirst|drumbridge|melfirst|hookover|breakdown)(\d+)", style)
    if m:
        nm = {"basscarry": "Bass carry", "rhyfirst": "Rhythm first", "drumbridge": "Drum bridge",
              "melfirst": "Melody first", "hookover": "Hook over", "breakdown": "Breakdown"}[m.group(1)]
        return f"Layered: {nm} ({m.group(2)} bars)"
    m = re.match(r"gen(\d)_", style)
    if m:
        return f"Layered: AI blend ({m.group(1)} bars)"
    m = re.match(r"xfade(\d)", style)
    if m:
        return f"{m.group(1)}-bar crossfade"
    return {"punch": "Punch in (on the beat)", "blend": "Quick blend", "game": "Darktide-style change",
            "mix": "DJ mix", "dj": "DJ mix", "djverb": "DJ mix + reverb tail", "djblend": "DJ blend",
            "seam": "Seamless jump", "echo": "Echo out", "plain": "Cut"}.get(style, style)


def _intent(st: str) -> Intent:
    return Intent(scene="mission", urgent=st == "boss", dynamic=True, family="mission", **INTENT[st])


class Session:
    def __init__(self, playlist: str | None, cache: str):
        from . import playlists as pls
        from . import states
        from .library import Library
        from .players import LocalPlayer
        from .ratings import Ratings
        from .states import StateDirector
        # a DM's button means now: no in-game hysteresis (the game flips states every few seconds; you don't)
        states.UP_HOLD = states.DOWN_WAIT = states.DOWN_HOLD = states.CALM_WAIT = 0.0
        self.cache = cache
        self.ratings = Ratings(os.path.join(cache, "ratings.json"))
        self.player = LocalPlayer(volume=0.8)
        self.lists, self.ready = {}, {}
        self.playlist, self.lib, self.director = None, None, None
        self.job = None                       # a playlist being added / processed: {"frac", "text", "state", "name"}
        self._job_seen = None
        self._refresh_lists()
        try:
            cfg_cur = json.load(open(pls.CONFIG_PATH, encoding="utf-8")).get("playlist") if os.path.exists(pls.CONFIG_PATH) else None
        except ValueError:
            cfg_cur = None
        done = [n for n in self.lists if self.ready.get(n, (0, 0))[0] > 0]
        pick = next((n for n in (playlist, cfg_cur) if n in done), done[0] if done else None)
        self.state = "calm"
        self.paused = True                    # nothing plays until the DM presses a state (or Play)
        self.started = False
        self.lock = threading.Lock()
        self.msg = "Add a playlist below to start (a Spotify or YouTube Music link)" if not self.lists else \
            "Pick a playlist below - it gets processed first" if pick is None else ""
        if pick:
            self._open(pick)
        self.change = None                    # the last change requested: what it'll look like (the page draws it)

    # ── buttons ──
    def set_state(self, st: str) -> None:
        if self.director is None:
            self.msg = "No songs ready yet - add or pick a playlist below"
            return
        with self.lock:
            self.press_t = time.monotonic()                 # the display's lead-in starts at the press
            if st != self.state:
                self.pressed_from = self.state if self.started else None
            self.state = st
            self.director.forced_state = st
            self.director._hold_until = self.director._hold_strong_until = 0.0   # a press beats "let it play out"
            if self.paused:
                self._resume()
            self.msg = LABEL[st]

    def pause(self) -> None:
        with self.lock:
            if self.paused or not self.started:
                return
            self.paused = True
            self.player.set_trim(0.0, FADE_OUT)
            def stop():
                time.sleep(FADE_OUT + 0.3)
                with self.lock:
                    if self.paused:
                        try:
                            self.player._stream.stop()     # freezes the song where it is
                        except Exception:
                            pass
            threading.Thread(target=stop, daemon=True).start()
            self.msg = "Paused"

    def _resume(self) -> None:
        self.paused = False
        try:
            if not self.player._stream.active:
                self.player._stream.start()
        except Exception:
            pass
        self.player.set_trim(1.0, FADE_IN)

    def play(self) -> None:
        if self.director is None:
            self.msg = "No songs ready yet - add or pick a playlist below"
            return
        with self.lock:
            if self.paused:
                self._resume()
                self.msg = LABEL[self.state]

    def next_song(self) -> None:
        if self.director is None:
            return
        with self.lock:
            self.director.skip()
            self.msg = "Next song"

    def set_volume(self, v: float) -> None:
        self.player.volume = max(0.0, min(1.0, v))

    def _refresh_lists(self) -> None:
        from . import playlists as pls
        try:
            pls.write_index()                 # also finds folders pasted into data/playlists
        except Exception as e:
            print("playlists:", e, flush=True)
        self.lists = pls.all_playlists()
        rd = {}
        for n, f in self.lists.items():
            try:
                rd[n] = pls.counts(f)
            except Exception:
                rd[n] = (0, 0)
        self.ready = rd

    def _open(self, name: str) -> bool:
        """Play from this playlist; one that isn't processed yet gets processed first (then opens by itself)."""
        from .library import Library
        from .states import StateDirector
        ready, total = self.ready.get(name, (0, 0))
        if ready < total:
            self._process(name)
        if ready == 0:
            return False
        lib = Library.build(self.lists[name], self.cache, progress=lambda m: None, backfill=False, cached_only=True)
        if not lib.tracks:
            return False
        with self.lock:
            self.playlist, self.lib = name, lib
            if self.director is None:
                self.director = StateDirector(lib, ratings=self.ratings)
                self.director.forced_state = self.state
            else:
                self.director.lib = lib
                if self.started:
                    self.director.skip()
            if not (self.job and self.job.get("state") == "running"):
                self.msg = f"{name}: {len(lib.tracks)} songs ready"
        return True

    def _process(self, name: str) -> None:
        from . import addlist
        ready, total = self.ready.get(name, (0, 0))
        if addlist.start("local:" + self.lists[name]):
            self.msg = f"Processing {name}: {total - ready} songs (a few minutes each on a GPU)"
            self._want = name
        else:
            self.msg = "Already processing a playlist - pick it again when that's done"

    def set_playlist(self, name: str) -> None:
        if name not in self.lists or name == self.playlist:
            return
        if not self._open(name):
            self.msg = self.msg if self.msg.startswith(("Processing", "Already")) else f"{name} has no songs yet"

    def add(self, url: str) -> None:
        from . import addlist
        url = url.strip()
        if not url.startswith("http"):
            self.msg = "Paste a Spotify or YouTube Music playlist link"
            return
        if addlist.start(url):
            self.msg = "Adding playlist - downloading, then processing (progress below)"
            self._want = "*"
        else:
            self.msg = "Already adding a playlist - wait for it to finish"

    def _poll_job(self) -> None:
        """The add/process job's progress (the same file the game's on-screen bar reads)."""
        from . import addlist
        try:
            with open(addlist.PROGRESS, encoding="utf-8") as f:
                frac, text, state, name = (f.readline().rstrip("\n").split("|") + ["", "", "", ""])[:4]
            st = os.path.getmtime(addlist.PROGRESS)
        except (OSError, ValueError):
            return
        if not getattr(self, "_want", None) and state != "running":
            self.job = None                   # nothing of ours (an old finished job, or none)
            return
        try:
            self.job = {"frac": float(frac), "text": text, "state": state, "name": name}
        except ValueError:
            return
        if state in ("done", "error") and self._job_seen != st:
            self._job_seen = st
            if getattr(self, "_want", None):
                self._refresh_lists()
                want, self._want = self._want, None
                if state == "done" and (self.lib is None or want == "*" or want == name) and name in self.lists:
                    self._open(name)
                self.msg = text

    def _note_cue(self, cue, prev_state, pos) -> None:
        """What the page shows: from -> to, the kind of change, when it starts, how long, and each layer's
        level across it (the real lanes once the player has built the blend)."""
        from . import routes
        r = cue.route or {}
        style = r.get("style") or cue.style or "punch"
        old = self.player.now_playing() or self.director.track
        now = time.monotonic()
        starts_in = max(0.0, (cue.at - pos) if (cue.at is not None and pos is not None) else (cue.delay or 0.0))
        layered = bool(r.get("layered")) or style in routes.LAYER_STYLES
        if layered:
            length = routes._bars_len(cue.track, float(cue.start), routes.LEAD_STYLES.get(style, 4))
            try:
                lanes = routes.layer_plan(style, routes.lands_on_drop(cue.track, float(r.get("land", cue.start))))
            except Exception:
                lanes = []
        else:
            length = float(r.get("dur") or cue.fade or 0.5)
            if style == "punch" or length < 0.6:
                lanes = [{"lane": "Old song", "pts": [(0, 1), (0.98, 1), (1, 0)]},
                         {"lane": "New song", "pts": [(0, 0), (0.98, 0), (1, 1)]}]
            else:
                lanes = [{"lane": "Old song", "pts": [(0, 1), (1, 0)]}, {"lane": "New song", "pts": [(0, 0), (1, 1)]}]
        same = old is not None and old.path == cue.track.path
        self.press_t = getattr(self, "press_t", -1e9)
        frm = getattr(self, "pressed_from", prev_state)
        self.pressed_from = self.state                     # later changes within the same state: X -> X
        self.change = {"id": int(now * 1000), "from_state": frm, "to_state": self.state,
                       "from": old.title if old else "", "to": cue.track.title, "same_song": same,
                       "style": style, "kind": _style_name(style), "layered": layered,
                       "t0": now + starts_in, "length": round(float(length), 2), "lanes": lanes,
                       "lead": round(now + starts_in - self.press_t if now - getattr(self, "press_t", -1e9) < 30
                                     else starts_in, 2),
                       "plan_from": id(routes.LAST_PLAN), "actual": False}

    def status(self) -> dict:
        from . import routes
        t = self.player.now_playing() or (self.director.track if self.director else None)
        ch = None
        if self.change:
            c = self.change
            if c["layered"] and not c["actual"] and routes.LAST_PLAN and id(routes.LAST_PLAN) != c["plan_from"]:
                c["lanes"], c["actual"] = routes.LAST_PLAN, True        # the blend as the player really built it
            now = time.monotonic()
            ch = {k: v for k, v in c.items() if k not in ("t0", "plan_from")}
            ch["starts_in"] = round(c["t0"] - now, 2)
        job = self.job if self.job and (self.job.get("state") == "running" or getattr(self, "_want", None)) else None
        return {"state": self.state, "paused": self.paused, "song": t.title if t else "",
                "playlist": self.playlist, "msg": self.msg, "job": job,
                "playlists": [{"name": n, "ready": r, "total": tt} for n, (r, tt) in self.ready.items()],
                "volume": round(self.player.volume, 2), "change": ch}

    # ── the director loop ──
    def run(self) -> None:
        self._t_start = time.time()
        k = 0
        while True:
            now = time.monotonic()
            k += 1
            if k % 10 == 0:
                try:
                    self._poll_job()
                except Exception as e:
                    print("job poll:", e, flush=True)
            if self.director is not None and not self.paused and not (hasattr(self.player, "busy") and self.player.busy()):
                with self.lock:
                    before = self.director.state
                    pos = self.player.position()
                    cue = self.director.tick(_intent(self.state), now, pos)
                if cue:
                    try:
                        self._note_cue(cue, before, pos)
                    except Exception as e:
                        print("change display:", e, flush=True)
                    self.player.play(cue)
                    self.started = True
            time.sleep(0.1)


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Conductor</title><style>
:root{--bg:#101114;--fg:#e8e6e1;--mute:#8a8a90;--card:#1b1c21;--line:#2c2d34}
body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;display:flex;justify-content:center}
main{width:100%;max-width:560px;padding:20px 16px}
h1{font-size:18px;margin:0 0 4px}#song{color:var(--mute);min-height:1.3em;margin-bottom:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{font:600 20px system-ui;padding:22px 10px;border-radius:12px;border:2px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{border-color:var(--c);box-shadow:0 0 0 2px var(--c) inset;background:color-mix(in srgb,var(--c) 18%,var(--card))}
#calm{--c:#5aa9e6}#build{--c:#c9a227}#fight{--c:#e07b39}#horde{--c:#d64545}#boss{--c:#9b59d0;grid-column:span 2}
.row{display:flex;gap:10px;margin-top:14px}.row button{flex:1;font-size:17px;padding:16px}
#pause{--c:#888}select,input{width:100%;margin-top:14px;font-size:16px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px}
#msg{color:var(--mute);font-size:14px;margin-top:12px;min-height:1.2em}
.add{display:flex;gap:8px;align-items:stretch}.add input{flex:1}.add button{margin-top:14px;font-size:16px;padding:0 16px}
#job{display:none;margin-top:12px;font-size:14px;color:var(--mute)}.bar{height:6px;background:var(--line);border-radius:3px;margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:#5aa9e6}
#chg{margin-top:18px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 12px 8px;display:none}
.ch-h{display:flex;justify-content:space-between;gap:8px;font-weight:600}#chw{color:var(--mute);font-weight:500;white-space:nowrap}
#cht{color:var(--mute);font-size:14px;margin:4px 0 8px}canvas{width:100%;display:block}
</style></head><body><main>
<h1>Conductor</h1><div id=song></div>
<div class=grid>
<button id=calm>Calm</button><button id=build>Tension</button>
<button id=fight>Fight</button><button id=horde>Horde</button>
<button id=boss>Boss</button></div>
<div class=row><button id=pause>Pause</button><button id=next>Next song</button></div>
<label style="display:block;margin-top:14px;color:var(--mute);font-size:14px">Volume<input id=vol type=range min=0 max=1 step=0.01></label>
<section id=chg><div class=ch-h><span id=chk></span><span id=chw></span></div>
<div id=cht></div><canvas id=lanes height=10></canvas></section>
<select id=pl></select>
<div class=add><input id=url placeholder="Add a playlist: paste a Spotify or YouTube Music link"><button id=addb>Add</button></div>
<div id=job><div id=jobt></div><div class=bar><i id=jobb></i></div></div>
<div id=msg></div>
</main><script>
const post=(p,b)=>fetch(p,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json()).then(show);
for(const s of ['calm','build','fight','horde','boss'])document.getElementById(s).onclick=()=>post('/state',{state:s});
let paused=true;
document.getElementById('pause').onclick=()=>post(paused?'/play':'/pause');
document.getElementById('next').onclick=()=>post('/next');
const vol=document.getElementById('vol');vol.onchange=()=>post('/volume',{volume:+vol.value});
const pl=document.getElementById('pl');pl.onchange=()=>{document.getElementById('msg').textContent='Loading '+pl.value+'...';post('/playlist',{name:pl.value})};
const url=document.getElementById('url');
const add=()=>{if(url.value.trim()){post('/add',{url:url.value});url.value=''}};
document.getElementById('addb').onclick=add;url.onkeydown=e=>{if(e.key==='Enter')add()};
let filled='';
function show(s){paused=s.paused;showChange(s.change);
 for(const k of ['calm','build','fight','horde','boss'])document.getElementById(k).classList.toggle('on',!s.paused&&s.state==k);
 document.getElementById('pause').textContent=s.paused?'Play':'Pause';
 document.getElementById('song').textContent=s.song?(s.paused?'Paused: ':'')+s.song:'Press a state to start';
 document.getElementById('msg').textContent=s.msg;
 if(document.activeElement!==vol)vol.value=s.volume;
 const sig=JSON.stringify(s.playlists);
 if(filled!==sig&&document.activeElement!==pl){pl.innerHTML='';
  if(!s.playlists.length){const o=document.createElement('option');o.textContent='(no playlists yet - add one below)';o.value='';pl.appendChild(o)}
  if(!s.playlist&&s.playlists.length){const o=document.createElement('option');o.textContent='Pick a playlist...';o.value='';pl.appendChild(o)}
  for(const p of s.playlists){const o=document.createElement('option');o.value=p.name;
   o.textContent=p.name+(p.total?(p.ready>=p.total?' ('+p.total+' songs)':p.ready?' ('+p.ready+'/'+p.total+' ready - pick to process the rest)':' ('+p.total+' songs, not processed - pick to process)'):'');pl.appendChild(o)}
  filled=sig}
 if(document.activeElement!==pl)pl.value=s.playlist||'';
 const j=document.getElementById('job');j.style.display=s.job?'block':'none';
 if(s.job){document.getElementById('jobt').textContent=s.job.text;document.getElementById('jobb').style.width=(s.job.frac*100).toFixed(1)+'%'}}
const COL={drums:'#e07b39',bass:'#c9a227',melody:'#5aa9e6',hook:'#d64545',sound:'#9b59d0',vocal:'#888'};
let ch=null,chAt=0;
function drawChange(){requestAnimationFrame(drawChange);if(!ch)return;
 document.getElementById('chg').style.display='block';
 // timeline: lead-in (from your press: the current music alone) | the blend | lead-out (the new part alone)
 const si=ch.starts_in-(performance.now()-chAt)/1000, L=Math.max(.3,ch.length);
 const pre=Math.max(2,ch.lead||0), post=Math.max(2,.35*L), T=pre+L+post;
 document.getElementById('chw').textContent=si>0.05?'blend in '+si.toFixed(1)+'s':(si>-L?'blending '+Math.min(100,(-si/L*100)).toFixed(0)+'%':(si>-L-post?'landed':'done'));
 const cv=document.getElementById('lanes'),dpr=devicePixelRatio||1,lanes=ch.lanes||[],rh=18,lw=84,top=14;
 const W=cv.clientWidth,H=top+lanes.length*rh+4;cv.style.height=H+'px';cv.width=W*dpr;cv.height=H*dpr;
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);g.font='11px system-ui';
 const css=getComputedStyle(document.documentElement),mute=css.getPropertyValue('--mute'),line=css.getPropertyValue('--line'),fg=css.getPropertyValue('--fg');
 const X=t=>lw+t/T*(W-lw), xs=X(pre), xe=X(pre+L);
 g.fillStyle=fg;g.globalAlpha=.06;g.fillRect(xs,top,xe-xs,H-top);g.globalAlpha=1;           // the blend window
 g.fillStyle=mute;g.fillText('lead-in',lw+2,10);g.fillText('blend '+L.toFixed(1)+'s',xs+2,10);if(W-xe>40)g.fillText('lead-out',xe+2,10);
 lanes.forEach((ln,i)=>{const y=top+i*rh,k=(ln.lane.split(' ')[1]||''),c=COL[k]||(ln.lane.startsWith('Old')?'#5aa9e6':'#e07b39');
  const P=ln.pts,b=y+rh-4,h=rh-4,v0=P[0][1],v1=P[P.length-1][1];
  g.fillStyle=mute;g.fillText(ln.lane,0,y+12);g.strokeStyle=line;g.strokeRect(lw,y,W-lw,h);
  g.beginPath();g.moveTo(lw,b);g.lineTo(lw,b-v0*h);for(const [x,v] of P)g.lineTo(X(pre+x*L),b-v*h);g.lineTo(W,b-v1*h);g.lineTo(W,b);g.closePath();
  g.globalAlpha=(ln.lane.startsWith('A')||ln.lane.startsWith('Old'))?.55:.9;g.fillStyle=c;g.fill();g.globalAlpha=1});
 g.strokeStyle=mute;g.setLineDash([3,3]);for(const x of [xs,xe]){g.beginPath();g.moveTo(x,top);g.lineTo(x,H);g.stroke()}g.setLineDash([]);
 const t=pre-si;if(t>=0&&t<=T){const x=X(t);g.fillStyle=fg;g.fillRect(x-1,top-2,2,H-top+2)}}
requestAnimationFrame(drawChange);
const NAMES={calm:'Calm',build:'Tension',fight:'Fight',horde:'Horde',boss:'Boss'};
function showChange(c){if(!c)return;
 const drift=ch?Math.abs(c.starts_in-(ch.starts_in-(performance.now()-chAt)/1000)):9;
 if(!ch||c.id!==ch.id||c.actual!==ch.actual||drift>0.3){ch=c;chAt=performance.now()}
 document.getElementById('chk').textContent=(NAMES[c.from_state]||'Start')+' \u2192 '+(NAMES[c.to_state]||'')+' \u00b7 '+c.kind;
 document.getElementById('cht').textContent=(c.same_song?'Same song, new part: ':(c.from?c.from+' \u2192 ':''))+c.to+(c.layered?(c.actual?'  (as built)':'  (planned)'):'')}
setInterval(()=>fetch('/status').then(r=>r.json()).then(show).catch(()=>{}),700);
fetch('/status').then(r=>r.json()).then(show);
</script></body></html>"""


def serve(sess: Session, host: str, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, d):
            b = json.dumps(d).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/status"):
                return self._json(sess.status())
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                d = {}
            p = self.path
            if p == "/state" and d.get("state") in STATES:
                sess.set_state(d["state"])
            elif p == "/pause":
                sess.pause()
            elif p == "/play":
                sess.play()
            elif p == "/next":
                sess.next_song()
            elif p == "/volume":
                sess.set_volume(float(d.get("volume", 0.8)))
            elif p == "/playlist":
                sess.set_playlist(str(d.get("name", "")))
            elif p == "/add":
                sess.add(str(d.get("url", "")))
            self._json(sess.status())

    ThreadingHTTPServer((host, port), H).serve_forever()


def main(playlist: str | None = None, port: int = 8770, lan: bool = False, cache: str | None = None) -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sess = Session(playlist, cache or os.path.join(here, "cache"))
    threading.Thread(target=sess.run, daemon=True).start()
    host = "0.0.0.0" if lan else "127.0.0.1"
    print(f"Conductor standalone: {sess.msg or sess.playlist}. Open http://127.0.0.1:{port}"
          + (" (or this PC's address on your network)" if lan else "") + " - close this window to stop.", flush=True)
    try:
        serve(sess, host, port)
    except KeyboardInterrupt:
        pass
    finally:
        sess.player.close()
