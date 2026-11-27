"""Mirror the critic dataset build's progress into cache/lab/build.json (the build was started
before the Lab existed). python -m conductor.lab_watch <build.log>"""
import glob, json, os, re, sys, time
from . import lab

LOG = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "critic")
while True:
    files = glob.glob(os.path.join(OUT, "ex_*.npz"))
    lines = open(LOG, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(LOG) else []
    prog = [l for l in lines if re.match(r"\[\d+/\d+\]", l)]
    total = int(re.match(r"\[\d+/(\d+)\]", prog[-1]).group(1)) if prog else None
    lab.update("build", songs_done=len(files), songs_total=total, last=prog[-1] if prog else "loading songs…",
               recent=prog[-6:], examples=len(files) * 160)
    if total and len(files) >= total:
        lab.update("build", done=True)
        break
    time.sleep(3)
