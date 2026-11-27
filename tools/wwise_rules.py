"""Darktide's own music transition rules (Wwise score decoded with wwiser, see memory
reference_darktide_wwise_music) -> state_rules.json: every rule labelled with the game states it
connects. usage: wwise_rules.py <music.bnk.txt> <names.json> <out.json>"""
import re, json, sys, collections
L = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
H = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
nm = lambda i: H.get(int(i), str(i))
objs, cur = [], None
for l in L:
    m = re.search(r"^(\s*)obj  (CAk\w+)\[", l)
    if m and len(m.group(1)) <= 20:
        cur = {"type": m.group(2), "rules": [], "args": [], "leaves": []}; objs.append(cur); stack = []; rule = None; side = None; continue
    if cur is None: continue
    if "ulID = " in l and "id" not in cur:
        cur["id"] = int(re.search(r"ulID = (\d+)", l).group(1))
    if "obj  AkMusicTransitionRule[" in l:
        rule = {"src": {}, "dst": {}}; cur["rules"].append(rule); side = None; continue
    if rule is not None:
        if "AkMusicTransSrcRule" in l: side = "src"; continue
        if "AkMusicTransDstRule" in l: side = "dst"; continue
        if "AkMusicTransitionObject" in l: side = "obj"
        m = re.search(r"(\w+) = (-?\w+)(?: \[(\w+)\])?", l)
        if m:
            k, v, n = m.groups()
            if k in ("srcID", "dstID") and side is None: rule[k] = int(v)
            elif side in ("src", "dst") and k in ("transitionTime", "iFadeOffset", "eSyncType", "eEntryType", "bPlayPostExit", "bPlayPreEntry", "eFadeCurve"):
                rule[side][k] = n or int(v, 0)
            elif side == "obj" and k in ("segmentID", "transitionTime", "iFadeOffset", "bPlayPreEntry", "bPlayPostExit"):
                rule.setdefault("bridge", {})[k] = int(v, 0)
            elif k == "AllocTransObjectFlag":
                rule["link"] = int(v, 0)
                if not rule["link"]: rule = None
            if side == "obj" and k == "bPlayPostExit": rule = None
    m = re.search(r"tid  ulGroup = (\d+)", l)
    if m: cur["args"].append(nm(m.group(1)))
    m = re.search(r"^(\s*)obj  Node\[", l)
    if m:
        d = len(m.group(1)); stack = [s for s in stack if s[0] < d]; stack.append([d, None]); continue
    m = re.search(r"tid  key = (\d+)", l)
    if m and stack: stack[-1][1] = nm(m.group(1))
    m = re.search(r"tid  audioNodeId = (\d+)", l)
    if m and stack: cur["leaves"].append(([s[1] for s in stack][1:], int(m.group(1))))
out = []
for o in objs:
    if o["type"] != "CAkMusicSwitchCntr": continue
    ns = collections.defaultdict(list)
    for path, node in o["leaves"]:
        ns[node].append(dict(zip(o["args"], path)))
    for r in o["rules"]:
        lab = lambda x: "ANY" if x == -1 else ns.get(x, [{"node": x}])
        out.append({"switch": o["id"], "args": o["args"], "src_id": r.get("srcID"), "dst_id": r.get("dstID"),
                    "src": lab(r.get("srcID")), "dst": lab(r.get("dstID")), "src_rule": r["src"], "dst_rule": r["dst"],
                    "link": r.get("link", 0), "bridge": r.get("bridge")})
json.dump(out, open(sys.argv[3], "w"), indent=0)
print(len(out), "rules from", sum(o["type"] == "CAkMusicSwitchCntr" for o in objs), "switch containers")
