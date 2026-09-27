"""Builds cfb_coach/data/cfb27_offense_macros.json — exact CFB 27 offense Custom Adjustment settings
per macro (ordered rows, status, citations) + live fire rules. Re-run after editing values."""
import json
C = "confirmed"; A = "assumed"
SRC = [
 {"id": "ea_help", "title": "EA Help — Custom Adjustments (CFB)", "url": "https://help.ea.com/en/articles/ea-sports-college-football/custom-adjustments/",
  "supports": "Create & Share > Custom Adjustments; presave protection, route concepts and more"},
 {"id": "ea_deep_dive", "title": "EA — College Football 27 Gameplay Deep Dive", "url": "https://www.ea.com/games/ea-sports-college-football/college-football-27/news/college-football-27-gameplay",
  "supports": "Custom Adjustments hold protection, route concepts, hot routes, per-player assignments; LB/L1 applies; ID Mike for the run game; untarget user defender via protection menu; Chip Block for TE/HB on any route; protection menu moved to LT"},
 {"id": "maddenprodigy_setup", "title": "MaddenProdigy — How to set up Custom Adjustments in CFB 27", "url": "https://www.maddenprodigy.com/how-to-set-up-custom-adjustments-college-football-27/",
  "supports": "20 O + 20 D saved, 10 active per game; offense = route concepts, hot routes, protections"},
 {"id": "mmoexp_offense", "title": "MMOexp — Biggest offensive changes in CFB 27", "url": "https://www.mmoexp.com/News/how-to-master-the-biggest-offensive-changes-in-college-football-27.html",
  "supports": "Offense custom adjustments save hot routes + blocking; mapped to depth-chart slots (WR1 follows that player even if the formation flips)"},
 {"id": "katkat", "title": "Katkat — CFB 27 Custom Adjustments guide", "url": "https://katkat.com/blog/college-football-27-custom-adjustments-guide",
  "supports": "Offense package rows: protection slide left/right, ID the Mike, Chip Block, hot route per eligible receiver"},
 {"id": "cfbgg_hot_routes", "title": "CollegeFootball.gg — All of the new hot routes in CFB 26 (full per-position menu)", "url": "https://collegefootball.gg/all-of-the-new-hot-routes-in-cfb-26/",
  "supports": "Hot route menu + button per alignment: Outside WR / Slot WR / RB / TE (e.g. slot D-pad Right = Zig, RB D-pad Right = Texas, TE LT = Stick Nod)"},
 {"id": "civil_hot_routes", "title": "Civil.GG — CFB 26 hot routes guide", "url": "https://civil.gg/tips/cfb-26-hot-routes-guide",
  "supports": "Slot fade / choice / return routes; Return on D-pad Down for every receiver except HB"},
 {"id": "aidan_notes", "title": "Aidan's Dynasty offense macro notes (macro_catalog.json, status confirmed)", "url": "",
  "supports": "The per-macro values (routes per slot, protection, blocking, ball carrier) Aidan wrote down"},
]
# Hot-route menus per alignment (CFB 26 list; assumed unchanged in CFB 27)
MENU = {
 "outside": {"Fade": "LS Up", "Curl": "LS Down", "Out": "LS Left", "In": "LS Right", "Post": "RS Up", "Drag": "RS Down", "Corner": "RS Left", "Slant": "RS Right",
             "Deep Over": "D-pad Up", "Sluggo": "D-pad Right", "Return": "D-pad Left", "Post Sit": "D-pad Down", "Custom Stem": "LB", "Comeback": "LT", "Smoke Screen": "RT", "Smart Route": "RB"},
 "slot": {"Streak": "LS Up", "Curl": "LS Down", "Speed Out": "LS Left", "In": "LS Right", "Post": "RS Up", "Drag": "RS Down", "Corner": "RS Left", "Slant": "RS Right",
          "Deep Cross": "D-pad Up", "Zig": "D-pad Right", "Return": "D-pad Left", "Short Cross": "D-pad Down", "Custom Stem": "LB", "Slot Fade": "LT", "Choice": "RT", "Smart Route": "RB"},
 "rb": {"Streak": "LS Up", "Curl": "LS Down", "Out": "LS Left", "In": "LS Right", "Flat": "RS Up", "Triple Option": "RS Down", "Swing Left": "RS Left", "Swing Right": "RS Right",
        "Corner": "D-pad Up", "Texas": "D-pad Right", "Wheel": "D-pad Left", "Post": "D-pad Down", "Custom Stem": "LB", "Block & Release": "LT", "Pass Block": "RT", "Smart Route": "RB"},
 "te": {"Streak": "LS Up", "Curl": "LS Down", "In": "LS Left", "Out": "LS Right", "Post": "RS Up", "Drag": "RS Down", "Slant": "RS Left", "Corner": "RS Right",
        "Wheel": "D-pad Up", "Deep Cross": "D-pad Right", "Return": "D-pad Left", "Zig": "D-pad Down", "Custom Stem": "LB", "Stick Nod": "LT", "Pass Block": "RT", "Smart Route": "RB"},
}
SLOT_ALIGN = {"WR1": "outside", "WR2": "slot", "WR3": "slot", "TE": "te", "HB": "rb"}

def hr(slot, note_route, menu_route=None, status=C, note=""):
    """Hot route row; note_route = what Aidan's notes say, menu_route = in-game menu name."""
    menu_route = menu_route or note_route
    align = SLOT_ALIGN[slot]
    btn = MENU[align].get(menu_route)
    val = f"{menu_route}" + (f" ({btn})" if btn else "")
    if menu_route != note_route:
        status = A
        note = (note + " " if note else "") + f"Notes say '{note_route}'; the {align} menu has no '{note_route}' — closest real option '{menu_route}'."
    if btn is None:
        status = A
        note = (note + " " if note else "") + f"'{menu_route}' is not a button in the CFB 26 {align} hot-route menu."
    return {"section": "Hot Routes", "setting": slot, "value": val, "status": status, "note": note.strip(),
            "cite": ["cfbgg_hot_routes", "mmoexp_offense"] + (["aidan_notes"] if status == C else [])}

def row(section, setting, value, status, note="", cite=None):
    return {"section": section, "setting": setting, "value": value, "status": status, "note": note, "cite": cite or []}

SECTIONS = [
 {"name": "General", "rows": ["Ball Carrier", "Blocking"], "wording": A, "cite": ["aidan_notes", "ea_help"],
  "note": "Row names taken from Aidan's own RUN notes ('aggressive blocking; conservative ball carrier'); exact CFB 27 label wording not published."},
 {"name": "Pass Protection", "rows": ["Protection", "Untarget User Defender", "Chip Block — TE", "Chip Block — HB"], "wording": A,
  "cite": ["ea_deep_dive", "katkat", "ea_help"], "note": "Slide Left/Right, Max Protect, untarget-user and Chip Block are documented CFB 27 protection options; exact row labels in the Custom Adjustments editor are assumed."},
 {"name": "Run Blocking", "rows": ["OL Technique", "Double Team", "ID the Mike"], "wording": A, "cite": ["ea_deep_dive", "aidan_notes"],
  "note": "EA: ID Mike updated for the run game (pullers / targeting rules). Row names from Aidan's RUN notes."},
 {"name": "Hot Routes", "rows": ["WR1", "WR2", "WR3", "TE", "HB"], "wording": "cited", "cite": ["mmoexp_offense", "cfbgg_hot_routes"],
  "note": "Custom Adjustment hot routes are saved per depth-chart slot (WR1 follows that player even if the formation flips). Route names + buttons are the CFB 26 per-alignment menu (assumed unchanged in CFB 27). Assumes WR1 aligned outside, WR2/WR3 in the slot (Gun Bunch X Nasty)."},
]
DEF = "Default"

def build(values):
    """values: list of explicit rows; fill every other row with Default in section order."""
    by = {(r["section"], r["setting"]): r for r in values}
    out = []
    for sec in SECTIONS:
        for s in sec["rows"]:
            out.append(by.get((sec["name"], s)) or row(sec["name"], s, "Default (stock)" if sec["name"] == "Hot Routes" else DEF, "default"))
    return out

M = {}
M["RZ"] = dict(
 fire_when="Inside the opponent 20 or goal-to-go, on a pass call from the pairs list (Mesh Spot, RZ PA X Whip, Z Spot GoalLine, Z Spot Shake). Skip it on runs, and never on a whip into a live Cover 2 Invert (hard flat jumps it).",
 fire={"zones": ["rz", "gl"], "pass": True, "coverages": [], "avoid": [{"coverage": "c2", "play_re": "whip"}]},
 play_re=r"mesh|spot|whip|shake|cross|fade|slant|stick", pairs_with=["Mesh Spot", "RZ PA X Whip", "Z Spot GoalLine", "Z Spot Shake", "PA RZ Crossers", "Mesh Post"],
 rows=[hr("WR1", "Fade"), hr("WR2", "Zig"), hr("WR3", "Slant"), hr("TE", "Stick Nod"), hr("HB", "Flat")])
M["O-RUN"] = dict(
 fire_when="Inside-zone / base / counter run vs a light box or two-high shell you can see pre-snap (Cover 2, Cover 4/6/9, Quarters, Palms).",
 fire={"zones": ["open", "rz", "gl"], "run": True, "coverages": ["two_high", "c2", "match"]},
 play_re=r"inside zone|hb base|counter|zone|duo|power|stretch|slam|dive", pairs_with=["Inside Zone", "HB Base", "Counter Y", "Inside Zone Split", "Outside Zone", "HB Counter", "Mtn Duo", "HB Stretch"],
 rows=[row("General", "Ball Carrier", "Conservative", C, cite=["aidan_notes"]), row("General", "Blocking", "Aggressive", C, cite=["aidan_notes"]),
       row("Run Blocking", "OL Technique", "Base", C, cite=["aidan_notes"]), row("Run Blocking", "Double Team", "Highest OVR", C, cite=["aidan_notes"]),
       row("Run Blocking", "ID the Mike", "On — ID the Mike (account for the user/threat defender)", C, cite=["aidan_notes", "ea_deep_dive"])])
M["MATCH"] = dict(
 fire_when="Pass call vs a live Cover 4 / Quarters / Palms / Cover 6 / Cover 9 (match) look. Early downs vs match: run first (no macro); fire MATCH when you do throw.",
 fire={"zones": ["open", "rz"], "pass": True, "coverages": ["match"]},
 play_re=r"mesh|drive|dig|spot|under|cross|post", pairs_with=["Mesh Spot", "Drive HB Under", "Mtn Speed Dig Under", "Mesh Post", "Z Spot Shake"],
 rows=[hr("WR1", "Post"), hr("WR2", "Deep Cross"), hr("WR3", "Drag"), hr("TE", "Streak"), hr("HB", "Texas")])
M["C2"] = dict(
 fire_when="Pass call vs a live Cover 2 / Cover 2 Invert / Tampa 2 look (soft middle, hard flats). Not on Whip/Flood into the hard flat.",
 fire={"zones": ["open", "rz"], "pass": True, "coverages": ["c2"], "avoid": [{"coverage": "c2", "play_re": "whip|flood"}]},
 play_re=r"mesh|spot|shake|dig|post|smash|corner", pairs_with=["Mesh Spot", "Z Spot Shake", "Mesh Post", "Mtn Speed Dig Under", "Z Spot", "PA Corner Dig"],
 rows=[hr("WR1", "Streak", "Fade"), hr("WR2", "Corner"), hr("WR3", "Post"), hr("TE", "Slot Fade", "Streak"), hr("HB", "Flat")])
M["MAN"] = dict(
 fire_when="Pass call vs a live Cover 1 / man look (or man repeated in this situation): rubs, mesh and whip beat it.",
 fire={"zones": ["open", "rz", "gl"], "pass": True, "coverages": ["man"]},
 play_re=r"mesh|whip|cross|rub|traffic|drive|shallow|slant", pairs_with=["Mesh Spot", "Mesh Traffic", "Return Whip Trail", "Whip Double Spot", "Irish Mesh Whip", "Mesh Corner"],
 rows=[hr("WR1", "Deep Cross", "Deep Over"), hr("WR2", "Zig"), hr("WR3", "Short Cross"), hr("TE", "Wheel"), hr("HB", "Texas")])
M["O-RPO"] = dict(
 fire_when="On an RPO call when the pre-snap look shows a soft edge / light box (two-high or Cover 3 with an overhang off the ball). Keeps the stock RPO routes; only cleans up blocking.",
 fire={"zones": ["open", "rz"], "rpo": True, "coverages": ["two_high", "c2", "match", "c3"]},
 play_re=r"rpo", pairs_with=["Mtn RPO Zone Alert", "RPO Alert TE Flat", "RPO Alert QB Draw"],
 rows=[row("General", "Blocking", "Balanced", C, cite=["aidan_notes"]),
       row("Run Blocking", "ID the Mike", "On — clean Mike ID", C, cite=["aidan_notes", "ea_deep_dive"]),
       row("Hot Routes", "WR1", "Default (stock) — do NOT overwrite RPO routes", C, cite=["aidan_notes"]),
       row("Hot Routes", "WR2", "Default (stock) — do NOT overwrite RPO routes", C, cite=["aidan_notes"]),
       row("Hot Routes", "WR3", "Default (stock) — do NOT overwrite RPO routes", C, cite=["aidan_notes"]),
       row("Hot Routes", "TE", "Default (stock) — do NOT overwrite RPO routes", C, cite=["aidan_notes"]),
       row("Hot Routes", "HB", "Default (stock)", C, cite=["aidan_notes"])])
M["C3"] = dict(
 fire_when="Pass call vs a live spot-drop Cover 3 (Sky/Buzz): flood / high-low. Never vs Cover 6/9/Quarters.",
 fire={"zones": ["open"], "pass": True, "coverages": ["c3"]},
 play_re=r"flood|cross|drive|boot|corner|sail|curl", pairs_with=["Deep Flood", "Mtn Cross Post", "Drive HB Under", "PA Boot Over", "Curl Pivot Dig"],
 rows=[hr("WR1", "Streak", "Fade"), hr("WR2", "Corner"), hr("WR3", "Flat", "Speed Out"), hr("TE", "Drag")])
M["ZERO"] = dict(
 fire_when="Pass call vs a live Cover 0 / all-out blitz look (also any live pressure look when neither HEAT nor PROT is in your Active 8): max protect + immediate hot throws.",
 fire={"zones": ["open", "rz", "gl"], "pass": True, "coverages": ["c0"], "fallback_pressure": True},
 play_re=r"mesh|slant|whip|spot|quick|out|hitch|stick", pairs_with=["Mesh Spot", "Quick Slants", "Return Whip Trail", "Z Spot", "Whip Double Spot"],
 rows=[row("Pass Protection", "Protection", "Max Protect", C, cite=["aidan_notes", "katkat"]),
       row("Pass Protection", "Chip Block — HB", "On — chip, then release", A, note="Notes say 'RB chip then release' (approx).", cite=["ea_deep_dive"]),
       hr("WR1", "Streak", "Fade"), hr("WR2", "Zig"), hr("WR3", "Slant")])
M["PROT"] = dict(
 fire_when="Pass call vs a live pressure / blitz look (walked-up LBs, sim pressure): protection first, then the hot throw.",
 fire={"zones": ["open", "rz", "gl"], "pass": True, "coverages": ["pressure"]},
 play_re=r"mesh|whip|slant|quick|spot", pairs_with=["Mesh Spot", "Return Whip Trail", "Quick Slants"],
 rows=[row("Pass Protection", "Protection", "Max Protect", A, note="Notes say 'max protect / slide to pressure side [approx]' — slide side cannot be known before the snap; slide at the line with LT if needed.", cite=["katkat", "ea_deep_dive"])])
M["O-HEAT"] = dict(
 fire_when="Pass call vs a live blitz / pressure look: TE chips, HB blocks then releases, quick zig/drag answers.",
 fire={"zones": ["open", "rz"], "pass": True, "coverages": ["pressure"]},
 play_re=r"mesh|whip|slant|quick|spot|hitch", pairs_with=["Mesh Spot", "Return Whip Trail", "Quick Slants"],
 rows=[row("Pass Protection", "Protection", "Default (base)", C, cite=["aidan_notes"]),
       row("Pass Protection", "Chip Block — TE", "On — chip, then release", C, cite=["aidan_notes", "ea_deep_dive"]),
       hr("WR1", "Streak", "Fade"), hr("WR2", "Zig"), hr("WR3", "Drag"), row("Hot Routes", "TE", "Default (stock) — chip is set under Pass Protection", C, cite=["aidan_notes", "ea_deep_dive"]),
       hr("HB", "Block & Release")])
M["SHOT"] = dict(
 fire_when="One scheduled play-action shot on 1st/2nd down in the open field vs a live single-high (Cover 3 / Cover 1) look, after the run is established. Not every drive.",
 fire={"zones": ["open"], "pass": True, "coverages": ["c3", "man"], "downs": [1, 2], "deep_only": True},
 play_re=r"pa |play action|shot|post|sluggo|deep|boot", pairs_with=["PA Read", "PA Boot", "Mtn Cross Post", "PA Stretch Shot", "PA Corner Dig"],
 rows=[hr("WR1", "Sluggo"), hr("WR2", "Deep Cross"), hr("WR3", "Streak"), hr("TE", "Pass Block"),
       row("Pass Protection", "Chip Block — HB", "On — chip, then release", C, cite=["aidan_notes", "ea_deep_dive"])])

out = {"version": "2026-09-27", "game": "CFB 27", "platform": "Xbox",
       "editor_path": ["Create & Share", "Custom Adjustments", "Offense", "Create Adjustment"],
       "in_game": "At the line: LB (Xbox) opens Custom Adjustments → press the button next to the macro name",
       "limits": "Up to 20 offense + 20 defense saved; 10 active per game in EA's UI (Aidan's rule: 8)",
       "status_legend": {"confirmed": "value from Aidan's own macro notes", "assumed": "value/label inferred (see note)", "default": "leave at Default"},
       "sources": SRC, "sections": SECTIONS, "hot_route_menu": MENU, "hot_route_menu_source": "cfbgg_hot_routes (CFB 26; assumed unchanged in CFB 27)",
       "macros": {}}
for mid, d in M.items():
    d = dict(d)
    d["settings"] = build(d.pop("rows"))
    out["macros"][mid] = d
import pathlib
json.dump(out, open(pathlib.Path(__file__).resolve().parents[1] / "cfb_coach/data/cfb27_offense_macros.json", "w"), indent=1, ensure_ascii=False)
print("ok", list(out["macros"]))
for r in out["macros"]["MAN"]["settings"]: print(r["section"], "|", r["setting"], "|", r["value"], "|", r["status"], "|", r["note"])
