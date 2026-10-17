"""Per MP3D scene: levels from the .house file, navigable area per level, and whether the
navmesh connects levels (stairs a wheeled robot could take)."""
import sys, json, os
import numpy as np, habitat_sim
root = "/data/rlwagun/rl_data/habitat_official/scene_datasets/mp3d"
SURF = ("table", "counter", "shelf", "shelving", "cabinet", "chest_of_drawers", "desk", "sink", "bed", "sofa", "chair", "stool", "dresser", "nightstand")

def house(sc):
    L, R, cats, objs = {}, [], {}, []
    for line in open(f"{root}/{sc}/{sc}.house"):
        p = line.split()
        if not p: continue
        if p[0] == "R": R.append((int(p[2]), float(p[11])))      # level, bbox z-min
        elif p[0] == "C": cats[int(p[1])] = p[3]                   # category index -> mpcat40-ish name (index 3)
        elif p[0] == "O": objs.append((int(p[2]), int(p[3]), float(p[6])))  # region, category, z
    floor = {}
    for lv, z in R: floor[lv] = min(floor.get(lv, 9e9), z)
    return floor, R, cats, objs

out = {}
for sc in sorted(os.listdir(root)):
    floor, R, cats, objs = house(sc)
    cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = f"{root}/{sc}/{sc}.glb"
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag])); pf = sim.pathfinder
    pts = np.array([pf.get_random_navigable_point() for _ in range(4000)])
    lv = {}
    for k, z in sorted(floor.items()):
        m = np.abs(pts[:, 1] - z) < 0.6
        lv[k] = {"floor_y": round(z, 2), "area_m2": round(m.mean() * pf.navigable_area, 1), "_pts": pts[m][:15]}
    used = [k for k in lv if lv[k]["area_m2"] > 10]
    conn = {}
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            ok = n = 0
            for p, q in zip(lv[a]["_pts"], lv[b]["_pts"]):
                sp = habitat_sim.ShortestPath(); sp.requested_start = p; sp.requested_end = q
                ok += pf.find_path(sp); n += 1
            conn[f"{a}-{b}"] = f"{ok}/{n}"
    for k in lv: lv[k].pop("_pts")
    rlev = {i: R[i][0] for i in range(len(R))}
    surf = {}
    for r, c, z in objs:
        name = cats.get(c, "?")
        if r in rlev and any(s in name for s in SURF):
            surf[rlev[r]] = surf.get(rlev[r], 0) + 1
    out[sc] = {"levels": lv, "usable_levels": len(used), "connected": conn, "surfaces_per_level": surf}
    print(sc, len(used), conn, flush=True)
    sim.close()
json.dump(out, open(sys.argv[1], "w"), indent=1)
