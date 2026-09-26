"""Episode video: top-down map (robots, trails, objects, goals) + each robot's head camera.

Written as H.264 mp4 (plays in VS Code and browsers), plus a PNG of the last frame.
"""

import os
from typing import Dict, List, Tuple

import cv2
import imageio
import numpy as np
from habitat.utils.visualizations import maps

ROBOT_COLORS = [(230, 25, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180)]  # RGB
OBJ_COLOR = (60, 60, 60)
GOAL_COLOR = (60, 180, 75)
MAP_H = 768  # output frame height; camera tiles share it


class Recorder:
    def __init__(self, host, path: str, every: int = 2, fps: int = 15):
        self.host, self.path, self.every = host, path, max(1, int(every))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                         macro_block_size=8)
        self.frames = 0
        self.last = None
        self.trails: Dict[str, List[Tuple[int, int]]] = {}

        sim = host.env.sim
        self.pf = sim.pathfinder
        active = [r for r in host.robots if host._active.get(r["id"])]
        height = float(host._pose(active[0]["id"])[0][1]) if active else 0.0
        topdown = maps.get_topdown_map(self.pf, height, map_resolution=1024, draw_border=True)
        self.base = maps.colorize_topdown_map(topdown)
        self.grid = topdown.shape[:2]
        self.tile = max(192, min(384, MAP_H // max(1, len(host.robots))))
        # goal locations never move: draw them once, one label per receptacle
        labelled = set()
        for i, o in enumerate(host._objects):
            idx = host._entity_idx.get(f"TARGET_any_targets|{i}")
            if idx is not None:
                label = None if o["goal"] in labelled else o["goal"]
                labelled.add(o["goal"])
                self._mark(self.base, host.entity_pos(idx), GOAL_COLOR, "x", label)

    def _px(self, pos) -> Tuple[int, int]:
        r, c = maps.to_grid(pos[2], pos[0], self.grid, pathfinder=self.pf)
        return int(c), int(r)

    def _mark(self, img, pos, color, shape, label=None):
        x, y = self._px(pos)
        if shape == "x":
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, 20, 3)
        else:
            cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), color, -1)
        if label:
            cv2.putText(img, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def capture(self, step: int) -> None:
        if step % self.every:
            return
        host = self.host
        img = self.base.copy()
        rom = host.env.sim.get_rigid_object_manager()
        for o in host._objects:
            obj = rom.get_object_by_handle(o["handle"])
            if obj is not None:
                self._mark(img, np.array(obj.translation), OBJ_COLOR, "box")

        tiles = []
        for k, r in enumerate(host.robots):
            rid = r["id"]
            color = ROBOT_COLORS[k % len(ROBOT_COLORS)]
            if not host._active.get(rid):  # keep a slot so the frame size never changes
                tile = np.full((self.tile, self.tile, 3), 40, np.uint8)
                cv2.putText(tile, f"{rid}: parked", (8, self.tile // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
                tiles.append(tile)
                continue
            pos, yaw = host._pose(rid)
            trail = self.trails.setdefault(rid, [])
            trail.append(self._px(pos))
            if len(trail) > 1:
                cv2.polylines(img, [np.array(trail, np.int32)], False, color, 3, cv2.LINE_AA)
            x, y = trail[-1]
            cv2.circle(img, (x, y), 13, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 10, color, -1, cv2.LINE_AA)
            cv2.putText(img, rid, (x + 14, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cam = host._last_obs.get(f"agent_{r['agent_idx']}_head_rgb")
            tile = np.zeros((self.tile, self.tile, 3), np.uint8) if cam is None else \
                cv2.resize(np.asarray(cam)[..., :3], (self.tile, self.tile))
            tile = np.ascontiguousarray(tile)
            label = f"{rid}: {host.labels.get(rid, 'idle')}"
            cv2.rectangle(tile, (0, 0), (self.tile, 22), color, -1)
            cv2.putText(tile, label[: self.tile // 8], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)

        h = max(MAP_H, self.tile * max(1, len(tiles)))
        scale = h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), h))
        right = np.concatenate(tiles, axis=0) if tiles else np.zeros((h, self.tile, 3), np.uint8)
        right = np.pad(right, ((0, h - right.shape[0]), (0, 0), (0, 0)))
        frame = np.concatenate([img, right], axis=1)
        header = np.full((28, frame.shape[1], 3), 255, np.uint8)
        cv2.putText(header, f"{host.benchmark} ep {host.env.current_episode.episode_id}  step {step}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        frame = np.concatenate([header, frame], axis=0)
        frame = frame[: frame.shape[0] // 8 * 8, : frame.shape[1] // 8 * 8]
        self.writer.append_data(frame)
        self.last = frame
        self.frames += 1

    def close(self) -> dict:
        self.writer.close()
        png = os.path.splitext(self.path)[0] + "_final.png"
        if self.last is not None:
            imageio.imwrite(png, self.last)
        return {"video": self.path, "final_frame": png, "frames": self.frames}
