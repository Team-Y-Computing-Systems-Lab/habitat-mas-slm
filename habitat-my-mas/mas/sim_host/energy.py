'''
Per robot energy, watts * time(s) = joules
'''

STEP_S = 0.25 # duration of one step in seconds

# power draw in watts per robot (moving, idle)
POWER_W = { 
    "FetchRobot": (40.0, 6.0),
    "StretchRobot": (35.0, 5.0),
    "SpotRobot": (30.0, 4.0),
}
DEFAULT_POWER_W = (30.0, 4.0) # fallback

# 
class EnergyLedger:
    def __init__(self, robots):
        self._type = {r["id"]: r.get("type", "") for r in robots}
        self.zero()
    # set every robot back to 0
    def zero(self) -> None:
        self._joules = {rid: 0.0 for rid in self._type}
    # calc energy used by moving/idle robots for only active robots
    def accumulate(self,  active, acting) -> None:
        for rid in active:
            moving, idle = POWER_W.get(self._type[rid], DEFAULT_POWER_W)
            self._joules[rid] += (moving if rid in acting else idle) * STEP_S
    # json snapshot, per robot joules
    def report(self) -> dict:
        return {
            "robots": {k: round(v, 2) for k, v in self._joules.items()},
            "total_J": round(sum(self._joules.values()), 2),
            "step_s": STEP_S,
        }


