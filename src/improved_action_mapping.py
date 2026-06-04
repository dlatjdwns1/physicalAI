import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 서버용
import matplotlib.pyplot as plt


class ActionPostprocessor:
    """7D VLA delta-action을 4DOF 로봇에 적용하기 전 후처리.
    개선 1) EMA로 dx,dy,dz 스무딩 -> step간 떨림(jitter) 감소
    개선 2) gripper 히스테리시스 -> 0.5 근처 open/close 채터링 방지
    """
    def __init__(self, ema_alpha=0.4, grip_close_th=0.6, grip_open_th=0.4,
                 max_delta_xyz=0.01, delta_scale=1.0):
        self.ema_alpha = ema_alpha
        self.grip_close_th = grip_close_th
        self.grip_open_th = grip_open_th
        self.max_delta_xyz = max_delta_xyz
        self.delta_scale = delta_scale
        self.reset()

    def reset(self):
        self._ema = None
        self._grip = 0.0  # 0=open, 1=close

    def process(self, action):
        d = np.clip(np.array(action[:3], float) * self.delta_scale,
                    -self.max_delta_xyz, self.max_delta_xyz)
        # 1) EMA 스무딩
        self._ema = d.copy() if self._ema is None else \
            self.ema_alpha * d + (1 - self.ema_alpha) * self._ema
        sdx, sdy, sdz = self._ema
        # 2) gripper 히스테리시스
        g = float(action[6])
        if g > self.grip_close_th:
            self._grip = 1.0
        elif g < self.grip_open_th:
            self._grip = 0.0
        # 사이값이면 직전 상태 유지
        return float(sdx), float(sdy), float(sdz), self._grip


def _old_map(action, max_delta_xyz=0.01, delta_scale=1.0):
    """개선 전(원래 방식): raw delta clip + 0.5 이진 gripper."""
    d = np.clip(np.array(action[:3], float) * delta_scale, -max_delta_xyz, max_delta_xyz)
    g = 1.0 if action[6] >= 0.5 else 0.0
    return d[0], d[1], d[2], g


def compare_and_plot(out_png="action_mapping_compare.png", seed=0, T=60):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # 합성 입력: 부드러운 목표 + 노이즈(VLA 떨림 모사), gripper는 0.5 근처 진동
    nx = 0.006*np.sin(t/8.0) + rng.normal(0, 0.004, T)
    ny = 0.005*np.cos(t/10.0) + rng.normal(0, 0.004, T)
    nz = -0.003 + rng.normal(0, 0.003, T)
    grip = 0.5 + 0.15*np.sin(t/5.0) + rng.normal(0, 0.08, T)

    post = ActionPostprocessor()
    old_dx, new_dx, old_g, new_g = [], [], [], []
    for i in range(T):
        a = [nx[i], ny[i], nz[i], 0, 0, 0, grip[i]]
        odx, *_, og = _old_map(a)
        ndx, *_, ng = post.process(a)
        old_dx.append(odx); new_dx.append(ndx); old_g.append(og); new_g.append(ng)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].plot(t, old_dx, "o-", alpha=.5, label="before (raw delta)")
    ax[0].plot(t, new_dx, "s-", label="after (EMA smoothed)")
    ax[0].set_title("dx per step — jitter reduced"); ax[0].set_ylabel("dx (m)"); ax[0].legend()
    ax[1].plot(t, grip, ":", color="gray", label="raw gripper signal")
    ax[1].step(t, old_g, where="mid", alpha=.6, label="before (binary @0.5)")
    ax[1].step(t, new_g, where="mid", label="after (hysteresis)")
    ax[1].set_title("gripper command — chatter removed"); ax[1].set_xlabel("step"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=120)
    print("saved:", out_png)


if __name__ == "__main__":
    compare_and_plot()
