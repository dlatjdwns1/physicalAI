import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

DATA_DIR = "/data/2021741025/Raccoonbot_Openvla/tensorflow_datasets"
LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def visualize(split="train", want="cube", n_frames=5, out_png="episode_report.png"):
    builder = tfds.builder_from_directory(f"{DATA_DIR}/raccoon_pick_place/1.0.0")
    ds = builder.as_dataset(split=split)

    chosen = None
    for ep in ds:
        instr = next(iter(ep["steps"]))["language_instruction"].numpy().decode()
        if want is None or want in instr:
            chosen = ep
            break
    if chosen is None:
        chosen = next(iter(ds))

    steps   = list(chosen["steps"])
    instr   = steps[0]["language_instruction"].numpy().decode()
    success = bool(chosen["episode_metadata"]["success"].numpy())
    actions = np.array([s["action"].numpy() for s in steps])          # (T,7)
    imgs    = [s["observation"]["image"].numpy() for s in steps]
    T = len(steps)
    idxs = np.linspace(0, T - 1, min(n_frames, T)).astype(int)

    fig = plt.figure(figsize=(3 * len(idxs), 6.5))
    gs = fig.add_gridspec(2, len(idxs))
    for col, i in enumerate(idxs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(imgs[i]); ax.set_title(f"t={i}", fontsize=9); ax.axis("off")

    axb = fig.add_subplot(gs[1, :])
    for k in range(actions.shape[1]):
        axb.plot(actions[:, k], label=LABELS[k])
    for i in idxs:
        axb.axvline(i, color="gray", ls=":", alpha=.4)
    axb.set_title("7D action over episode"); axb.set_xlabel("step")
    axb.legend(ncol=7, fontsize=8, loc="upper right")

    fig.suptitle(f"[{split}] {instr}  |  success={success}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print("saved:", out_png, "| steps:", T, "| instruction:", instr)


if __name__ == "__main__":
    visualize(split="train", want="cube")
