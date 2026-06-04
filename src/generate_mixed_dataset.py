import os, glob, json, shutil
from raccoon_grasp_multicolor_scene_dataset import collect_dataset

N_CYL = 40     # 연습용. 잘 되면 늘려 (예: 40)
N_CUBE = 40

COMMON = dict(
    use_viewer=False, camera_name="front_view", initial_settle_seconds=0.1,
    object_x_range=(-0.10, 0.10), object_y_range=(0.16, 0.25), min_object_distance=0.035,
)

# 1) 실린더 데이터
collect_dataset(xml_path="Raccoon_colored_cylinder.xml", dataset_root="_part_cyl",
                num_episodes=N_CYL, instruction_template="grasp the {color} cylinder",
                seed=0, **COMMON)

# 2) 큐브 데이터
collect_dataset(xml_path="Raccoon_colored_cube.xml", dataset_root="_part_cube",
                num_episodes=N_CUBE, instruction_template="grasp the {color} cube",
                seed=1, **COMMON)

# 3) 두 결과를 하나로 병합 (에피소드 번호 새로 부여)
FINAL = "raccoon_grasp_mixed_shape"
if os.path.exists(FINAL):
    shutil.rmtree(FINAL)
os.makedirs(FINAL)
idx = 0
for part in ["_part_cyl", "_part_cube"]:
    for ep in sorted(glob.glob(os.path.join(part, "episode_*"))):
        dst = os.path.join(FINAL, f"episode_{idx:06d}")
        shutil.copytree(ep, dst)
        mp = os.path.join(dst, "meta.json")
        try:
            with open(mp) as f: m = json.load(f)
            m["episode_id"] = idx
            with open(mp, "w") as f: json.dump(m, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("meta update skip:", dst, e)
        idx += 1
print(f"\n[merge] 총 {idx} 에피소드 -> {FINAL}")
