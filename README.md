# RaccoonBot OpenVLA — Assignment 1: Extending the Pipeline

**학번:** 2021741025
**과목:** Physical AI
**Base repo:** https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla

이 과제는 RaccoonBot + OpenVLA 파이프라인을 두 방향으로 확장했습니다.
- **Part 1 (Dataset Extension):** 실린더만 있던 색상 grasp 태스크에 **cube** 물체를 추가하고, instruction을 `grasp the {color} {cylinder|cube}` 로 다양화.
- **Part 2 (Code Improvement, 2개):** **(2-1) 7D→4DOF 액션 매핑 개선**, **(2-2) 에피소드 시각화 개선.**

---

## 추가/수정한 파일

| 파일 | 위치(원본 repo 기준) | 설명 |
|---|---|---|
| `Raccoon_colored_cube.xml` | `Mujoco/` | 실린더 씬을 복사해 4색 물체 geom을 `box`로 바꾼 cube 씬 |
| `generate_mixed_dataset.py` | `Mujoco/` | cylinder/cube 데이터를 각각 생성 후 하나의 데이터셋으로 병합하는 드라이버 |
| `improved_action_mapping.py` | `Mujoco/` | Part 2-1. EMA 스무딩 + gripper 히스테리시스 후처리 + before/after 비교 |
| `visualize_episode.py` | `Mujoco/` | Part 2-2. 프레임 + 7D 액션 궤적 동시 시각화 |
| `raccoon_pick_place_dataset_builder.py` (1줄 수정) | `Mujoco/rlds_dataset_builder/raccoon_pick_place/` | `INTERMEDIATE_ROOT`를 본인 작업 경로로 변경 |

> 원본 `raccoon_grasp_multicolor_scene_dataset.py`는 수정하지 않았습니다. `collect_dataset()`가 `xml_path` / `instruction_template`를 인자로 받기 때문에, cube 확장은 새 XML과 드라이버만으로 구현했습니다.

---

## Part 1. Dataset Extension

**핵심 아이디어:** 한 씬에 두 모양을 섞지 않고, cube 전용 XML을 따로 만들어 `collect_dataset()`을 cylinder용·cube용으로 각각 호출한 뒤 병합. 물체 body 이름(`target_object*`)·색·위치는 동일하게 두어 기존 타겟/grasp 로직을 그대로 재사용. RLDS 변환과 TFDS 빌더는 feature 모양(image 256×256×3, state 8, action 7)이 동일하므로 **수정 없이** 사용.

**결과**
- 생성: cube 40 + cylinder 40 = **80 에피소드** (instruction: `grasp the {red|blue|green|yellow} {cube|cylinder}`)
- TFDS: **train 72 / val 8** (`raccoon_pick_place/1.0.0`)
- 에피소드 시각화: `figures/episode_cube_vis.png` (예: "grasp the green cube", success=True)
- short LoRA 학습: `openvla/openvla-7b` 기반 LoRA(rank 32), **500 step** 완료

---

## Part 2. Code Improvement

### 2-1. 7D→4DOF 액션 매핑 개선 (`improved_action_mapping.py`)

원래 `raccoon_env.execute_delta_action7`는 (1) VLA의 raw delta를 그대로 적용하고 (2) gripper를 0.5 기준으로 이진 처리합니다. 두 가지 문제를 개선했습니다.

- EMA 스무딩: `dx,dy,dz`에 지수이동평균을 적용해 step간 떨림(jitter) 감소.
- gripper 히스테리시스: close(>0.6)/open(<0.4) 임계값을 분리해, gripper 신호가 0.5 근처에서 진동할 때 발생하는 open/close 채터링 제거

**before/after 증거:** `figures/action_mapping_compare.png`
- 위: dx가 raw 대비 부드러워짐(jitter 감소)
- 아래: 이진 처리에서 보이던 gripper 채터링이 히스테리시스로 사라짐

### 2-2. 에피소드 시각화 개선 (`visualize_episode.py`)

기존 Part 1 시각화는 프레임만 나열했습니다. 개선 버전은 **균등 샘플 프레임 + 해당 에피소드의 7D 액션 시계열**을 한 figure에 함께 그려, instruction 대비 모델/데이터의 행동(특히 grasp 시점의 gripper 전환)을 한눈에 확인할 수 있게 했습니다.

**결과:** `figures/episode_report.png` (예: "grasp the red cube", grip이 grasp 직전 0→1로 전환되는 것이 보임)

---

## 재현 방법 (How to Run)

> 모든 경로는 본인 작업 폴더 `/data/2021741025` 기준. Jupyter는 셀마다, 터미널은 `export` 사용

**0) 환경 (세션마다)**
```python
import os
MYDIR = "/data/2021741025"
os.environ["HF_HOME"] = f"{MYDIR}/hf_cache"
os.environ["PYTHONPATH"] = f"{MYDIR}/Raccoonbot_Openvla/openvla"
os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_MODE"] = "disabled"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

**1) 데이터 생성 (cube + cylinder, 각 40개)**
```bash
cd /data/2021741025/Raccoonbot_Openvla/Mujoco
python generate_mixed_dataset.py        # -> raccoon_grasp_mixed_shape/
```

**2) RLDS 변환**
```bash
cd /data/2021741025/Raccoonbot_Openvla/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
  --raw_root /data/2021741025/Raccoonbot_Openvla/Mujoco/raccoon_grasp_mixed_shape \
  --out_root /data/2021741025/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
  --val_ratio 0.1
```

**3) TFDS 빌드**
```bash
cd /data/2021741025/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite --data_dir /data/2021741025/Raccoonbot_Openvla/tensorflow_datasets
```

**4) LoRA 학습 (short, 500 step)**
```bash
cd /data/2021741025/Raccoonbot_Openvla/openvla
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/2021741025/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_pick_place \
  --run_root_dir /data/2021741025/Raccoonbot_Openvla/openvla/openvla-runs \
  --adapter_tmp_dir /data/2021741025/Raccoonbot_Openvla/openvla/openvla-adapter-tmp \
  --lora_rank 32 --batch_size 8 --grad_accumulation_steps 2 \
  --learning_rate 5e-4 --max_steps 500 --save_steps 500 \
  --run_id_note raccoon-cube-final
```

**5) 시각화 / before-after**
```bash
cd /data/2021741025/Raccoonbot_Openvla/Mujoco
python visualize_episode.py        # -> episode_report.png
python improved_action_mapping.py  # -> action_mapping_compare.png
```

---

## 폴더 구조
```
raccoon_submission/
├── README.md
├── report.pdf
├── src/
│   ├── Raccoon_colored_cube.xml
│   ├── generate_mixed_dataset.py
│   ├── improved_action_mapping.py
│   └── visualize_episode.py
├── figures/
│   ├── episode_cube_vis.png          # Part 1 에피소드 시각화
│   ├── action_mapping_compare.png    # Part 2-1 before/after
│   └── episode_report.png            # Part 2-2 시각화 개선
└── logs/
    └── train_500steps.log            # 학습 로그
```

## 비고
- 대용량 데이터셋(`raccoon_grasp_mixed_shape/`, `tensorflow_datasets/`)과 모델 체크포인트(`openvla-runs/`)는 과제 규정에 따라 업로드하지 않았습니다.
- 베이스 모델은 `openvla/openvla-7b` (Hugging Face), 학습은 A100 80GB 1장에서 수행.
