# G1-D SYSU：卷尺实测的学校桌面和标定后的机器人

这是 SYSU 实验室那台 G1-D（Dex1-1 夹爪）在学校绿幕桌前的仿真环境。桌子、黑胶带框、走线口和机器人站位按 2026-11-19 的卷尺实测重建，头相机外参用真机第 34、36 条头部图像标定。仿真里的 `cam_head` 是头部**左眼**，对应真机 `color_0`。

仿真器本身是 [RoboDojo](https://github.com/RoboDojo)（Isaac Sim）。本仓库只放**这张桌子和这台机器人**，叠到一份 RoboDojo 代码上就能用。采集、专家轨迹、布局生成仍用 RoboDojo 里的 `g1d_build_house`。

## 原来在哪里

集群上的完整后端在 `/shared/home/liujingzhi/realtosimtoreal/robodojo-main`：

| 内容 | 集群路径 |
|---|---|
| 环境配置 | `env_cfg/g1d_school_v2.yml`、`env_cfg/g1d_school_v2_collect.yml` |
| 机器人站位和头相机外参 | `env_cfg/robot/g1d_school_v2.yml` |
| 采集相机（头+深度、双腕、第三视角） | `env_cfg/camera/g1d_school_v2_collect_camera.yml` |
| 运行时房间 | `Assets/Room/real_scene4_v2/` |
| 实测场景源（尺寸、重建说明） | `real2sim/scenes/real_scene4_school_v2/`（说明见其中的 `PLAN.md`） |
| 机器人模型 | `Assets/Robots/g1d` → `/shared/opt/robodojo/robodojo-assets/Assets/Robots/g1d` |
| 头相机内参 | `scripts/g1d/calib_toolkit/examples/intrinsics/` |
| 测评前对齐检查 | `/shared/home/liujingzhi/realtosimtoreal/skill/g1d-school-v2-align/` |

本仓库是上面这些文件的拷贝。目录和 RoboDojo 一致，可以直接覆盖进去。

## 仓库里有什么

```
env_cfg/                          环境、机器人站位、采集相机
Assets/Room/real_scene4_v2/       仿真加载的房间（桌面 z = 0.758 m）
Assets/Robots/g1d/                G1-D + Dex1-1 的 URDF 和 USD
real2sim/scenes/real_scene4_school_v2/   卷尺值和重建过程（PLAN.md）
scripts/g1d/calib_toolkit/examples/intrinsics/   四路相机标定内参
align/                            每次测评前的对齐检查
```

站位（`env_cfg/robot/g1d_school_v2.yml`）：

- 根节点 `[-0.032, -0.7572, 0.1055]`，朝向 yaw 89.22°（四元数 wxyz `[0.7119, 0, 0, 0.70228]`）。
- 房间放在 `[-0.015, -0.30, 0]`，桌面近沿在 y = −0.60。根节点在近沿后方 0.157 m，比桌面中心向机器人左侧偏了 3.2 cm（相对最初的卷尺读数），这是用真机画面拟合出来的。
- 升降柱关节为 0。URDF 立柱比真机长约 6 mm，所以根节点比标称高度低 6 mm，腰、手臂和相机才和实测高度一致。轮子会陷进地面 6 mm，不影响操作。
- 头相机安装：俯仰 50.8°、向左偏 2.5°。`camera_override.cam_head` 的位置 `[0.0421, 0.0272, 0.4904]`，朝向 `[0.6727, 0.2453, -0.2291, -0.6594]`。右眼同朝向，y 为 −0.0324。
- 内参（头部左眼）：fx 279.32，fy 279.875，cx 314.997，cy 234.187。畸变 `[0.093616, -0.105871, 0.007352, -0.002534, 0.031005]`，由采集时的相机模型加到图像上，渲染本身仍是针孔。

坐标系：房间原点在桌面中心正下方的地面上，+x 是机器人右手，+y 远离机器人、朝绿幕。

## 接到 RoboDojo 上

在 RoboDojo 仓库根目录执行（已有同名文件会被覆盖）：

```bash
cp -a env_cfg/g1d_school_v2.yml env_cfg/g1d_school_v2_collect.yml   <robodojo>/env_cfg/
cp -a env_cfg/robot/g1d_school_v2.yml                               <robodojo>/env_cfg/robot/
cp -a env_cfg/camera/g1d_school_v2_collect_camera.yml               <robodojo>/env_cfg/camera/
cp -a Assets/Room/real_scene4_v2                                    <robodojo>/Assets/Room/
cp -a Assets/Robots/g1d                                             <robodojo>/Assets/Robots/
cp -a real2sim/scenes/real_scene4_school_v2                         <robodojo>/real2sim/scenes/
cp -a scripts/g1d/calib_toolkit/examples/intrinsics                 <robodojo>/scripts/g1d/calib_toolkit/examples/
```

RoboDojo 里还要有这三处，2026-11-20 之后的 `robodojo-main` 已经有了。没有的话按下面补上，否则头相机会用资产里的默认安装角，和真机对不上。

1. `scripts/g1d/build_asset.py` 的 `SCENES`：

```python
"g1d_school_v2": ("real_scene4_school_v2", "real_scene4_v2", [-0.015, -0.30, 0.0], (0.0, 0.0)),
```

2. `env_cfg/camera/template.py` 里的 `G1D_CALIB_HEAD_LEFT` / `RIGHT` / `LEFT_WRIST` / `RIGHT_WRIST`，以及 `env/camera_manager/camera_pose.py` 的 `REAL_MAP` 把 `g1d_calib_head_left` 等名字指到这些模板。头部左眼模板：

```python
G1D_CALIB_HEAD_LEFT = {  # fx 279.32, fy 279.875, cx 314.997, cy 234.187
    "resolution": (640, 480),
    "focal_length": 10.0,
    "horizontal_aperture": 22.91279,
    "vertical_aperture": 17.15051,
    "horizontal_aperture_offset": 0.17911,
    "vertical_aperture_offset": -0.20770,
    "clipping_range": (0.1, 10.0),
}
```

3. `env/robot_manager/robot_class/x5.py` 在创建相机之后应用 yml 里的 `camera_override`（外参和相机类型都从这份 yml 来，不改共享资产）：

```python
overrides = cfg.get("camera_override", None) or {}
if self.camera and overrides:
    self.camera = [dict(cam) for cam in self.camera]
    for cam in self.camera:
        for key, value in dict(overrides.get(cam.get("name"), None) or {}).items():
            cam[key] = list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value
```

使用哪个配置：

| 用途 | `--env-cfg` |
|---|---|
| 看场景、不采深度 | `g1d_school_v2` |
| 采集和训练数据（头部深度、双腕、第三视角） | `g1d_school_v2_collect` |

任务名是 `g1d_build_house`。不要和旧的 `g1d_school`（另一张桌子、旧站位、旧相机）混用。

## 在这个环境里采数据

房子搭在黑胶带框里，框在环境坐标 x −0.080..0.128、y −0.400..−0.337。四块积木和顺序固定：红色正方体放底层最右，原木色拱形放中间，蓝色正方体放底层最左，橙色三角放在拱形上当屋顶。红色方块用右手，蓝色方块用左手。拱形和屋顶在哪一侧随机。

散件优先放在黑框前面、两只手正前方：横向离房子中心 4–20 cm，y 从 −0.53 到黑框近边 −0.40。大约五分之四的积木在这里，其余在两侧稍远、但双臂仍够得到的地方，不会放到黑框后面，也不会进黑框（外扩 2 cm）。

一批已经采完的数据：seed 4042，1517 条成功，在 ModelScope 的 [g1d_build_house4_v4a_align](https://www.modelscope.cn/datasets/liujzh/G1_D_data/g1d_build_house4_v4a_align)。

在 RoboDojo 根目录生成布局（只要 numpy / scipy / yaml，不用 GPU）：

```bash
python scripts/g1d/build_house4_layouts.py --seed 4042 --count 2400 --split train \
  --config g1d_school_v2 --house-at 0.024 -0.38 0 --tilts 90 60 --no-across --v4a \
  --fixed-colors --keepout -0.080 0.128 -0.400 -0.337 \
  --supply-x 0 0.5 --supply-y -0.59 -0.337 \
  --supply-near 0.04 0.20 -0.53 -0.40 0.8 --extra-lights 0 \
  --workers 16 --rooms school_v2v_0    # 正式采集用 school_v2v_0 .. school_v2v_63
```

`school_v2v_*` 是这间房间的 64 套外观（灯光、桌面、绿幕），由 `scripts/g1d/school_v2_variants.py --prefix school_v2v --count 64 --seed 0` 从 `real_scene4_v2` 生成，不在本仓库里。

采集时环境变量和参数：

```bash
ENV_CFG=g1d_school_v2_collect
TASK=g1d_build_house
HOUSE4_INSTRUCTION=ordered          # 指令按红 → 拱形 → 蓝 → 屋顶来写
COLLECT_ARGS="--camera-model g1d_sysu --action-label executed"
```

- `--camera-model g1d_sysu`：按真机图像统计加曝光、模糊、噪声和 ±4% 白平衡，并加上头部镜头畸变。
- `--action-label executed`：`action/` 是实际发出的指令。开了执行噪声的 episode 另有 `action_expert/`。
- 扰动是 v4a：质量 ×0.7–1.3，摩擦 0.4–0.8，手臂初始姿态 ±3 cm / ±0.1 rad，抓前有概率挪动积木，一半 episode 加 τ = 0.8 s 的执行噪声。
- 每条 hdf5 的 `privileged/` 里有子任务、目标框、指腹接触、动作阶段和 0–1 进度。

5090 集群上对应的脚本是 `realtosimtoreal/v4a/layouts_school_v2.sbatch` 和 `v4a/datagen_school_v2.sbatch`（每卡 3–4 个仿真实例，输出到节点本地盘再同步到 OBS）。换了机器人站位之后必须换一个新的 seed 重新生成布局，旧布局里的可达性是按旧站位算的。

## 每次测评前的对齐

仿真站位是按某一次真机画面标定的。机器人重新摆过、桌子或胶带动过、头被碰过，或者只是换了一天，测评前都跑一次。登录机即可，不用 GPU，大约 10–20 秒。

先录一帧。要头部左目 `colors/000000_color_0.jpg` 和同一段的 `data.json`（有关节角，检查会把手臂挖掉，并检查相机安装角）。只有图片也能跑，只是不挖手臂、不查安装角。

在已经部署这套配置的集群上：

```bash
cd /shared/home/liujingzhi/realtosimtoreal/skill/g1d-school-v2-align
~/envs/usd/bin/python align_check.py \
    --image <episode>/colors/000000_color_0.jpg \
    --data <episode>/data.json \
    --out /tmp/align_today
```

用本仓库里的同一份脚本（路径相对仓库，不依赖上面那个目录）：

```bash
cd align
python align_check.py \
    --image <episode>/colors/000000_color_0.jpg \
    --data <episode>/data.json \
    --out /tmp/align_today
```

`--image` 也可以是 1280×480 的左右拼接图，脚本取左半边。退出码 0 是 PASS，2 是 RECALIBRATE。

- 手臂和夹爪轮廓的平均误差 ≤ 3.5 px 算安装角正常（标定完是 2.6–2.8 px）。更大说明头相机被碰过，加 `--fit-mount` 重标，再改 yml 的 `camera_override`。
- 桌子远边、左右端、黑框四边、走线口四边，最大法向偏差 ≤ 5 px 就 PASS，当天不用改仿真。
- 超过 5 px：测评用的是按这个仿真训练的策略，**挪真机去贴仿真，不要改 yml**。脚本会打印该怎么挪。改了 yml 就和训练数据的视角不一致，还要换 seed 重做布局。
- 叠加图在 `/tmp/align_today/align_overlay.jpg`，数值在 `align_report.json`。

只读抓图（不碰手臂、不改相机）用 `align/grab_from_g1d.sh`。机器人不在登录机上，需要一台在同一网络里的 PC 把机器人的 ssh 反向隧道到登录机 `127.0.0.1:12222`，密钥是 `~/.ssh/id_g1d`，用户 `unitree`。抓到的 `head_left_*.jpg` 就是 `color_0`。

白平衡：真机采集时是自动白平衡。测评前用抓到的图看颜色，手腕绿幕大约 (15, 80, 60)，头部 G 通道大约 100–110。差得明显再把白平衡固定到能还原这个颜色的色温。平时不用改。
