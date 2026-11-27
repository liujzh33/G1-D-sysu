# real_scene4 v2 学校实验室绿幕桌（实测重建）

`real_scene4_school` 的新版本，原目录不动。旧版的桌子和站位来自同事 MuJoCo 孪生的相机拟合（没有卷尺），这一版改用 2026-11-19 的卷尺实测和环境视频重建。env 配置 `g1d_school_v2`，机器人配置 `env_cfg/robot/g1d_school_v2.yml`。

## 来源

`/shared/home/liujingzhi/realtosimtoreal/data/sim_scene/image/`：

| 文件 | 内容 | 拷到 `reference/` |
| :-- | :-- | :-- |
| `环境.mp4` | 21 s 环绕视频，720×1280 | `video_t*s.jpg`（每秒一帧取 9 张）、`video_leg_frame.jpg`、`video_behind_robot.jpg` |
| `机器人高度/` | 升降柱最低、抬高后的特写，卷尺零点在底座黑板上，腰部盒顶读 48.5 | `height_*.jpg` |
| `机器人距离桌面距离/` | 远沿到腰部后沿 81、腰部前沿到远沿 66、俯视卷尺 | `dist_*.jpg`、`top_view_tape.jpg` |

注意：照片里的“桌子前边”指远离机器人、靠绿幕的那条边（远沿）。俯视照片 `top_view_tape.jpg` 里卷尺从机器人前沿拉到绿幕侧桌沿，读数约 66，近沿约在 6 cm 处，可以印证这一点。

## 卷尺值和机器人上的对应位置

腰部和底座在 g1d_v4 网格里的位置。`g1d_v4/meshes` 在本 checkout 里是 LFS 指针；`g1d_aurora_lite_ws/meshes/g1d` 里是同一批文件，sha256 和 `assemble.json` 一致。

| 实物 | 网格 | `AGV_link` 坐标 |
| :-- | :-- | :-- |
| 底盘上立柱的黑色底板上表面 | `LZ_ot_Link` 朝上面 z=−0.438（15.8 cm 方板，比底盘顶高 4 mm） | z = 0.5275 − 0.438 = 0.0895 |
| 腰部盒顶（三角支架下面那个银色方盒） | `Pitching_Link` 朝上面 z=+0.017，盒高 4.3 cm（照片也是 4.3） | z = 0.5805 + 升降量 |
| 腰部后沿 | `Pitching_Link` 方盒 x=−0.0555 | x = −0.055 |
| 腰部前沿（机器人正前方最前点） | `Yaw_Link` 三角支架 x=+0.044 | x = +0.100 |

- **前后**：远沿 = 腰部后沿 + 0.81 = 0.755，或 腰部前沿 + 0.66 = 0.760。两者相差 5 mm（URDF 腰部前后深 15.5 cm，卷尺差 15 cm），取平均 0.757。所以近沿在根节点前 **0.157 m**，机器人最前点离近沿 5.7 cm（俯视照片约 6 cm）。
- **高度**：卷尺测得腰部盒顶在黑板上方 0.485 m，升降柱从最低抬了约 7.5 cm（腰部盒顶到外立柱顶的读数从 7 变成 14.5；两节内柱分别露出约 3.8、3.5 cm）。URDF 在零升降时这段高度已经是 0.491 m，所以 URDF 立柱比真机至少长 6 mm；如果按真机最低位推算（0.485 − 0.075 = 0.41 m），长约 8 cm。仿真没法“抬 7.5 cm 还保持 0.485”，这里优先保证几何：**升降柱为 0，根节点比标称的 0.1115 低 6 mm（0.1055）**，这样腰部、躯干、手臂和相机相对地面和桌面的位置与真机一致：腰部盒顶离地 0.686 m，比桌面低 0.072 m。代价是轮子陷进地面 6 mm（底座固定，不影响仿真）。
- **横向**：卷尺从机器人左手一侧的桌端量到腰部中线（立柱轴线，和根节点横向差 0.2 mm）是 56.5 cm（`stand_565_from_left*.jpg`：从桌子远侧朝机器人拍，卷尺钩子在画面右端，也就是机器人的左手边），即根节点在桌面中心**右侧 1.5 cm**（x = +0.015）。之前从视频估的“左侧 8 cm”不对。
- **偏航**：取 0（底盘正对桌沿）。俯视照片里卷尺和桌沿是垂直的；旧孪生拟合给的是 1.6°，没有采用。

站位换算写在 `build_scene.py` 的 `robot_stand()` 里（读 URDF 关节链），结果见 `usd/build_params.json` 的 `robot`，`checks` 里是反算回卷尺的值。

## 场景

坐标系和 real_scene1–5 一样：原点在桌面中心正下方的地面上，+X 沿桌长方向（机器人右手），+Y 指向绿幕。

| 部件 | 做法 | 依据 |
| :-- | :-- | :-- |
| 桌面 | 1.10 × 0.60 m，离地 0.758 m，板厚 25 mm，6 mm 圆角；浅橡木纹理贴图覆盖整块板（不重复），封边单色 | 卷尺；板厚、圆角目测 |
| 桌架 | 深灰钢管：两端各一个闭合“口”字框（前后立柱 5×5 cm、落地横杆、桌下横梁），后侧一根纵梁，黑色脚垫 | `video_leg_frame.jpg`、`video_t02s.jpg`，尺寸目测 |
| 隐藏式排插（线盒） | 银白色拉丝铝，有反光。铝框 29.6 × 11.8 cm：离机器人右手端 40.4–70.0 cm（中心 x = −0.002；照片里绿幕在画面左、近沿在右，所以钩子那端是机器人右手端），离近沿 43.0–54.8 cm（远侧边离远沿 5.2 cm）；靠机器人一侧是黑色毛刷缝 | 卷尺：`grommet_along_length.jpg`、`grommet_close.jpg`、`grommet_across_width.jpg` |
| 黑色胶带框（放置目标区） | 外沿离机器人右手端 40.7–61.5 cm、离近沿 20.0–26.3 cm（x −0.065…0.143，y −0.100…−0.037）；胶带宽 5 mm 目测 | 卷尺：`target_box_along_length.jpg`、`target_box_across_width.jpg` |
| 绿幕 | U 形：背幕离远沿 0.45 m（中间能站人），宽 2 m；左右侧幕在 x=±1.0，延伸到 y=−0.45。布面高 2.3 m，有随机竖褶和斜褶，下摆铺到地上 12 cm；黑色方管框架，左侧幕前端立柱上有夹子 | `video_t01s/t02s`（右后角）、`video_t09s/t10s`（左侧立柱）；距离目测 |
| 房间 | 浅色 0.6 m 地砖、白墙、天花板（挡住默认 HDR），左墙蓝色窗帘 | `video_t13s`、`video_t19s` |
| 灯光 | 四块吸顶面板灯 6000 K | 没有记录 |

颜色取自视频帧（绿幕亮处 sRGB (55,187,144)，暗处 (27,120,68)，取 (40,160,100)；桌面约 (195,178,155)）。

## 运行

```bash
# 依赖：numpy usd-core trimesh opencv-python-headless shapely mapbox-earcut（本机即可，不需要 Isaac）
python real2sim/scenes/real_scene4_school_v2/build_scene.py   # usd/、real_scene4_v2.usd、usd/build_params.json
python real2sim/scenes/real_scene4_school_v2/preview.py       # preview/{side,top,behind,overview}.jpg（CPU 点云渲染，含机器人）
python scripts/scenes/scene_parts.py real2sim/scenes/real_scene4_school_v2   # usd/parts.json
```

`preview/side.jpg` 是过机器人中心面的剖面，图上标了桌面、腰部盒顶、底板高度，以及按仿真几何反算的两个卷尺值（0.812 / 0.657，卷尺 0.81 / 0.66）。

## 接入（还没做）

`scripts/g1d/build_asset.py` 的 `SCENES` 加一行，Room 放在 (−0.015, −0.30)，近沿落在 y=−0.60、机器人在 x=0：

```python
"g1d_school_v2": ("real_scene4_school_v2", "real_scene4_v2", [-0.015, -0.30, 0.0], (0.0, -0.027)),
```

物体平移 (0, −0.027)：办公室的根节点在近沿后 0.13 m，这里是 0.157 m。之后按 `real2sim/scenes/SKILL.md` 的流程，在 GPU 机上跑 `build_asset.py`、`scripted_demo`、`render_scenes.py`。

## 待补测

1. 地面到底座黑板上表面的高度。现在假定和 URDF 底盘一致（0.201 m，减去根节点下沉后是 0.195 m）。官方底盘高 215 mm，比网格高 18 mm；如果真机确实高这么多，腰部离地要再高约 2 cm。这一项决定躯干离桌面的高差，建议优先量。
2. 背幕到远沿、侧幕到桌子两端的距离，以及绿幕高度。
3. 机器人头部、腕部相机画面和对应关节角，用来做叠图验收。
4. 灯光记录。
