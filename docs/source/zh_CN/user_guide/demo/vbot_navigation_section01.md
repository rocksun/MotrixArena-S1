# VBot Section01 越障导航实验

本实验训练 VBot 从 `Y=-2.4 m` 附近出发，依次通过 7 个路径点，穿越崎岖地面、落差和上坡，到达 `(0.0, 7.8)` 附近的平台。环境名为 `vbot_navigation_section01`，环境实现位于 `motrix_envs/src/motrix_envs/navigation/vbot_section01_np.py`，场景文件为 `scene_section01.xml`。

本文以仓库当前版本为准。奖励总和为 `Σ(奖励项原始值 × scale)`，最后会被裁剪到 `[-100, 1000]`。因此，正权重是鼓励项，负权重是惩罚项；权重大小只有结合对应原始值的数值范围才有意义。

本实验还依赖与代码匹配的 `uv.lock`。早期使用旧锁文件训练时策略没有出现有效进展；更换为本实验对应的 `uv.lock` 并同步依赖环境后，训练才开始产生有效结果。因此，`uv.lock` 是实验配置的一部分：它锁定 Python 包及其传递依赖的精确版本，避免同一份环境与奖励代码因底层依赖版本不同而出现不可复现的训练行为。

## 1. 环境准备

Section01 的基础接入包括以下两个连续步骤：

| 步骤 | 作用 |
| --- | --- |
| 注册导航包 | 在 `motrix_envs/__init__.py` 中导入 `navigation` 包，使导航环境能够在包加载时注册。 |
| 接入 Section01 | 新增 navigation 配置、Section01 NumPy 环境、场景接入，以及 `vbot_navigation_section01` 的 PPO 配置。 |

开始实验前，确认以下文件均已存在，并且环境注册名保持为 `vbot_navigation_section01`：

- `motrix_envs/src/motrix_envs/__init__.py`
- `motrix_envs/src/motrix_envs/navigation/cfg.py`
- `motrix_envs/src/motrix_envs/navigation/vbot_section01_np.py`
- `motrix_envs/src/motrix_envs/navigation/xmls/scene_section01.xml`
- `motrix_rl/src/motrix_rl/cfgs.py`

随后在仓库根目录执行命令。依赖由项目的 `uv` 环境管理；`--no-sync` 表示使用已安装的锁定依赖，不在本次运行中同步或改动环境。首次使用本实验的锁文件、或更新了 `uv.lock` 后，必须先执行一次 `uv sync`，使虚拟环境实际切换到锁定版本；仅替换 `uv.lock` 再直接执行带 `--no-sync` 的命令，不会更新既有环境。

环境的关键设定如下：

- 回合长度为 40 秒 / 4000 个控制步，控制周期 `ctrl_dt=0.01 s`。
- 初始位置为 `(0.0, -2.4, 0.5)`，平面位置随机扰动为 `±0.5 m`，初始偏航随机范围为 `[-0.15, 0.15] rad`。
- 使用机体坐标系速度指令。路径点为 `(0,-0.6)`、`(0,1.2)`、`(0,2.25)`、`(0,4.0)`、`(0,6.0)`、`(0,7.0)` 和 `(0,7.8)`；距当前路径点 `0.45 m` 内即切换到下一点，不需要停留。
- 正常区域速度上限为 `vx∈[-0.6,0.9] m/s`、`vy∈[-0.35,0.35] m/s`、`ωz∈[-1,1] rad/s`；崎岖区域进一步限制为 `vx∈[-0.25,0.45] m/s`、`vy∈[-0.25,0.25] m/s`、`ωz∈[-0.8,0.8] rad/s`。
- 动作为 12 个关节的目标位置增量，`action_scale=0.5`，PD 参数为 `stiffness=80`、`damping=6`。

## 2. RewardConfig 参数说明

配置位置：`motrix_envs/src/motrix_envs/navigation/cfg.py` 中 `@registry.envcfg("vbot_navigation_section01")` 的内部 `RewardConfig`。奖励计算位置：`motrix_envs/src/motrix_envs/navigation/vbot_section01_np.py::_compute_reward()`。

### 2.1 导航与到达奖励

| 参数 | 当前值 | 原始项 / 作用 | 调整建议 |
| --- | ---: | --- | --- |
| `tracking_lin_vel` | `1.2` | `exp(-线速度误差平方 / tracking_sigma)`，匹配机体坐标系线速度指令。 | 走得慢或不按指令走时提高；过度追速导致跌倒时降低。 |
| `tracking_ang_vel` | `0.54` | `exp(-(ωz_cmd-ωz)² / tracking_sigma)`，匹配偏航角速度。 | 转向不足时提高；蛇形或高频转向时降低。 |
| `tracking_goal_vel` | `2.0` | 沿“机器人到当前路径点”方向的速度，与前进指令归一化后的投影。 | 是直接朝目标推进的重要信号；提高会更积极地对准路径点，但过大可能牺牲姿态。 |
| `tracking_yaw` | `0.55` | `exp(-目标方位角与当前偏航角的绝对误差)`。 | 对不准路径或横向走时提高；频繁扭头、步态受扰时降低。 |
| `forward_progress` | `1.0` | 沿当前速度指令方向的正向速度，截断至 `1.5 m/s`。 | 用于抑制原地踏步；过高会鼓励盲目冲刺。 |
| `target_progress` | `1.0` | 当前路径点距离的单步减少量，裁剪到 `[-0.2, 0.2]`。 | 更强调真实接近路径点时提高；若在转弯/复杂区抖动，可适当降低。 |
| `reach_goal` | `8.0` | 本控制步进入某个路径点的阈值时给一次奖励。 | 路径点切换不积极时提高；过大可能在路径点附近反复投机，应同时检查切换逻辑。 |
| `reach_all_goal` | `300.0` | 到达最后一个路径点并完成整条路线时给奖励。 | 稀疏的最终成功信号，建议保持显著高于单点奖励；过高会使回报方差增大。 |

`tracking_sigma=0.2` 是两个指数速度跟踪项的误差尺度：减小会让奖励对误差更严格、学习信号更尖锐；增大会让容忍范围更宽。它不是权重。 

### 2.2 稳定性、平滑性与接触约束

| 参数 | 当前值 | 原始项 / 作用 | 调整建议 |
| --- | ---: | --- | --- |
| `termination` | `-10.0` | 回合终止标志。 | 用于明确区分可恢复的低回报与失败；若策略冒险撞击，可增大绝对值。 |
| `lin_vel_z` | `-2.0` | 机体竖直速度平方。 | 抑制弹跳和落地冲击；过强会限制跨越动作。 |
| `ang_vel_xy` | `-0.05` | 横滚、俯仰角速度平方和。 | 抑制翻滚/点头抖动；需要更强爬坡姿态变化时不宜过大。 |
| `orientation` | `-0.5` | 重力在机体 `xy` 平面的投影平方和。 | 保持机身竖直；过强会与上坡前倾目标冲突。 |
| `torques` | `-9e-06` | 各关节力矩平方和。 | 限制大力矩；如无法跨障，可先减小绝对值而非直接提高动作幅度。 |
| `dof_vel` | `-1e-4` | 各关节速度平方和。 | 抑制过快甩腿；过强会削弱抬腿能力。 |
| `dof_acc` | `-2.5e-7` | 关节速度差除以控制周期后的平方和。 | 抑制突然加减速；过强会导致动作迟钝。 |
| `action_rate` | `-0.01` | 当前动作与上一步动作的差平方和。 | 控制策略输出平滑度；振荡时增大绝对值，反应过慢时减小。 |
| `dof_pos_limits` | `-0.45` | 超过软关节上下限的偏差和。 | 防止关节打到极限；出现极限姿态时增大绝对值。 |
| `undesired_contacts` | `-1.0` | 非足部碰撞体接触数。 | 约束腿/机身擦碰；若环境接触噪声较大，先检查碰撞体配置再调权重。 |
| `base_contact` | `-10.0` | 基座接触标志。 | 直接惩罚趴地或机身撞地；与终止惩罚共同约束失败。 |
| `anti_stall` | `-0.8` | 有移动命令时，指令速度与实际速度之差的正部分。 | 抑制“站着拿奖励”；若策略过于急躁，可减小绝对值。 |
| `feet_air_time` | `1.0` | 落脚瞬间的 `(腾空时长 - feet_air_time_target)` 之和，仅有移动指令时生效。 | 鼓励有效摆腿和步幅；过大易出现过高或过长摆腿。 |
| `energy` | `0.0` | 各关节“力矩 × 关节速度”绝对值之和，作为机械功率代理。 | 当前被关闭，避免能耗约束压制后腿迈步。步态已稳定后可设为小负值改善能效。 |

`feet_air_time_target=0.55 s` 是足端腾空时间奖励的基准。提高会鼓励更长摆腿，降低会鼓励更快、更紧凑的步态。

### 2.3 步态与分地形塑形奖励

| 参数 | 当前值 | 原始项 / 生效区域 | 调整建议 |
| --- | ---: | --- | --- |
| `gait_symmetry` | `0.22` | 对角腿 `(FR,RL)`、`(FL,RR)` 接触状态的一致度，仅移动时生效。 | 用于形成对角小跑；过高可能压制跨障所需的非对称落脚。 |
| `per_leg_swing` | `3.0` | 四条腿中最短的当前腾空时长，截断至 `0.55 s`。 | 重点防止后腿不迈步；若悬腿时间过长或步态僵硬，降低。 |
| `swing_foot_height` | `1.5` | 仅崎岖区；奖励未触地脚相对最低脚的高度接近 `0.12 m`。 | 崎岖区绊脚时提高；抬腿过高、能耗上升时降低。 |
| `drop_leg_catchup` | `4.0` | 仅 `1.3 < Y ≤ 1.8` 的落差段；奖励最慢腿腾空。 | 后腿卡在落差边缘时提高；过大可能导致在边缘过度抬腿。 |
| `drop_pitch` | `-2.0` | 仅落差段；机体前后倾代理 `gravity_x²`。 | 抑制落差处过度前倾；若正常下台仍被限制，可减小绝对值。 |
| `slope_leg_drive` | `3.0` | 仅 `1.8 ≤ Y < 6.9`；奖励最慢腿腾空。 | 促进所有腿在过渡坎和坡道持续迈步；上坡缺动力时提高。 |
| `slope_front_drive` | `3.0` | 仅 `1.8 ≤ Y < 6.9`；奖励两条前腿中较短的腾空时长。 | 防止前腿上坡消极；若前腿抬得过高、失去支撑则降低。 |
| `slope_pitch` | `-1.5` | 仅 `2.0 ≤ Y < 6.9`；惩罚 `gravity_x` 偏离 `0.13`（约 7.5° 温和前倾）。 | 上坡前栽时增大绝对值；无法上坡或明显后仰时检查目标/减小绝对值。 |
| `slope_hip` | `-1.0` | 仅 `2.0 ≤ Y < 6.9`；四个髋关节偏离默认角超过 `0.2 rad` 的平方和。 | 抑制前腿髋外张、腿软；若阻碍正常跨步，可减小绝对值或放宽阈值。 |

调奖励时优先一次只改变一类信号，并观察 TensorBoard 的 `Reward/*` 分项及 `metrics/*`。若成功率下降，先区分是“走不到下一路径点”（导航项问题）、“特定地形摔倒”（区域塑形/接触项问题），还是“全程抖动”（动作和平滑正则问题），避免同时改多项而无法归因。

## 3. PPO 配置与调优

配置位置：`motrix_rl/src/motrix_rl/cfgs.py` 中 `@rlcfg("vbot_navigation_section01")` 的 `VBotNavigationSection01PPOConfig`。使用 SKRL PPO，观测和值函数均会采用运行时标准化。

### 3.1 本实验覆盖的参数

| 参数 | 当前值 | 说明与调优方向 |
| --- | ---: | --- |
| `seed` | `42` | 固定随机种子，便于比较奖励或 PPO 改动；验证泛化时应换多个种子复跑。 |
| `num_envs` | `4096` | 训练并行环境数。每次 rollout 采样 `4096 × 48 = 196,608` 条转移；显存不足时先降低此值。 |
| `play_num_envs` | `1` | 回放时的并行环境数，设为 1 便于观察单个机器人。 |
| `max_env_steps` | `1024 × 60,000 × 3 = 184,320,000` | 训练总环境步数预算。按当前实现会折算为以检查点间隔对齐的批次数。 |
| `check_point_interval` | `1000` | 每 1000 次批量迭代写日志并保存检查点。增大可减少 I/O，减小可更密集地保留中间模型。 |
| `learning_rate` | `3e-4` | PPO 初始学习率。回报剧烈波动或 KL 偏大时降低；长期无提升时小幅提高。 |
| `learning_rate_scheduler_kl_threshold` | `0.008` | KL 自适应学习率调度的目标阈值。阈值小会更保守，阈值大允许单次策略更新更激进。 |
| `rollouts` | `48` | 每次更新前每个环境采样的控制步数；增大提高样本量与吞吐，但降低更新频率。 |
| `learning_epochs` | `6` | 每批 rollout 重复优化次数。过多可能过拟合当前批并使 KL 上升，过少则样本利用不足。 |
| `mini_batches` | `32` | 将一个 rollout 批次拆成 32 个小批。当前每个小批约 6144 条样本；减少可得到更大的小批和更平滑梯度。 |
| `discount_factor` | `0.99` | 折扣因子 `γ`，保留长期到达终点奖励的价值。降低会更短视。 |
| `lambda_param` | `0.95` | GAE 参数 `λ`，在优势估计的偏差与方差间折中。更高更重视长时序但方差更大。 |
| `grad_norm_clip` | `1.0` | 梯度范数裁剪上限，防止异常更新。频繁裁剪可降低学习率或减少 epoch。 |
| `entropy_loss_scale` | `0.0` | 熵正则权重，当前不额外鼓励探索。早期陷入局部步态时可尝试很小正值，再在后期减回 0。 |
| `value_loss_scale` | `2.0` | 值函数损失权重。价值估计跟不上回报时可提高；策略更新被价值项主导时降低。 |
| `time_limit_bootstrap` | `True` | 因 40 秒时间上限结束时，用价值估计做 bootstrap，避免把时间截断误判为失败。 |
| `ratio_clip` | `0.2` | PPO 策略概率比裁剪范围。增大允许更激进更新，降低会更稳定但学习更慢。 |
| `value_clip` | `0.2` | 值函数更新的裁剪范围。值函数震荡时可降低。 |
| `clip_predicted_values` | `True` | 启用预测价值裁剪，与 `value_clip` 配合减少值函数突变。 |
| `policy_hidden_layer_sizes` | `(512, 256, 128)` | Actor 的三层 MLP 宽度。容量不足时可增大；显存或吞吐受限时减小。 |
| `value_hidden_layer_sizes` | `(512, 256, 128)` | Critic 的三层 MLP 宽度。若价值损失长期偏大，可优先增大 Critic。 |

本配置未覆盖、因此沿用 `PPOCfg` 默认值的参数包括：`share_policy_value_features=True`（Torch 后端且网络结构相同时共享特征）、`random_timesteps=0`、`learning_starts=0`、`kl_threshold=0` 和 `rewards_shaper_scale=1.0`。当前命令默认优先选择可用的 JAX 后端；需要固定后端时可附加 `--train-backend jax` 或 `--train-backend torch`。

### 3.2 建议的调优顺序

1. 先保持 PPO 参数不变，仅确认 `goal_success_rate_window200`、`course_distance` 和各种 `reset_*` 指标是否符合预期。
2. 若主要失败原因为 `reset_base_contact`，先针对发生区段修改对应奖励项，例如落差段的 `drop_*` 或坡道的 `slope_*`；不要首先提高学习率。
3. 若奖励、成功率和 KL 同时剧烈震荡，将 `learning_rate` 降至 `1e-4` 或减少 `learning_epochs`；若学习稳定但进展很慢，再小幅提高学习率或 rollout。
4. 在单一 `seed` 得到候选配置后，至少用多个种子复现，并在 `play_num_envs=1` 下逐段观察路线，确认不是偶然成功。

### 3.3 训练与回放

训练前先确认当前工作区的 `uv.lock` 是本实验对应版本。它固定了训练所使用的 Python 依赖版本，也是“更换锁文件后训练开始有效”的必要条件之一。若锁文件刚被替换或本地虚拟环境尚未同步，请先执行：

```bash
uv sync
```

同步完成后，使用以下命令启动训练；此时 `--no-sync` 可避免在每次运行时重复解析或修改依赖环境：

```bash
uv run --no-sync scripts/train.py --env vbot_navigation_section01
```

训练日志和检查点写入 `runs/vbot_navigation_section01/`。回放脚本默认选取最近一次运行中的 `best_agent.*`；没有该文件时选择步数最大的 `agent_*.pt` 或 `agent_*.pickle`：

```bash
uv run --no-sync scripts/play.py --env vbot_navigation_section01
```

若要指定模型或在回放时打印每次重置原因，可使用：

```bash
uv run --no-sync scripts/play.py --env vbot_navigation_section01 --policy <检查点路径> --num-envs 1 --log-resets
```

### 3.4 辅助分析与奖励权重扫描脚本

| 脚本 | 作用 | 适用场景 |
| --- | --- | --- |
| `scripts/report_section01_max_distance.py` | 读取 `runs/vbot_navigation_section01/` 下各次训练的 TensorBoard 事件文件，按 `metrics / course_distance (max)` 降序排列。该指标是相对各回合随机起点的最大 Y 向前进距离，单位为米。 | 快速比较多次训练的最远推进距离，定位有希望的权重组合或检查点。 |
| `scripts/sweep_reward_scales.py` | 对指定奖励项逐一施加相对变化，并为每次试验创建独立训练名称。每次训练只改变一个奖励项，脚本在正常结束、失败或 `Ctrl+C` 后都会尝试恢复 `cfg.py` 原文。 | 用可归因的消融实验验证某个奖励项的权重是否应提高或降低。 |

训练完成后，可生成最远距离排名：

```bash
uv run --no-sync scripts/report_section01_max_distance.py
```

该脚本还会显示每个运行中 `best_agent.pt` 对应的检查点步数，以及该步的平均课程距离；可用 `--csv <文件路径>` 导出表格，或用 `--runs-dir <目录>` 分析其他运行目录。已完成且超过 24 小时的运行会在 `runs/vbot_navigation_section01/.section01_max_distance_cache.json` 中缓存读取结果，以减少重复解析事件文件的时间。

以下示例仅扫描 `tracking_lin_vel` 和 `feet_air_time`：对每个参数各训练基准值、降低 10% 和提高 10% 三组实验。

```bash
uv run --no-sync scripts/sweep_reward_scales.py 0 -0.1 0.1 --parameters tracking_lin_vel,feet_air_time --train-arg=--seed=42
```

位置参数是相对变化率：`0.1` 表示原始权重乘以 `1.1`，`-0.1` 表示乘以 `0.9`，且数值必须大于 `-1`。`--train-arg` 可重复传入并会透传给 `train.py`。扫描前不要手动修改 `cfg.py`；脚本会检测外部改动并拒绝覆盖。当前脚本支持扫描其内置列表中的奖励项；`base_contact`、`slope_leg_drive`、`slope_pitch`、`slope_hip` 与 `slope_front_drive` 不在该列表内，如需扫描应先明确把它们加入脚本的 `DEFAULT_REWARDS`。

## 4. 结果展示

> 视频占位：请在录制完成后替换为 Section01 策略从起点穿越崎岖区、落差和坡道，最终到达 `Y=7.8 m` 平台的回放视频。

建议视频同时展示以下信息：完整路线、关键落差处的后腿跟随、上坡时的前后腿协同，以及最终到达路径点后的终止状态。可在此处嵌入视频：

```{video} /_static/videos/vbot_section01.mp4
:width: 100%
```

录制文件放入 `docs/source/_static/videos/vbot_section01.mp4` 后，取消或保留上述指令即可在 Sphinx 文档中展示。
