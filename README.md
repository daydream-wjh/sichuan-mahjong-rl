# sichuan-mahjong-rl —— 四川麻将强化学习（Suphx 特征 + PPO2）

> 四川麻将（定缺 / 换三张 / 无吃牌 / 27 张牌）的强化学习智能体。
> 沿用上饶麻将版本（`mahjong-rl-trainer`）的框架与模型代码，把规则层换成四川玩法、特征编码改为 `card_preprocess_suphx_sc`。

## 项目简介

这个仓库是**上饶麻将 RL 工程的四川麻将分支**：环境封装、PPO2 训练脚本、ResNet 策略网络、so 规则库的组织方式都沿用上饶版本，真正的改动集中在三处——

1. **规则层换血**：`mahjong.py` 的 `Game_RL` / `MahjongEnv` 改成四川规则，新增**定缺**（`dingque`）、**换三张**（`switch_cards`）、**选花色**（`choose_color`）；
2. **牌张与动作空间收缩**：四川麻将只用万/条/筒三门共 27 张，没有字牌，也**没有吃**，所以 `action_space = Discrete(27)`、`observation_space = (852, 27, 1)`；
3. **精牌机制作废**：`mah_state.RL_state` 里直接写 `self.jing_card = 0  # 精牌 作废`，上饶的「精牌/宝牌」相关逻辑不再参与特征。

计分沿用四川玩法：刮风（补杠）、下雨（暗杠）、查花猪、大叫等，在 `mahjong.py::mah_step` 的 `score_op()` 分支里逐项结算。

## 方法 / 模型

### 环境接口

`mahEnv.MahEnv(gym.Env)`：

```python
four_player_model_name = ["training", "random_partly", "random_partly", "random_partly"]
self.mahjong = MahjongEnv(four_player_model_name)
self.action_space = spaces.Discrete(27)
self.observation_space = spaces.Box(low=0, high=1, dtype=np.int, shape=(852, 27, 1))
```

- `step(action)` 返回 `(state, reward, done, info)`。
- **`reset()` 只返回一个特征张量**（`return mah_state`），不像上饶版那样返回 `(state, mask)` —— 两仓库的 gym 契约不一致，迁移代码时要注意。
- `step()` 里 `mah_action = a` 直接透传动作（上饶版还有一层 `action_set[a]` 映射），说明四川版的动作索引与内部动作已对齐。

### 特征编码

`mah_tool/tool2.py::card_preprocess_suphx_sc()` —— 模仿 Suphx 的多通道平面编码：

```python
features = feature_extract.calculate_sys_suphx_sc(state, state.seat_id, search, global_state, dropout_prob)
features = np.array(features).T
features = np.expand_dims(features, 0)
features = features.transpose([2, 1, 0])   # → C × 27 × 1
```

实现位于 `mah_tool/feature_extract_v10.py`，形状特征（`splitColorFlags` / `splitColor` 按三门花色分离）与向听/进张计算（`mah_tool/sc_hu.py`，支持 `dingque` 定缺花色 `{0: 万, 1: 条, 2: 筒}`）配合。通道数在训练脚本里出现过 **848**（`INFO_ON_LOG["n_fea"]`）与 **852**（tensorboard 目录名 `PPO2_fea852_`）两种写法，应属同一套特征的版本差异。

### 算法与超参

主干 `stable_baselines.PPO2` + 自定义 TF1 ResNet 策略：

- 策略网络：`sbln_learning/custom_policy/tf/resnet{18,34,50,101}_policy_tf.py`（与上饶版同构，`conv2d` 用 `kernel_size=(3,1)`、`channels_first`）、`custom_policy/keras/resnet50_policy*.py`。
- 训练脚本：`sbln_learning/training_files/train_with_raw/train_ppo{1,2}_with_resnet{18,34,50,101}.py`，以及 `train_from_saveModel/` 下四个「从存档续训 + 调参」变体：`adjust_approxkl/`、`adjust_ent/`、`adjust_ent_and_approxkl/`、`only_data_nd/`。
- `sbln_learning/PPO2OnServer.py` 是 PPO2 源码的本地副本（文件首行注释「PPO2源码」），用于在服务器上改算法细节。
- 以 `train_ppo2_with_resnet18.py` 为例的实测超参：

  | 超参 | 值 |
  | --- | --- |
  | `N_STEPS` | 2048 |
  | `NMINIBATCHES` | 2048 // 256 = 8 |
  | `NOPTEPOCHS` | 8 |
  | `TOTAL_TIMESTEPS` | 1e8 |
  | `GAMMA` | 0.925 |
  | `ENT_COEF` | 0.009 |
  | `VF_COEF` | 0.5 |
  | `INIT_LR` | 带 `LinearSchedule` 线性衰减 |
  | `NUMB` / `DATACLIP` | 实验编号（如 `A7_`）/ 数据截断开关 |

### 专家模型接口

`mah_tool/url_recommend.py` 通过 HTTP 访问作者的四川麻将专家服务，共四个端点：

```python
outcard_sichuan_url      = "http://10.0.10.202:8089/sichuanMJ/RLV2/outcard"
operate_sichuan_url      = "http://10.0.10.202:8089/sichuanMJ/RLV2/operate"
switch_cards_sichuan_url = "http://10.0.10.203:8989/sichuan/v2/switch_cards"
choose_color_sichuan_url = "http://10.0.10.203:8989/sichuan/v2/choose_color"
```

后两个正是四川麻将特有的**换三张**与**定缺（选花色）**决策。地址均为私有 IP，外部环境不可达。

### 经验回放与评测

- `experience_replay/exp_feature_extract.py`：把落盘的 json 对局数据（`auc_record.battle_info_record` 写的记录）编码成训练用特征，调用 `calculate_exp_reply_sys_suphx_sc`。
- `auc_record.py`：为每局生成 `round_id`，并按「座位号 / 弃牌 / 副露 / 手牌 / 动作类型（摸 G、弃 d、碰 N、杠 Kkt、胡 A）/ 被操作牌 / 回合数」逐手记录。
- `test_/model_winRate.py`：加载模型跑 `EPOCH = 1000000` 局，按 `RATE_WIN_DISPLAY = 1000` 统计胜率 / 败率 / 流局率并画图，结果写入 `./indicator/win_rate_<model>.txt`。
- `test_/test_env.py`：`TestMahjong(MahjongEnv)` + `TestEnv(model)`，把 `player1` 换成待测 PPO 模型、`player0` 用 `zhiyi_last` 专家，做一对一评测。
- `test_/whiteBox.py`（白盒测试）、`test_/read_npy.py`（读 `obs.npy`）目前基本是注释掉的脚手架。

## 技术栈 / 依赖

仓库**没有** `requirements.txt`，从 import 与脚本 shebang 反推：

| 依赖 | 说明 |
| --- | --- |
| Python | 3.6（shebang `venvs/scmj_ppo2/bin/python3.6`） |
| `tensorflow` | 1.x（用 `tf.compat.v1`、`tf.contrib`，与上饶版 `tensorflow-gpu==1.15.0` 同代） |
| `stable-baselines` | PPO2 / PPO1（上饶版为 `2.10.0`） |
| `gym` | `spaces.Discrete` / `spaces.Box` |
| `numpy`、`matplotlib` | 特征与曲线 |

平台约束同 `mah_tool/接口说明.txt`：so 包为 Ubuntu 16.04 编译、Python 3.6.5、不支持 3.7+。**但本仓库 `.gitignore` 里有 `*.so`**，所以 `mah_tool/so_lib/` 下只剩 Python 源码（`lib_MJ.py`、`shangraoMJ_v5.py`、`sr_xt_ph.py`、`fan_cal.py`、`effec_C.py`，文件头写 `#!/usr/bin/python2.7` 与 `# cython: language_level=2`），需要自行用 cython 编译，或直接用这些纯 Python 版本。

## 目录结构

```
sichuan-mahjong-rl/
├── mahEnv.py                 # gym.Env：动作 27、观测 (852, 27, 1)
├── mah_state.py              # RL_state：四川版（jing_card 作废）
├── mahjong.py                # 规则核心（84 KB / 1550 行）：定缺、换三张、刮风下雨、查花猪
├── mah_player.py             # Player_RL：出牌/操作/换牌/定缺决策
├── auc_record.py             # 对局逐手记录 + round_id
├── experience_replay/
│   └── exp_feature_extract.py    # json 对局 → 模型特征
├── interface/sichuanMJ/      # 四川规则接口层
│   ├── sichuanMJ_v1.py / v2 / v3 / zlc_v2.py
│   ├── lib_MJ*.py            # 三个版本的麻将库（含 zlcV2）
│   └── bak/                  # 早期备份（2 个文件为 Python 2 语法）
├── mah_tool/
│   ├── tool2.py              # 牌值转换 + card_preprocess_suphx_sc
│   ├── feature_extract_v10.py# Suphx 特征（含四川分支）
│   ├── sc_hu.py              # 四川胡牌/向听（定缺花色）
│   ├── shangrao_hu.py        # 上饶胡牌（沿用）
│   ├── so_lib/               # 仅 Python 源码（.so 被 .gitignore 排除）
│   ├── training_recommend/   # 训练期推荐器（sichuanMJ_v2_3_4.py）
│   ├── suphx_extract_features/
│   └── 接口说明.txt
├── sbln_learning/
│   ├── PPO2OnServer.py       # PPO2 源码本地副本
│   ├── custom_policy/tf|keras/   # ResNet18/34/50/101 策略
│   └── training_files/
│       ├── train_with_raw/       # 主训练脚本
│       ├── train_from_saveModel/ # 续训 + 四组调参变体 + train_log
│       ├── pretrain/             # 专家示范预训练
│       ├── progress.json
│       └── ppo2_our_entCoef0.005_...png
├── test_/                    # 评测：model_winRate.py / test_env.py / whiteBox.py / read_npy.py
├── LICENSE                   # Apache License 2.0
└── README.md
```

代码规模：86 个 Python 文件，约 47000 行（不含 `.git`）。

## 快速开始

前提：Linux x86_64 + NVIDIA GPU + Python 3.6。macOS / Windows 跑不了规则 so 库（`mah_tool/接口说明.txt`）。

```bash
# 1. 依赖（仓库未提供 requirements.txt，按上饶版对齐）
pip install tensorflow-gpu==1.15.0 stable-baselines==2.10.0 gym==0.17.3 numpy==1.19.4 matplotlib

# 2. 补齐 so 库（本仓库 .gitignore 排除了 *.so，需自行 cython 编译或改用手写 Python 实现）
ls mah_tool/so_lib/            # 目前只有 .py 源码

# 3. 冒烟测试
python -c "from mahEnv import MahEnv; e=MahEnv(); print(e.reset().shape)"   # 期望 (852, 27, 1) 量级

# 4. 训练（在该脚本目录内执行，脚本里有 sys.path.append('../../../')）
cd sbln_learning/training_files/train_with_raw
python train_ppo2_with_resnet18.py

# 5. 曲线
tensorboard --logdir sbln_learning/training_files/train_with_raw/tensorboardLog
```

## 已知限制 / TODO

**已知问题（均为实测确认）**

1. **模型与存档均不在仓库内**：README 声称「沿用上饶麻将版本及模型」，但仓库里没有任何 `.pth` / `.ckpt` / `save_model` 目录，`test_/model_winRate.py` 里引用的存档路径来自作者另一台机器（`/media/lei/0EAA16130EAA1613/...`）。
2. `test_/model_winRate.py` 的两处导入在本仓库中不存在：`from sbln_learning.resnetPolicy import ResNetPolicy`、`from sbln_learning.self_policy import selfPolicy`（`sbln_learning/` 顶层只有 `__init__.py` 和 `PPO2OnServer.py`），脚本无法运行。
3. `.gitignore` 排除了 `*.so`，`mah_tool/so_lib/` 只剩 cython/Python 2.7 风格的源码，未经编译时 `from mah_tool.so_lib.shangraoMJ_v5 import recommend_op` 一类调用会失败。
4. `interface/sichuanMJ/bak/SCMJ.py` 与 `bak/sichuanMJ_v1.py` 是 Python 2 语法，在 Python 3 下无法编译（`py_compile` 直接报错），属死代码。
5. 专家接口写死内网 IP（`10.0.10.202:8089`、`10.0.10.203:8989`），无法外部复现。
6. 特征维度表述不统一：`mahEnv` 声明 `852`，`INFO_ON_LOG` 写 `n_fea: 848`，tensorboard 目录名用 `fea852`。
7. `reset()` 只返回特征张量（上饶版返回 `(state, mask)`），`step()` 返回 4 元组 —— 两个仓库的 gym 契约不一致。
8. 仓库命名不一致：README 标题写 `sichuang_mj_rl_v2_suphx`，`.idea/*.iml` 写 `sichuang_mj_rl_v3_suphx-master`。
9. 无 `requirements.txt`；`.idea/` 已入库；git 只有一个 `Default Changelist` 提交，无历史可追溯。

**TODO**

- [ ] 补 `requirements.txt`，写清 tensorflow / stable-baselines / gym 版本。
- [ ] 提供 `so_lib` 的可编译方案（cython 构建脚本），或在 `mahEnv` 里支持纯 Python 规则回退。
- [ ] 修 `test_/model_winRate.py` 的导入与存档路径，把评测入口做成可配置参数。
- [ ] 统一 `n_fea`（848 / 852）与 `reset()` 返回值契约，并在 README 与代码里对齐。
- [ ] 清理 `interface/sichuanMJ/bak/` 下的 Python 2 死代码。
- [ ] 把专家接口地址抽成配置项，并提供本地随机对手作为默认。
- [ ] 补训练结果记录（`train_log/SAVETIME-*` 已有日志，但仓库未汇总任何对局强度结论）。

## License

本仓库包含 [Apache License 2.0](LICENSE) 许可证文件。
