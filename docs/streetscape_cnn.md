# Krea 2 LoRA＋冻结CNN共融辅助损失

交付状态：**代码已实现，运行验证待Linux服务器完成。** 本地不安装完整训练依赖、不下载生成底模、不运行单元测试、CUDA烟测或训练。语法检查不代表功能验证，以下命令仅供服务器阶段执行。

## 1. 改造范围与固定合同

- 官方上游：`ostris/ai-toolkit`；Fork：`qinghui316/ai-toolkit`。
- 基础提交：`ecee894ed2b1f3716d9d7326693061ec1a3105bb`；研究分支：`research/krea2-cnn`。
- 独立进程：`streetscape_cnn_trainer`，继承上游 `SDTrainer`。普通 `sd_trainer` 未修改。
- CNN：已有400张全量开发集最终ResNet18 shared-delta模型。权重、12项定义、目标标准化统计和六维共融公式全部复用；来源摘要见扩展目录的 `provenance.json`。
- 本轮不决定正式LoRA训练轮数、步数、学习率、rank、CNN损失权重或启用区间；模板里的 `null` 必须显式填写。模板不是预注册训练配置。
- 不复制/删除现有图片，不替LoRA分配训练、验证或测试角色，不改标签，不追加安全感专项目标。

v1启用路径只支持标准Krea 2 RAW文本生成LoRA。使用 `model.arch: krea2`、`model.name_or_path: krea/Krea-2-Raw`；可提前把RAW下载到Hugging Face缓存。启用时不接受仅凭文件名声明的本地RAW/Turbo文件，不支持编辑模式、其他额外预测变换、正则化数据或加权/带掩码数据。这样避免在未知语义下套用速度还原公式。

该研究扩展当前按单进程、单GPU验收设计。多卡未验收。标准框架仍保留原能力，不以本扩展的状态推断其他训练器。

## 2. 数学路径、开关与日志

先保留上游基础损失，再计算：

`z_hat = noisy_latents - actual_sigma * predicted_velocity`

实际sigma从当前调度器表按训练时间步查得，不把 `timestep / 1000` 当成通用换算，也不调用会改变调度器状态的采样步进接口。Krea 2原生VAE负责潜变量反归一化及解码；RGB张量转换到0—1，再可微缩放至224×336，使用原ImageNet均值与标准差。

冻结CNN和VAE的参数。CNN保持eval模式，Dropout和BatchNorm行为冻结。评分分支保持计算图，不经过PIL、NumPy、磁盘图片或 `no_grad()`。只有日志在损失构造完成后detach。

CNN输出按检查点中的均值/差值统计还原12项评分，裁剪到1—10，反转无聊和压抑两个负向维度。各维共融为 `0.5 × 两组均值 + 0.5 × 两组较低值`，六维取均值。使用原hard minimum，不新增soft-min。

`L_CNN = sum((10 - inclusive_all6) / 9 for eligible samples) / full_microbatch_size`

`L_total = L_base + weight * L_CNN`

这是一步干净图像估计上的训练代理目标，不是完整采样结果的奖励。不能据代码实现宣称生成质量或论文指标提升。

配置字段：

| 字段 | 规则 |
|---|---|
| `enabled` | 默认false；关闭时直接返回上游基础损失，不加载CNN、不增加解码 |
| `checkpoint`、`manifest` | 启用时必填；另将原始config.json放到manifest同目录 |
| `weight` | 必填，有限且大于0 |
| `start_step` | 必填，非负，必须在本次总步数以内 |
| `sigma_min`、`sigma_max` | 必填，`0 < min < max < 1`；0为干净端、1为纯噪声端 |
| `every_n_steps` | 默认1；相对start_step计算执行间隔 |

step沿用上游全局步数语义，不另设epoch/更新计数器。未到启用步、间隔不满足或无合格sigma样本时跳过评分；不按合格样本数量或执行间隔放大损失。

日志包括基础、未加权/加权辅助损失、合格样本数、完整微批次大小、裁剪比例、原始评分均值/极值、原始/裁剪共融、青年/老年质量、差距与最低安全感。损失按原生微批次调用平均，样本数累加，评分按合格样本数加权平均，极值取全部合格样本的极值，不把最后一个微批次冒充整个累积过程。

## 3. 缓存、卸载和恢复

图像潜变量与描述embedding缓存仍由上游负责。文本编码器完成缓存后可按上游逻辑卸载。辅助评分使用新预测，不能缓存其VAE解码或CNN分数。

如果VAE已在CPU上，首次需要评分时移至当前计算设备，并保留到本次原生训练调用的全部反向传播结束；随后恢复原CPU位置。冻结VAE不意味着禁用输入梯度。启用该分支一定会增加显存和计算，不能沿用普通LoRA的显存估计。

使用原生 `gradient_accumulation: N` 的微批次列表路径。v1拒绝旧式 `gradient_accumulation_steps > 1` 跨迭代路径，避免在未保存的累积梯度中断点恢复；不重写上游梯度累积或优化器逻辑。

每个研究run保存内容身份：完整训练配置、相对数据文件清单及哈希、描述哈希、RAW/TE/VAE实际权重及配置文件、冻结CNN、源码版本和引导参数。HF资产在首次运行时解析到本地缓存，恢复时复用之前的snapshot revision；不接受默默更新的main。迁移主机不要求相同绝对路径，但文件内容和相对布局必须一致。

原冻结CNN manifest中的Windows绝对路径只用于历史记录；加载时使用明确传入的checkpoint和manifest，并核对同目录config.json的哈希。不得重新生成或编辑冻结manifest。

研究目录额外产物：

- `streetscape_identity.json`：内容身份；配置/数据变化需新run名。
- `streetscape_latest.json`：最新原生LoRA＋optimizer.pt的配对哈希和原生step。
- LoRA safetensors元数据：研究身份及引导参数。
- `streetscape_environment.json`：实际服务器软件/硬件版本，不作为验收通过标志。
- `streetscape_gradient_check.json`：仅服务器显式开启梯度验收后产生。

恢复使用原生最新进度续训；身份保护会拒绝跨目标、跨CNN、变化的数据、配置、底模及不完整配对。加载优化器后的状态会与文件核对，防止上游打印加载错误后静默继续。跨主机时根据已绑定的最新文件恢复，避免依赖ctime重新猜测。

不承诺任意历史断点、逐位一致、随机数/数据读取位置或EMA完整状态恢复。模板关闭EMA；改动该设置需新身份。初始化后尚无完整原生检查点的失败目录不能作为恢复起点，应保留故障现场并换新run名，不自动清理目录。

## 4. 数据准备注意事项

本次发现打标器目录包含960份PNG、480份TXT。只读审计会识别同名图文对及剩余图片，不自动删除、复制或分组：

2026-10-08只读审计确认480对、八类各60对；480张额外图片未配对。候选图文内容身份为 `9a132a427fa1daeed6de886bcbb516689293471c85b129dca22a6758343dda11`。这是整理状态，不是LoRA训练划分或模型验收结果。

```powershell
python tools/streetscape_cnn/audit_captions.py "E:\个人项目\lunwen\训练集\images" --expected-count 480 --allow-unpaired
```

`--allow-unpaired`只用于审计。训练器始终要求独立图文目录，不允许把未配对图片静默当成空描述训练。后续划分锁定后，在服务器提供该次实验对应的独立图文目录；不默认把480张候选池全部分配为训练集。

## 5. Linux服务器验收与启动

研究分支已推送；每次部署仍必须记录并拉取确切提交。所有命令均从仓库根目录执行；不要复用原CNN环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# 依据服务器GPU/驱动安装兼容的torch与torchvision，再安装仓库依赖。
pip install -r requirements.txt
mkdir -p server_acceptance_results
pip freeze > server_acceptance_results/requirements.txt
python tools/streetscape_cnn/server_validate.py --unit
```

把原 `final.pt`、`model_manifest.json`、`config.json` 原样放到 `research_assets/cnn/`；不放Git。RAW和文本/VAE资产可以通过标准HF缓存准备。本扩展在服务器首次启动时下载缺少的资产并绑定实际内容，不在本地实现阶段下载。

```bash
python tools/streetscape_cnn/server_validate.py \
  --cnn-checkpoint research_assets/cnn/final.pt \
  --cnn-manifest research_assets/cnn/model_manifest.json \
  --cnn-image /absolute/path/to/approved-development-image.png
```

复制两个示例YAML到 `config/`，填入正式决定的参数；另建独立工程烟测配置 `config/krea2_streetscape_smoke.yaml`，使用少量已批准的图文样例、独立输出目录与少量步数。烟测参数不作为正式实验或自动调参结果。确保CNN起始步位于烟测范围内，且噪声区间能产生合格样本。

```bash
python tools/streetscape_cnn/server_train_smoke.py \
  config/krea2_streetscape_smoke.yaml --check-gradients --stop-after 2

# 完全同一配置，原生最新进度恢复；不改steps/lr/weight/schedule。
python tools/streetscape_cnn/server_train_smoke.py \
  config/krea2_streetscape_smoke.yaml --check-gradients

python tools/streetscape_cnn/server_validate.py \
  --run-dir research_runs/REPLACE_WITH_SMOKE_NAME \
  --export research_runs/REPLACE_WITH_SMOKE_NAME/REPLACE_WITH_EXPORT.safetensors
```

结构检查不能替代真实LoRA加载。以下入口使用普通上游 `sd_trainer`，加载导出的LoRA并执行**零优化器步数**的采样检查，必须提供独立空目录和明确的推理参数：

```bash
python tools/streetscape_cnn/server_export_inference.py \
  config/krea2_streetscape_smoke.yaml /absolute/path/to/export.safetensors \
  --output server_acceptance_results/export-load \
  --prompt "A stone-paved neighborhood street with pedestrians." \
  --steps 52 --cfg 4.5 --width 768 --height 512 --seed 42
```

该命令验证RAW上原生加载/采样；不宣称已验证Turbo迁移或生成质量。普通训练正式入口仍为 `python run.py config/FILLED_CONFIG.yaml`。

必须逐项确认：关闭兼容、标准化还原与共融一致、step/sigma门控、非法配置与身份拒绝、CNN→LoRA有限非零梯度、冻结参数/缓冲区、真实VAE解码、缓存复用、目标量化/卸载设置、微批次累积、保存/退出/恢复、导出加载。

当前5090 32GB只验收已登记的FP8和卸载配置，不要求完整无量化模型驻留。任何失败立即停止正式训练，保留日志修复，不自动改科研目标或权重，也不自动开展正式实验。

## 6. 2026-10-12工程部署合同

本轮仅部署与24步工程短测。参考T8视频BV1erTg66Eo5的batch 2、1024单档、rank 32、AdamW8Bit和1e-4起始学习率。视频中的后期提速与回落用于验证学习率日程，不预设正式训练最优参数。

- `local_model_assets`包含raw、text_encoder、vae、manifest。公共模型库须与固定官方版本的权重SHA-256和配置git-blob一致；仅核验实际所用组件，不复制或修改模型库。
- `research_lr_schedule`使用显式enabled及stages，每项为start_step和lr。工程日程为原生step 0/8/16开始的1e-4/1.5e-4/1e-4；仅与factor=1的原生constant调度器配合。
- `research_recording`包含enabled、per_sample_scores、archive_checkpoints布尔字段。关闭保持原行为；开启后记录成功更新数与图片呈现次数，二者不能混称训练步数。
- 研究训练首次OOM及非有限值直接失败，不让上游吞异常跳批。只有OOM可触发已登记的独立batch 1、原生微批次累积2复测，两组配置一致且重新初始化；不降低分辨率或rank。
- 停止入口为`python tools/streetscape_cnn/server_control.py request-stop --run-dir RUN_DIR`。停止请求绑定内容身份，完整保存后归档消费；恢复不会再次消费旧请求。该入口不加载模型或关闭实例。
- 逐步与逐样本JSONL位于run的records子目录。未评分为null；所有记录detach后写出，不缓存新预测分数。
- 每次完整保存归档LoRA、optimizer、scheduler与计数状态、身份及哈希。周期保存使用下一原生step作为恢复进度，避免重复执行已完成更新。仍不保证随机数或数据读取位置逐位恢复。
- 原生done_hook在释放模型后调用，因此冻结参数/缓冲区校验在最终保存前完成。完成报告单独保存，文件存在不等于验收通过。
- 独立环境由tools/streetscape_cnn/requirements_server.txt定义，核心组合为Python 3.12、torch 2.9.1/cu128、vision 0.24.1、audio 2.9.1、torchao 0.15.0、torchcodec 0.9.0、bitsandbytes 0.50.2；实际解析结果须冻结。

数据入口为prepare_dataset.py：480个候选ID独立划分384/48/48，原型各48/6/6，新增同批次同原型四状态不拆组，原CNN角色只保留为历史字段。发现打标器无损重编码PNG时，必须对每张核对封存源文件哈希、模式、尺寸和完整像素；新清单同时保留原始和当前文件哈希，不修改原图、描述或旧清单。任一像素不一致立即停止。

服务器依次运行环境和GPU预检、全部单元测试、模型资产与token门禁、真实CNN梯度、普通LoRA及CNN短测（step 8保存退出再恢复）、真实导出加载。烟测采样FlowMatch 30步、CFG 4、seed 42、768×512，仅验证接口。完成后保留全部失败现场与备份，正式训练需另行确认。
# 32GB memory acceptance v1

The opt-in `cnn_memory_profile` is `legacy` (default), `vae_checkpoint`, or
`vae_checkpoint_cpu`. The latter policies explicitly enable the frozen Qwen VAE
decoder's upstream cache-safe, single-frame checkpointing patch. CPU saving uses
native `torch.autograd.graph.save_on_cpu(pin_memory=False)` only around the
differentiable VAE/CNN branch. It does not detach the reward or change its math.

`research_memory_acceptance: true` requires durable recording. It measures phase
allocator peaks, sampled current-process GPU memory (200 ms), RSS and host
available memory. Native updates fail closed above 28 GiB or when host available
memory drops below the 2 GiB operational safety reserve. CPU packed-tensor byte
counts are transfer/work counters, not unique RAM usage. GPU sampling can miss
short spikes; allocator peaks are recorded separately.

Generate registered independent configs with `make_smoke_configs.py --memory-tier
1|2|3`. All tiers fix microbatch 1, native accumulation 2, RAW/1024/rank32 and the
unchanged smoke objective/schedule. P1 uses decoder checkpointing and 50% model
offload, P2 adds CPU activations, P3 increases model offload to 100%. Run the real
FP32 parity gate before memory probes. Only a memory-gate failure permits the
next tier; gradient, frozen-state, resume or host-memory failures stop acceptance.
Old output identities are read-only, not resume starting points for new sources.

The short runs and sample outputs are engineering artifacts, never formal
generative-quality results. Do not automatically start long training or resize,
purchase, stop or reinstall the cloud instance.
