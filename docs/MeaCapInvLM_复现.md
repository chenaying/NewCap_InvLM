# MeaCap_InvLM 复现指南

MeaCap_InvLM = **ViECap 解码器**（`coco_prefix-0014.pt` + GPT-2）+ **MeaCap 记忆库 retrieve-then-filter**（替代 ViECap 的 COCO 词表实体）。

## 1. 目录与文件

```
MeaCapInvLM/
├── viecap_inference.py      # InvLM 入口（你已修改）
├── infer_by_instance.py     # ViECap 基线对比
├── meacap_utils/            # retrieve_concepts + parse_tool
├── models/clip_utils.py     # 记忆检索用 HF CLIP
├── checkpoints/train_coco/coco_prefix-0014.pt
├── data/memory/coco/
│   ├── memory_captions.json
│   ├── memory_clip_embeddings.pt
│   └── memory_wte_embeddings.pt
└── scripts/run_meacap_invlm.sh
```

## 2. 依赖与数据

- ViECap `checkpoints.zip` → `checkpoints/`、`annotations/`（基线对比需要）
- MeaCap 记忆库：[HuggingFace memory/coco](https://huggingface.co/JoeyZoZ/MeaCap/tree/main/memory)
- 本地模型（离线）：gpt2、clip-vit-base-patch32、flan-t5-base-VG-factual-sg、all-MiniLM-L6-v2

## 3. 运行 MeaCap InvLM

### 单张图

```bash
cd ~/project/MeaCap_InvLM
conda activate meacap

bash scripts/run_meacap_invlm.sh images/instance1.jpg
```

### 批量评测（CIDEr）

| 数据集 | MeaCap InvLM 脚本 | 对应 ViECap 原脚本 |
|--------|-------------------|-------------------|
| COCO | `bash scripts/eval_coco_meacap.sh train_coco 0 14` | `bash eval_coco.sh train_coco 0 '' 14` |
| Flickr30k | `bash scripts/eval_flickr30k_meacap.sh train_flickr30k 0 29` | `bash eval_flickr30k.sh train_flickr30k 0 '' 29` |
| NoCaps | `bash scripts/eval_nocaps_meacap.sh train_coco 0 14` | `bash eval_nocaps.sh train_coco 0 '--top_k 3 --threshold 0.2' 14` |

默认记忆库：`MEMORY_ID=coco`（COCO/NoCaps）、`MEMORY_ID=flickr30k`（Flickr30k 域内）。跨域时可 `export MEMORY_ID=coco` 覆盖。

所需特征 pickle：

- `annotations/coco/test_captions_ViT-B32.pickle`
- `annotations/flickr30k/test_captions_ViT-B32.pickle`
- `annotations/nocaps/nocaps_corpus_ViT-B32.pickle`

或手动：

```bash
python viecap_inference.py \
  --device cuda:0 \
  --memory_id coco \
  --image_path images/instance1.jpg \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --using_hard_prompt \
  --soft_prompt_first \
  --local_files_only
```

成功输出示例：

```
memory concepts: ['girl', 'bed', 'stuffed animal', ...]
the generated caption: ...
```

## 4. 与 ViECap 基线对比

| 项目 | ViECap | MeaCap InvLM |
|------|--------|----------------|
| 脚本 | `infer_by_instance.py` | `viecap_inference.py` |
| 硬提示来源 | COCO 词表 + CLIP | 记忆库 + 场景图过滤 |
| 推荐参数 | `--using_hard_prompt --soft_prompt_first --prompt_ensemble --name_of_entities_text coco_entities --threshold 0.4` | `--using_hard_prompt --soft_prompt_first` |

```bash
# ViECap（已验证可用）
python infer_by_instance.py \
  --device cuda:0 \
  --image_path images/instance1.jpg \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --language_model /path/to/gpt2 \
  --using_hard_prompt --soft_prompt_first --prompt_ensemble \
  --name_of_entities_text coco_entities --threshold 0.4
```

## 5. 提示模式

| 模式 | 参数 |
|------|------|
| InvLM 默认（软+硬，与 ViECap eval 一致） | `--using_hard_prompt --soft_prompt_first` |
| MeaCap 论文默认（硬+软） | `--using_hard_prompt`（不加 `--soft_prompt_first`） |
| 仅硬提示 | `--using_hard_prompt --only_hard_prompt` |
| 仅软提示 | 不推荐，易重复乱码；用 `infer_by_instance.py` 且勿加 `--using_hard_prompt` |

## 6. 常见问题

- **`local_files_only` / CLIP 报错**：同步 `models/clip_utils.py`，或使用已更新的 `viecap_inference.py` 中的 `_load_memory_clip`
- **记忆库找不到**：检查 `data/memory/coco/` 三个文件
- **重复 Eleven**：不要单独软提示；必须 `--using_hard_prompt` + 正确 checkpoint
- **bash 换行**：用 `\`，不要用 PowerShell 的 `` ` ``
