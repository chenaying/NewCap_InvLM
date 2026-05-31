# ViECap论文：CLIP-based Entity Classifier 模块详解

> **论文**: Transferable Decoding with Visual Entities for Zero-Shot Image Captioning (ICCV 2023)
> 
> **作者**: Junjie Fei, Teng Wang, Jinrui Zhang, Zhenyu He, Chengjie Wang, Feng Zheng

---

## 一、模块概述

### 1.1 背景与动机

在零样本图像描述（Zero-Shot Image Captioning）任务中，模型需要描述从未见过的图像内容。传统方法存在两个主要问题：

1. **模态偏差（Modality Bias）**: 语言模型倾向于生成训练时常见的描述，而忽略实际图像内容
2. **对象幻觉（Object Hallucination）**: 生成的描述可能包含图像中不存在的物体

ViECap提出使用**CLIP-based Entity Classifier**来解决这些问题。该模块利用CLIP模型强大的零样本分类能力，从图像中提取视觉实体，并将其作为**Hard Prompt**引导语言模型生成更准确的描述。

### 1.2 核心思想

```
输入图像 → CLIP视觉编码 → 与实体词汇表匹配 → 提取Top-K实体 → 构建Hard Prompt
                                                              ↓
                                              "There are person, dog, park in image."
```

---

## 二、技术架构

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CLIP-based Entity Classifier                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐              ┌──────────────────────────────┐    │
│   │    输入图像       │              │       实体词汇表              │    │
│   │   (H × W × 3)    │              │  V = {e₁, e₂, ..., eₙ}       │    │
│   └────────┬─────────┘              │  例: [dog, cat, person, ...]  │    │
│            │                        └──────────────┬───────────────┘    │
│            ▼                                       │                     │
│   ┌──────────────────┐                             │                     │
│   │   CLIP图像编码器  │                             ▼                     │
│   │    (ViT-B/32)    │              ┌──────────────────────────────┐    │
│   │                  │              │      Prompt Ensemble          │    │
│   │  f_img ∈ R^512   │              │  对每个实体生成多个prompt:      │    │
│   └────────┬─────────┘              │  • "itap of a {entity}."      │    │
│            │                        │  • "a photo of the {entity}." │    │
│            │                        │  • "art of the {entity}."     │    │
│            │                        └──────────────┬───────────────┘    │
│            │                                       │                     │
│            │                                       ▼                     │
│            │                        ┌──────────────────────────────┐    │
│            │                        │     CLIP文本编码器            │    │
│            │                        │  多模板特征 → 平均 → 归一化    │    │
│            │                        │  f_text ∈ R^(N×512)          │    │
│            │                        └──────────────┬───────────────┘    │
│            │                                       │                     │
│            ▼                                       ▼                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      余弦相似度计算                               │   │
│   │                                                                  │   │
│   │              sim(f_img, f_text_i)                                │   │
│   │   logits_i = ─────────────────────                               │   │
│   │                      τ                                           │   │
│   │                                                                  │   │
│   │   probs = softmax(logits)                                        │   │
│   └─────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  Top-K 选择 + 阈值过滤                            │   │
│   │                                                                  │   │
│   │   • 选择概率最高的 K 个实体 (默认 K=3)                             │   │
│   │   • 过滤概率低于阈值 θ 的实体 (默认 θ=0.2)                         │   │
│   └─────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Hard Prompt 生成                            │   │
│   │                                                                  │   │
│   │           "There are {entity₁}, {entity₂}, ... in image."       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心组件详解

### 3.1 实体词汇表（Entity Vocabulary）

#### 3.1.1 支持的词汇表类型

| 词汇表名称 | 来源 | 类别数量 | 特点 |
|-----------|------|---------|------|
| `vinvl_vgoi_entities` | VG + COCO + OpenImages | ~1600+ | **默认使用**，覆盖面最广 |
| `visual_genome_entities` | Visual Genome | ~3000+ | 细粒度物体和属性 |
| `coco_entities` | MS-COCO | 80 | 常见物体 |
| `open_image_entities` | Google Open Images | 601 | 多样化类别 |
| `vinvl_vg_entities` | VinVL预训练词汇 | ~1500 | 检测器常用词汇 |

#### 3.1.2 词汇表加载代码

```python
# load_annotations.py

def load_entities_text(name_of_entities: str, path_of_entities: str, all_entities: bool = True) -> List[str]:
    """
    加载实体词汇表
    
    Args:
        name_of_entities: 词汇表名称
        path_of_entities: 词汇表文件路径
        all_entities: True表示包含多词实体(如"traffic light")，False只保留单词实体
    
    Return:
        实体列表 [entity1, entity2, ...]
    """
    if name_of_entities == 'visual_genome_entities':
        return load_visual_genome_entities(path_of_entities, all_entities)
    elif name_of_entities == 'coco_entities':
        return load_coco_entities(path_of_entities, all_entities)
    # ... 其他词汇表
```

#### 3.1.3 词汇表示例

```python
# vinvl_vgoi_entities 词汇表示例
entities = ['person', 'dog', 'cat', 'car', 'tree', 'building', 'street', 
            'sky', 'grass', 'water', 'table', 'chair', 'food', ...]
```

---

### 3.2 Prompt Ensemble（提示集成）

#### 3.2.1 技术原理

Prompt Ensemble是CLIP论文中提出的技术，通过使用**多个不同的提示模板**对同一类别进行编码，然后取平均，可以获得更鲁棒的类别表示。

**数学表达**：

对于实体 $e$，使用 $M$ 个提示模板 $\{T_1, T_2, ..., T_M\}$：

$$\mathbf{f}_e = \frac{1}{M} \sum_{m=1}^{M} \text{normalize}(\text{CLIP}_{\text{text}}(T_m(e)))$$

其中 $T_m(e)$ 表示将实体 $e$ 填入模板 $T_m$。

#### 3.2.2 使用的提示模板

```python
# generating_prompt_ensemble.py

prompt_templates = [
    'itap of a {}.',                    # "itap" = "I took a picture"
    'a bad photo of the {}.',           # 模拟低质量图片
    'a origami {}.',                    # 折纸风格
    'a photo of the large {}.',         # 大尺寸物体
    'a {} in a video game.',            # 游戏场景
    'art of the {}.',                   # 艺术风格
    'a photo of the small {}.'          # 小尺寸物体
]
```

**设计思想**：这些模板覆盖了不同的场景和风格，使得模型对实体的表示更加全面和鲁棒。

#### 3.2.3 Ensemble特征生成代码

```python
# generating_prompt_ensemble.py

@torch.no_grad()
def generate_ensemble_prompt_embeddings(
    device: str,
    clip_type: str,
    entities: List[str],
    prompt_templates: List[str],
    outpath: str,
):
    """
    为每个实体生成Prompt Ensemble特征
    
    流程:
    1. 对每个实体，用所有模板生成文本
    2. 用CLIP文本编码器编码所有文本
    3. 对编码结果取平均并归一化
    """
    model, _ = clip.load(clip_type, device)
    model.eval()
    
    embeddings = []
    for entity in tqdm(entities):
        # Step 1: 生成所有提示文本
        texts = [template.format(entity) for template in prompt_templates]
        # 例: entity="dog" → ["itap of a dog.", "a bad photo of the dog.", ...]
        
        # Step 2: CLIP文本编码
        tokens = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(tokens).to('cpu')
        
        # Step 3: 归一化 → 平均 → 再归一化
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        
        embeddings.append(class_embedding)
    
    embeddings = torch.stack(embeddings, dim=0)  # (num_entities, 512)
    
    # 保存到文件
    with open(outpath, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
    
    return embeddings
```

---

### 3.3 图像-文本相似度计算

#### 3.3.1 计算公式

给定：
- 图像特征 $\mathbf{f}_{\text{img}} \in \mathbb{R}^{512}$
- 实体文本特征矩阵 $\mathbf{F}_{\text{text}} \in \mathbb{R}^{N \times 512}$（N为实体数量）
- 温度参数 $\tau$（默认0.01）

相似度计算：

$$\text{sim}_i = \frac{\mathbf{f}_{\text{img}} \cdot \mathbf{f}_{\text{text}_i}^T}{||\mathbf{f}_{\text{img}}|| \cdot ||\mathbf{f}_{\text{text}_i}||}$$

$$\text{logits}_i = \frac{\text{sim}_i}{\tau}$$

$$\text{prob}_i = \frac{\exp(\text{logits}_i)}{\sum_{j=1}^{N} \exp(\text{logits}_j)}$$

#### 3.3.2 温度参数的作用

温度参数 $\tau$ 控制概率分布的"锐度"：

| τ值 | 效果 |
|-----|------|
| τ → 0 | 分布趋向one-hot，只有最相似的类别概率接近1 |
| τ = 1 | 标准softmax |
| τ → ∞ | 分布趋向均匀分布 |

ViECap使用 $\tau = 0.01$，使得分布非常尖锐，有利于准确选择最相关的实体。

#### 3.3.3 实现代码

```python
# retrieval_categories.py

def image_text_simiarlity(
    texts_embeddings: torch.Tensor,      # (num_categories, 512)
    temperature: float = 0.01,
    image_path: Optional[str] = None,
    images_features: Optional[torch.Tensor] = None,  # (num_images, 512)
    clip_type: Optional[str] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    计算图像与所有实体类别的相似度
    
    Return:
        logits: (num_images, num_categories) 概率分布
    """
    # 如果只提供图像路径，先提取图像特征
    if images_features is None:
        encoder, preprocess = clip.load(clip_type, device)
        image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            images_features = encoder.encode_image(image)
    
    # 特征归一化（在CPU上计算避免显存不足）
    images_features = images_features.float().to('cpu')
    texts_embeddings = texts_embeddings.float().to('cpu')
    images_features /= images_features.norm(dim=-1, keepdim=True)
    texts_embeddings /= texts_embeddings.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度 / 温度
    image_to_text_similarity = torch.matmul(
        images_features, 
        texts_embeddings.transpose(1, 0)
    ) / temperature
    
    # Softmax转换为概率
    image_to_text_logits = torch.nn.functional.softmax(
        image_to_text_similarity, 
        dim=-1
    )
    
    return image_to_text_logits
```

---

### 3.4 Top-K选择与阈值过滤

#### 3.4.1 选择策略

为了避免引入噪声实体，ViECap采用两步筛选：

1. **Top-K选择**：选择概率最高的K个实体
2. **阈值过滤**：过滤掉概率低于阈值θ的实体

#### 3.4.2 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 3 | 最多选择3个实体 |
| `threshold` | 0.2 | 概率低于20%的实体被过滤 |

#### 3.4.3 实现代码

```python
# retrieval_categories.py

def top_k_categories(
    texts: List[str],              # 实体词汇表
    logits: torch.Tensor,          # (num_images, num_categories) 概率
    top_k: Optional[int] = 5,
    threshold: Optional[float] = 0.0
) -> Tuple:
    """
    选择Top-K实体并过滤低置信度结果
    
    Return:
        top_k_texts: 每张图像检测到的实体列表
        top_k_probs: 对应的概率
    """
    # 获取Top-K
    top_k_probs, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    
    top_k_texts = []
    for i in range(len(top_k_probs)):
        per_image_top_k_probs = top_k_probs[i]
        per_image_top_k_indices = top_k_indices[i]
        
        temp_texts = []
        for j in range(top_k):
            # 阈值过滤：如果概率低于阈值，停止添加
            if per_image_top_k_probs[j] < threshold:
                break
            temp_texts.append(texts[per_image_top_k_indices[j]])
        
        top_k_texts.append(temp_texts)
    
    return top_k_texts, top_k_probs
```

#### 3.4.4 示例

假设对一张包含人和狗在公园的图像进行分类：

```
实体概率分布:
  person: 0.45
  dog: 0.32
  park: 0.15
  grass: 0.03
  tree: 0.02
  ...

Top-3选择: [person, dog, park]
阈值过滤(θ=0.2): [person, dog]  # park的概率0.15 < 0.2，被过滤
```

---

### 3.5 Hard Prompt生成

#### 3.5.1 Prompt模板

ViECap使用固定的模板将检测到的实体组装成自然语言提示：

```
"There are {entity1}, {entity2}, ... in image."
```

#### 3.5.2 实现代码

```python
# utils.py

def compose_discrete_prompts(
    tokenizer,
    process_entities: List[str],  # 检测到的实体列表
) -> torch.Tensor:
    """
    将实体列表组装成Hard Prompt
    
    Examples:
        输入: ['person', 'dog']
        输出: "There are person, dog in image."
        
        输入: []  (没有检测到实体)
        输出: "There are something in image."
    """
    prompt_head = 'There are'
    prompt_tail = ' in image.'
    
    if len(process_entities) == 0:
        # 没有检测到实体时使用占位符
        discrete_prompt = prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities:
            discrete_prompt += ' ' + entity + ','
        discrete_prompt = discrete_prompt[:-1]  # 移除最后的逗号
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail
    
    # Tokenize
    entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))
    
    return entities_tokens
```

#### 3.5.3 示例

| 检测实体 | 生成的Hard Prompt |
|---------|------------------|
| `['person', 'dog', 'park']` | "There are person, dog, park in image." |
| `['cat']` | "There are cat in image." |
| `[]` | "There are something in image." |

---

## 四、与语言模型的集成

### 4.1 Soft Prompt vs Hard Prompt

ViECap同时使用两种类型的prompt：

| 类型 | 来源 | 特点 |
|------|------|------|
| **Soft Prompt** | 通过MLP将CLIP图像特征映射到GPT embedding空间 | 连续向量，包含丰富的视觉信息 |
| **Hard Prompt** | Entity Classifier检测的实体 | 离散token，提供明确的实体语义 |

### 4.2 Prompt拼接策略

```python
# infer_by_instance.py

if args.using_hard_prompt:
    # 生成Hard Prompt
    logits = image_text_simiarlity(texts_embeddings, images_features=image_features)
    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects[0])
    discrete_embeddings = model.word_embed(discrete_tokens)
    
    if args.only_hard_prompt:
        # 只使用Hard Prompt
        embeddings = discrete_embeddings
    elif args.soft_prompt_first:
        # Soft Prompt在前（推荐）
        embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
    else:
        # Hard Prompt在前
        embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
else:
    # 只使用Soft Prompt
    embeddings = continuous_embeddings
```

### 4.3 输入到语言模型的格式

```
[Soft Prompt (10 tokens)] + [Hard Prompt] + [Generated Caption]
     ↓                          ↓                   ↓
 视觉特征映射              "There are..."        "A person..."
```

---

## 五、关键超参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `clip_model` | ViT-B/32 | CLIP模型类型 | 特征质量和速度的权衡 |
| `temperature` | 0.01 | 相似度计算的温度 | 越小分布越尖锐 |
| `top_k` | 3 | 最多选择的实体数 | 太大可能引入噪声 |
| `threshold` | 0.2 | 概率阈值 | 太小可能引入低置信度实体 |
| `prompt_ensemble` | True | 是否使用Prompt Ensemble | 提升分类鲁棒性 |

---

## 六、跨域迁移能力分析

### 6.1 为什么CLIP-based方法具有强跨域能力？

1. **大规模预训练**：CLIP在4亿图文对上预训练，见过海量的概念组合
2. **开放词汇表**：不受限于固定类别，可以检测任意预定义的实体
3. **语义对齐**：图像和文本在同一空间中，天然具有跨模态匹配能力

### 6.2 实验证据

| 任务 | 方法 | CIDEr |
|------|------|-------|
| COCO → NoCaps (Out-domain) | CapDec | 28.7 |
| COCO → NoCaps (Out-domain) | **ViECap** | **65.0** |
| COCO → Flickr30k | DeCap | 35.7 |
| COCO → Flickr30k | **ViECap** | **38.4** |

在Out-domain场景（图像包含训练集从未见过的物体类别），ViECap的提升尤为显著（+36.3 CIDEr）。

---

## 七、与传统目标检测器的对比

| 方面 | CLIP-based Entity Classifier | 传统目标检测器 (如Faster R-CNN) |
|------|------------------------------|--------------------------------|
| **训练需求** | 零样本，无需额外训练 | 需要在目标数据集上训练 |
| **词汇表** | 开放词汇，可任意扩展 | 固定类别（如COCO 80类） |
| **输出格式** | 全局实体概率 | 边界框 + 类别 + 置信度 |
| **计算成本** | 较低（单次前向传播） | 较高（RPN + ROI Pooling） |
| **跨域能力** | 强 | 弱（受限于训练集分布） |
| **空间信息** | 无（全局分类） | 有（边界框位置） |

---

## 八、代码使用示例

### 8.1 单图推理

```bash
python infer_by_instance.py \
    --prompt_ensemble \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --top_k 3 \
    --threshold 0.2
```

### 8.2 批量验证

```bash
# 在NoCaps数据集上评估
bash eval_nocaps.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

---

## 九、总结

CLIP-based Entity Classifier是ViECap实现零样本图像描述跨域迁移的核心模块，其主要贡献包括：

1. **利用CLIP的零样本能力**：无需额外训练即可检测开放词汇表中的实体
2. **Prompt Ensemble技术**：提升实体分类的鲁棒性
3. **Hard Prompt机制**：将视觉实体显式注入语言模型，减少对象幻觉
4. **灵活的词汇表设计**：支持多种词汇表来源，适应不同场景

这种设计使得ViECap在跨域场景（特别是Out-domain）下显著优于现有方法，为零样本图像描述提供了一种有效的解决方案。

---

## 参考资料

- [ViECap官方仓库](https://github.com/FeiElysia/ViECap)
- [CLIP论文](https://arxiv.org/abs/2103.00020)
- [ViECap论文](https://arxiv.org/abs/2307.16525)


