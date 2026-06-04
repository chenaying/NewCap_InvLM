# MeaCap InvLM single-image inference (flat ViECap layout, NOT `from viecap.xxx`).
# If you see ModuleNotFoundError: No module named 'viecap', replace this file from the repo.

import copy
import os
import json

import clip
import torch
import argparse
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import compose_discrete_prompts
from search import greedy_search, beam_search, opt_search
from meacap_utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP

cpu_device = torch.device("cpu")


def _hf_load_kwargs(local_files_only: bool) -> dict:
    return {'local_files_only': True} if local_files_only else {}


@torch.no_grad()
def main(args) -> None:
    device = torch.device(args.device)
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512
    hf_kw = _hf_load_kwargs(args.local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model, **hf_kw)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device=device)

    vl_model = CLIP(args.vl_model, local_files_only=args.local_files_only)
    vl_model = vl_model.to(device)
    print('Load memory-retrieval CLIP from {}.'.format(args.vl_model))

    wte_model = SentenceTransformer(args.wte_model_path)
    print('Load sentenceBERT from {}.'.format(args.wte_model_path))

    parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint, **hf_kw)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint, **hf_kw)
    parser_model.eval()
    parser_model.to(device)
    print('Load Textual Scene Graph parser from the checkpoint {}.'.format(args.parser_checkpoint))

    memory_id = args.memory_id
    memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
    memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
    memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")
    memory_clip_embeddings = torch.load(memory_clip_embedding_file, map_location=device)
    memory_wte_embeddings = torch.load(memory_wte_embedding_file, map_location=device)
    with open(memory_caption_path, 'r', encoding='utf-8') as f:
        memory_captions = json.load(f)

    if memory_id in ('cc3m', 'ss1m'):
        retrieve_on_CPU = True
        print('CC3M/SS1M memory is large; running retrieval on CPU...')
        vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
        memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
    else:
        vl_model_retrieve = vl_model
        retrieve_on_CPU = False

    image = preprocess(Image.open(args.image_path)).unsqueeze(dim=0).to(device)
    image_features = encoder.encode_image(image).float()
    image_features /= image_features.norm(2, dim=-1, keepdim=True)
    continuous_embeddings = model.mapping_network(image_features).view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )

    if args.using_hard_prompt:
        batch_image_embeds = vl_model.compute_image_representation_from_image_path(args.image_path)

        if not retrieve_on_CPU:
            clip_score, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                batch_image_embeds, memory_clip_embeddings
            )
        else:
            batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
            clip_score_cpu, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                batch_image_embeds_cpu,
                memory_clip_embeddings,
            )
            clip_score = clip_score_cpu.to(device)

        select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
        select_memory_captions = [memory_captions[i] for i in select_memory_ids]
        detected_objects = retrieve_concepts(
            parser_model=parser_model,
            parser_tokenizer=parser_tokenizer,
            wte_model=wte_model,
            select_memory_captions=select_memory_captions,
            image_embeds=batch_image_embeds,
            device=device,
        )

        print("memory concepts:", detected_objects)
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
        discrete_embeddings = model.word_embed(discrete_tokens)
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
    else:
        embeddings = continuous_embeddings

    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.gpt,
            )
            sentence = sentence[0]
        else:
            sentence = greedy_search(embeddings=embeddings, tokenizer=tokenizer, model=model.gpt)
    else:
        sentence = opt_search(
            prompts=args.text_prompt,
            embeddings=embeddings,
            tokenizer=tokenizer,
            beam_width=args.beam_width,
            model=model.gpt,
        )
        sentence = sentence[0]

    print(f'the generated caption: {sentence}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument(
        '--language_model',
        default='/home/teacher5/data1/cyp/project/NewCap/gpt2',
        help='local GPT-2 dir or HuggingFace model id',
    )
    parser.add_argument(
        '--vl_model',
        type=str,
        default='/home/teacher5/data1/cyp/project/NewCap/checkpoints/clip-vit-base-patch32',
        help='local HF CLIP dir or model id for memory retrieval',
    )
    parser.add_argument(
        '--parser_checkpoint',
        type=str,
        default='/home/teacher5/data1/cyp/project/NewCap/checkpoints/flan-t5-base-VG-factual-sg',
    )
    parser.add_argument(
        '--wte_model_path',
        type=str,
        default='/home/teacher5/data1/cyp/project/NewCap/checkpoints/all-MiniLM-L6-v2',
    )
    parser.add_argument(
        '--local_files_only',
        action='store_true',
        default=True,
        help='offline mode: load HF models only from local paths (no huggingface.co)',
    )
    parser.add_argument('--continuous_prompt_length', type=int, default=10)
    parser.add_argument('--clip_project_length', type=int, default=10)
    parser.add_argument('--weight_path', default='checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default='images/instance1.jpg')
    parser.add_argument('--using_hard_prompt', action='store_true', default=True)
    parser.add_argument('--soft_prompt_first', action='store_true', default=False)
    parser.add_argument('--only_hard_prompt', action='store_true', default=False)
    parser.add_argument('--using_greedy_search', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--text_prompt', type=str, default=None)
    parser.add_argument('--memory_id', type=str, default='coco', help='memory bank name')
    parser.add_argument('--memory_caption_num', type=int, default=5)
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))
    main(args)
