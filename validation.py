import os
import json
import clip
import torch
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
from typing import List, Optional
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories
from meacap_utils.invlm_prompt import (
    MeaCapInvLMResources,
    hard_prompt_embeddings,
    compute_continuous_embeddings,
    resolve_image_path,
    check_nocaps_images_available,
)


def _decode_caption(args, model, tokenizer, embeddings) -> str:
    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.gpt,
            )
            return sentence[0]
        return greedy_search(embeddings=embeddings, tokenizer=tokenizer, model=model.gpt)
    sentence = opt_search(
        prompts=args.text_prompt,
        embeddings=embeddings,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        model=model.gpt,
    )
    return sentence[0]


def validation_nocaps(
    args,
    inpath: str,
    entities_text: List[str],
    texts_embeddings: torch.Tensor,
    model: ClipCaptionModel,
    tokenizer: AutoTokenizer,
    preprocess: clip = None,
    encoder: clip = None,
    invlm_resources: Optional[MeaCapInvLMResources] = None,
) -> None:
    device = args.device
    if args.using_image_features:
        if not os.path.isfile(inpath):
            raise FileNotFoundError(
                f'Image feature pickle not found: {inpath}\n'
                'Download ViECap checkpoints.zip or run: python images_features_extraction.py'
            )
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile)
    else:
        with open(inpath, 'r', encoding='utf-8') as infile:
            annotations = json.load(infile)

    n_samples = len(annotations)
    print(f'[validation] dataset=nocaps inpath={inpath} num_samples={n_samples}')
    if n_samples == 0:
        raise ValueError(f'Empty annotations in {inpath}. Cannot run evaluation.')

    if args.use_meacap_invlm:
        samples = []
        for ann in annotations[:10]:
            if args.using_image_features:
                samples.append((ann[0], ann[1]))
            else:
                samples.append((ann['image_id'], ann['split']))
        check_nocaps_images_available(args, samples)

    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    for idx, annotation in tqdm(enumerate(annotations), total=n_samples):
        if args.using_image_features:
            image_id, split, image_features, captions = annotation
            image_features = image_features.float().unsqueeze(dim=0).to(device)
            image_path = resolve_image_path(args, image_id, split)
        else:
            image_id = annotation['image_id']
            split = annotation['split']
            captions = annotation['caption']
            image_path = resolve_image_path(args, image_id, split)
            image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
            image_features = encoder.encode_image(image).float()

        continuous_embeddings = compute_continuous_embeddings(
            args, model, image_features, invlm_resources, image_path
        )

        if args.using_hard_prompt:
            embeddings = hard_prompt_embeddings(
                args, model, tokenizer, continuous_embeddings, image_features,
                image_path, device, invlm_resources, entities_text, texts_embeddings,
            )
        else:
            embeddings = continuous_embeddings

        sentence = _decode_caption(args, model, tokenizer, embeddings)

        predict = {
            'split': split,
            'image_name': image_id,
            'captions': captions,
            'prediction': sentence,
        }
        overall.append(predict)
        if split == 'in_domain':
            indomain.append(predict)
        elif split == 'near_domain':
            neardomain.append(predict)
        elif split == 'out_domain':
            outdomain.append(predict)

    with open(os.path.join(args.out_path, 'overall_generated_captions.json'), 'w') as outfile:
        json.dump(overall, outfile, indent=4)
    with open(os.path.join(args.out_path, 'indomain_generated_captions.json'), 'w') as outfile:
        json.dump(indomain, outfile, indent=4)
    with open(os.path.join(args.out_path, 'neardomain_generated_captions.json'), 'w') as outfile:
        json.dump(neardomain, outfile, indent=4)
    with open(os.path.join(args.out_path, 'outdomain_generated_captions.json'), 'w') as outfile:
        json.dump(outdomain, outfile, indent=4)


def validation_coco_flickr30k(
    args,
    inpath: str,
    entities_text: List[str],
    texts_embeddings: torch.Tensor,
    model: ClipCaptionModel,
    tokenizer: AutoTokenizer,
    preprocess: clip = None,
    encoder: clip = None,
    invlm_resources: Optional[MeaCapInvLMResources] = None,
) -> None:
    device = args.device
    if args.using_image_features:
        if not os.path.isfile(inpath):
            raise FileNotFoundError(
                f'Image feature pickle not found: {inpath}\n'
                'Download ViECap checkpoints.zip or run: python images_features_extraction.py'
            )
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile)
    else:
        with open(inpath, 'r', encoding='utf-8') as infile:
            annotations = json.load(infile)

    n_samples = len(annotations)
    print(f'[validation] dataset={args.name_of_datasets} inpath={inpath} num_samples={n_samples}')
    if n_samples == 0:
        raise ValueError(f'Empty annotations in {inpath}. Cannot run evaluation.')

    predicts = []
    for idx, item in tqdm(enumerate(annotations), total=n_samples):
        if args.using_image_features:
            image_id, image_features, captions = item
            image_features = image_features.float().unsqueeze(dim=0).to(device)
            image_path = resolve_image_path(args, image_id)
        else:
            image_id = item
            captions = annotations[item]
            image_path = resolve_image_path(args, image_id)
            image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
            image_features = encoder.encode_image(image).float()

        continuous_embeddings = compute_continuous_embeddings(
            args, model, image_features, invlm_resources, image_path
        )

        if args.using_hard_prompt:
            embeddings = hard_prompt_embeddings(
                args, model, tokenizer, continuous_embeddings, image_features,
                image_path, device, invlm_resources, entities_text, texts_embeddings,
            )
        else:
            embeddings = continuous_embeddings

        sentence = _decode_caption(args, model, tokenizer, embeddings)

        predicts.append({
            'split': 'valid',
            'image_name': image_id,
            'captions': captions,
            'prediction': sentence,
        })

    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent=4)


def _load_viecap_vocabulary(args, clip_name: str):
    if args.name_of_entities_text == 'visual_genome_entities':
        path = './annotations/vocabulary/all_objects_attributes_relationships.pickle'
        entities_text = load_entities_text(args.name_of_entities_text, path, not args.disable_all_entities)
        emb_path = f'./annotations/vocabulary/visual_genome_embedding_{clip_name}'
    elif args.name_of_entities_text == 'coco_entities':
        path = './annotations/vocabulary/coco_categories.json'
        entities_text = load_entities_text(args.name_of_entities_text, path, not args.disable_all_entities)
        emb_path = f'./annotations/vocabulary/coco_embeddings_{clip_name}'
    elif args.name_of_entities_text == 'open_image_entities':
        path = './annotations/vocabulary/oidv7-class-descriptions-boxable.csv'
        entities_text = load_entities_text(args.name_of_entities_text, path, not args.disable_all_entities)
        emb_path = f'./annotations/vocabulary/open_image_embeddings_{clip_name}'
    elif args.name_of_entities_text == 'vinvl_vg_entities':
        path = './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json'
        entities_text = load_entities_text(args.name_of_entities_text, path, not args.disable_all_entities)
        emb_path = f'./annotations/vocabulary/vg_embeddings_{clip_name}'
    elif args.name_of_entities_text == 'vinvl_vgoi_entities':
        path = './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json'
        entities_text = load_entities_text(args.name_of_entities_text, path, not args.disable_all_entities)
        emb_path = f'./annotations/vocabulary/vgoi_embeddings_{clip_name}'
    else:
        raise ValueError('The entities text should be input correctly!')

    if args.prompt_ensemble:
        emb_path += '_with_ensemble.pickle'
    else:
        emb_path += '.pickle'
    texts_embeddings = clip_texts_embeddings(entities_text, emb_path)
    return entities_text, texts_embeddings


@torch.no_grad()
def main(args) -> None:
    device = torch.device(args.device) if isinstance(args.device, str) else args.device
    args.device = device
    clip_name = args.clip_model.replace('/', '')
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    invlm_resources = None
    entities_text = None
    texts_embeddings = None

    if args.use_ilr and not args.use_meacap_invlm:
        raise ValueError('ILR fusion at inference requires --use_meacap_invlm (memory bank cosine retrieval).')

    if args.use_meacap_invlm:
        mode = 'MeaCap InvLM (memory retrieve-then-filter)'
        if args.use_ilr:
            mode += ' + ILR feature fusion'
        print(f'Validation mode: {mode}')
        invlm_resources = MeaCapInvLMResources(args, device)
    else:
        print('Validation mode: ViECap (CLIP entity classifier)')
        entities_text, texts_embeddings = _load_viecap_vocabulary(args, clip_name)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
        fusion_type=getattr(args, 'fusion_type', 'linear'),
    )
    ckpt = torch.load(args.weight_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'[warn] missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
    model.eval()
    model.to(device)

    encoder = preprocess = None
    if not args.using_image_features:
        encoder, preprocess = clip.load(args.clip_model, device=device)
        inpath = args.path_of_val_datasets
    else:
        inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}.pickle'

    if args.name_of_datasets == 'nocaps':
        validation_nocaps(
            args, inpath, entities_text, texts_embeddings, model, tokenizer,
            preprocess, encoder, invlm_resources,
        )
    else:
        validation_coco_flickr30k(
            args, inpath, entities_text, texts_embeddings, model, tokenizer,
            preprocess, encoder, invlm_resources,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--language_model', default='gpt2')
    parser.add_argument('--continuous_prompt_length', type=int, default=10)
    parser.add_argument('--clip_project_length', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--using_image_features', action='store_true', default=False)
    parser.add_argument('--name_of_datasets', default='coco', choices=('coco', 'flickr30k', 'nocaps'))
    parser.add_argument('--path_of_val_datasets', default='./annotations/coco/val_captions.json')
    parser.add_argument('--disable_all_entities', action='store_true', default=False)
    parser.add_argument(
        '--name_of_entities_text',
        default='coco_entities',
        choices=('visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities'),
    )
    parser.add_argument('--prompt_ensemble', action='store_true', default=False)
    parser.add_argument('--weight_path', default='./checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_folder', default='./annotations/coco/val2014/')
    parser.add_argument('--out_path', default='./generated_captions.json')
    parser.add_argument('--using_hard_prompt', action='store_true', default=False)
    parser.add_argument('--soft_prompt_first', action='store_true', default=False)
    parser.add_argument('--only_hard_prompt', action='store_true', default=False)
    parser.add_argument('--using_greedy_search', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--text_prompt', type=str, default=None)

    # MeaCap InvLM
    parser.add_argument('--use_meacap_invlm', action='store_true', default=False)
    parser.add_argument('--memory_id', type=str, default='coco')
    parser.add_argument('--memory_caption_num', type=int, default=5)
    parser.add_argument('--vl_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--parser_checkpoint', type=str, default='lizhuang144/flan-t5-base-VG-factual-sg')
    parser.add_argument('--wte_model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--local_files_only', action='store_true', default=False)

    # ILR feature fusion (requires --use_meacap_invlm at inference)
    parser.add_argument('--use_ilr', action='store_true', default=False)
    parser.add_argument('--fusion_w1', type=float, default=0.8)
    parser.add_argument('--fusion_w2', type=float, default=0.2)
    parser.add_argument('--fusion_type', default='linear', choices=['linear', 'gated', 'internal_gated'])

    # C3 Collapse (must match training when --remove_mean was used)
    parser.add_argument('--remove_mean', action='store_true', default=False)
    parser.add_argument('--re_normalize_prefix', action='store_true', default=True)
    parser.add_argument('--no_re_normalize_prefix', dest='re_normalize_prefix', action='store_false')
    parser.add_argument('--text_embed_mean_path', default='./annotations/coco/normalized_text_embed_mean.pt')
    parser.add_argument('--image_embed_mean_path', default='./annotations/coco/normalized_image_embed_mean.pt')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', type=int, default=True)

    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))
    main(args)
