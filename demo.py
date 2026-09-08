import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import eval_mask as loaders
from models.rag.ze_retriever import ZeRetriever
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset_name', choices=['humanml3d', 'kit'], default='humanml3d')
    parser.add_argument('--checkpoints_dir', default='logs')
    parser.add_argument('--mtrans_name', default='v2_mtrans_vgate')
    parser.add_argument('--which_epoch', default='net_best_fid_ep0309.tar')
    parser.add_argument('--projector_path', default='logs/query_projector_repair_ep/best_projector.pt')
    parser.add_argument('--ze_database_path', default='database_ze')
    parser.add_argument('--clip_version', default='ViT-B/32')
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument('--text_prompt')
    text.add_argument('--text_path', type=Path)
    parser.add_argument('--motion_length', type=int, default=196)
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10107)
    parser.add_argument('--time_steps', type=int, default=10)
    parser.add_argument('--cond_scale', type=float, default=4.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--topkr', type=float, default=0.9)
    parser.add_argument('--retrieval_topk', type=int, default=1)
    parser.add_argument('--retrieval_pool', type=int, default=10)
    parser.add_argument('--retr_cfgmix_w', type=float, default=0.75)
    parser.add_argument('--rt_in_value', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--retr_vgate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ext', default='demo')
    parser.add_argument('--mp4', action='store_true')
    parser.add_argument('--bvh', action='store_true')
    args = parser.parse_args()
    if args.motion_length % 4 or not 40 <= args.motion_length <= 196:
        parser.error('--motion_length must be a multiple of 4 between 40 and 196')
    if args.bvh and args.dataset_name != 'humanml3d':
        parser.error('--bvh requires humanml3d')
    device = torch.device('cuda', args.gpu_id)
    torch.cuda.set_device(device)
    fixseed(args.seed)
    root = Path(args.checkpoints_dir) / args.dataset_name
    model_opt = get_opt(str(root / args.mtrans_name / 'opt.txt'), device)
    model_opt.checkpoints_dir = args.checkpoints_dir
    model_opt.dataset_name = args.dataset_name
    model_opt.name = args.mtrans_name
    vq_dir = root / model_opt.vq_name
    vq_opt = get_opt(str(vq_dir / 'opt.txt'), device)
    vq_opt.checkpoints_dir = args.checkpoints_dir
    vq_opt.dataset_name = args.dataset_name
    vq_opt.name = model_opt.vq_name
    loaders.opt = SimpleNamespace(device=device)
    loaders.dim_pose = 251 if args.dataset_name == 'kit' else 263
    loaders.clip_version = args.clip_version
    (vq, _) = loaders.load_vq_model(vq_opt)
    loaders.vq_model = vq
    model_opt.num_tokens1d = vq.num_code1d
    model_opt.num_tokens2d = vq.num_code2d
    aux = loaders.load_trans_aux(model_opt, args.which_epoch).to(device).eval()
    spatial = loaders.load_trans_ts(model_opt, args.which_epoch, retrieval_dim=vq.code_dim2d).to(device).eval()
    vq = vq.to(device).eval()
    for module in spatial.semanticTransEncoder:
        module.rt_in_value = args.rt_in_value
        module.vgate = args.retr_vgate
    spatial.retr_cfgmix_w = args.retr_cfgmix_w
    retriever = ZeRetriever(database_path=args.ze_database_path, query_projector_path=args.projector_path, top_k=args.retrieval_topk, num_retrieval=args.retrieval_pool).to(device).eval()
    retriever._clip_model = spatial.clip_model
    retriever._clip_device = retriever.motion_features.device
    mean = np.load(vq_dir / 'meta/mean.npy')
    std = np.load(vq_dir / 'meta/std.npy')
    prompts = [args.text_prompt] if args.text_prompt is not None else [line.strip() for line in args.text_path.read_text().splitlines() if line.strip()]
    output = Path('outputs') / args.ext
    output.mkdir(parents=True, exist_ok=True)
    joints_num = 21 if args.dataset_name == 'kit' else 22
    fps = 12.5 if args.dataset_name == 'kit' else 20
    for (index, prompt) in enumerate(prompts):
        for repeat in range(args.repeat_times):
            fixseed(args.seed + index * args.repeat_times + repeat)
            captions = [prompt]
            lengths = torch.tensor([args.motion_length // 4], device=device, dtype=torch.long)
            with torch.no_grad():
                retrieved = retriever(captions)
                text_embedding = retriever.encode_text(retriever.tokenize(captions).to(device))
                ids_aux = aux.generate(text_embedding, lengths, args.time_steps, args.cond_scale, temperature=args.temperature, topk_filter_thres=args.topkr, force_mask=False)
                ids_spatial = spatial.generate(text_embedding, lengths, args.time_steps, args.cond_scale, temperature=args.temperature, topk_filter_thres=args.topkr, force_mask=False, n_j=6, re_dict=retrieved)
                (_, decoded) = vq.forward_decoder(ids_aux.unsqueeze(-1), ids_spatial.unsqueeze(-1))
                normalized = decoded[0, :args.motion_length].detach().cpu().numpy()
            features = normalized * std + mean
            joints = recover_from_ric(torch.from_numpy(features).float(), joints_num).numpy()
            name = f'{index:03d}_{repeat:02d}'
            np.save(output / f'{name}_features.npy', features)
            np.save(output / f'{name}_joints.npy', joints)
            (output / f'{name}.txt').write_text(prompt + '\n')
            if args.mp4:
                from utils.plot_script import plot_3d_motion
                from utils.paramUtil import kit_kinematic_chain, t2m_kinematic_chain
                chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain
                plot_3d_motion(str(output / f'{name}.mp4'), chain, joints.copy(), title=prompt, fps=fps)
            if args.bvh:
                from visualization.joints2bvh import Joint2BVHConvertor
                Joint2BVHConvertor().convert(joints, str(output / f'{name}.bvh'), iterations=100)
            print(output / name)
if __name__ == '__main__':
    main()
