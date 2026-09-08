import os
from os.path import join as pjoin
import numpy as np
import torch
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
import utils.eval_t2m_ddp as eval_t2m
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.transformer.transformer_aux import MaskTransformer
from models.transformer.transformer_ts import MaskTransformer2D
from models.vq.model import RVQVAE
from models.rag.ze_retriever import ZeRetriever

def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt, dim_pose, vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'), map_location=opt.device)
    vq_model.load_state_dict(ckpt['vq_model'])
    print(f'Loading VQ Model {vq_opt.name} Completed!', ckpt['ep'], ckpt['value'])
    return (vq_model, vq_opt)

def load_trans_aux(model_opt, which_model):
    transformer = MaskTransformer(code_dim=vq_model.code_dim1d, cond_mode='text', latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size, num_layers=model_opt.n_layers, num_heads=model_opt.n_heads, dropout=model_opt.dropout, clip_dim=512, cond_drop_prob=model_opt.cond_drop_prob, clip_version=clip_version, opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model), map_location=opt.device)
    (missing_keys, unexpected_keys) = transformer.load_state_dict(ckpt['t2m_transformer_aux'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f"Loading Mask Transformer {model_opt.name} from epoch {ckpt['ep']}!", 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else '')
    return transformer

def load_trans_ts(model_opt, which_model, retrieval_dim=None):
    transformer = MaskTransformer2D(code_dim=vq_model.code_dim2d, cond_mode='text', latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size, num_layers=model_opt.n_layers, num_heads=model_opt.n_heads, dropout=model_opt.dropout, clip_dim=512, cond_drop_prob=model_opt.cond_drop_prob, clip_version=clip_version, opt=model_opt, retrieval_dim=retrieval_dim)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model), map_location=opt.device)
    (missing_keys, unexpected_keys) = transformer.load_state_dict(ckpt['t2m_transformer_ts'], strict=False)
    assert len(unexpected_keys) == 0
    assert all((k.startswith('clip_model.') for k in missing_keys)), missing_keys
    print(f"Loading Mask Transformer {model_opt.name} from epoch {ckpt['ep']}!", 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else '')
    return transformer
if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)
    opt.device = torch.device('cpu' if opt.gpu_id == -1 else 'cuda:' + str(opt.gpu_id))
    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.mtrans_name)
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    out_path = pjoin(out_dir, '%s.log' % opt.ext)
    f = open(pjoin(out_path), 'w')
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    print(model_opt_path)
    clip_version = opt.clip_version
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.checkpoints_dir = opt.checkpoints_dir
    vq_opt.dataset_name = opt.dataset_name
    (vq_model, vq_opt) = load_vq_model(vq_opt)
    model_opt.checkpoints_dir = opt.checkpoints_dir
    model_opt.dataset_name = opt.dataset_name
    model_opt.name = opt.mtrans_name
    model_opt.num_tokens1d = vq_model.num_code1d
    model_opt.num_tokens2d = vq_model.num_code2d
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' else 'checkpoints/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, opt.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    retriever = ZeRetriever(database_path=opt.ze_database_path, query_projector_path=opt.projector_path, top_k=opt.retrieval_topk, num_retrieval=opt.retrieval_pool).to(opt.device).eval()
    torch.cuda.empty_cache()
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22
    (eval_val_loader, _) = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)
    retrieval_dim = opt.retrieval_dim
    if retrieval_dim is None:
        retrieval_dim = vq_model.code_dim2d
    print(f'[Config] retrieval_dim={retrieval_dim}, rt_in_value={opt.rt_in_value}')
    file = opt.which_epoch
    print('loading checkpoint {}'.format(file))
    transformer_aux = load_trans_aux(model_opt, file)
    transformer_ts = load_trans_ts(model_opt, file, retrieval_dim=retrieval_dim)
    transformer_aux.eval()
    transformer_ts.eval()
    vq_model.eval()
    transformer_aux.to(opt.device)
    transformer_ts.to(opt.device)
    vq_model.to(opt.device)
    for module in transformer_ts.semanticTransEncoder:
        module.rt_in_value = opt.rt_in_value
        module.vgate = opt.retr_vgate
    transformer_ts.retr_cfgmix_w = opt.retr_cfgmix_w
    retriever._clip_model = transformer_ts.clip_model
    retriever._clip_device = retriever.motion_features.device
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    repeat_time = opt.repeat_times
    for i in range(repeat_time):
        with torch.no_grad():
            (best_fid, best_div, Rprecision, best_matching, best_mm) = eval_t2m.evaluation_mask_transformer_test(eval_val_loader, vq_model, transformer_aux, transformer_ts, i, eval_wrapper=eval_wrapper, time_steps=opt.time_steps, cond_scale=opt.cond_scale, temperature=opt.temperature, topkr=opt.topkr, force_mask=opt.force_mask, cal_mm=True, retriever=retriever)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mm.append(best_mm)
    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    print(f'{file} final result:')
    print(f'{file} final result:', file=f, flush=True)
    msg_final = f'\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n'
    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()
