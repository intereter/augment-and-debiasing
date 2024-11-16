from audioop import mul
import warnings, argparse, os, sys, queue,math
import re
import copy
from queue import PriorityQueue
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
from utils.progressbar import ProgressBar
from dataset.FeatureLoader import load_batch_feature
from ruamel import yaml
import copy
warnings.filterwarnings("ignore")
from utils.misc import (
    load_config,
    make_model_dir,
    make_logger, make_writer, make_wandb,
    set_seed
)
from dataset.Dataloader import build_dataloader
from prediction import evaluation
import wandb
import torchvision
from torchvision import utils as vutils
from dataset.VideoLoader import load_batch_video
from modelling.utils_my import GradReverse,cal_GradReverse_lambda

def apply_spatial_ops(x, spatial_ops_func):
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, 16, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x    

def save_model(model, optimizer, scheduler, output_file, epoch=None, global_step=None, current_score=None):
    base_dir = os.path.dirname(output_file)
    os.makedirs(base_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'global_step':global_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_score': best_score,
        'current_score': current_score,
    }
    torch.save(state, output_file)
    logger.info('Save model state as '+ output_file)
    return output_file

def evaluate_and_save(model, optimizer, scheduler, val_dataloader, cfg, 
        tb_writer, wandb_run=None,
        epoch=None, global_step=None, generate_cfg={}):
    tag = 'epoch_{:02d}'.format(epoch) + '_step_{}'.format(global_step)
    global best_score, ckpt_queue, best_num_ckpt_queue
    eval_results = evaluation(
        model=model, val_dataloader=val_dataloader, cfg=cfg, 
        tb_writer=tb_writer, wandb_run=wandb_run,
        epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
        save_dir=os.path.join(cfg['training']['model_dir'],'validation',tag),
        do_recognition=cfg['task'] not in ['G2T','S2T_glsfree'],
        do_translation=cfg['task']!='S2G',
        if_DDP=False)
    metric = 'bleu4' if '2T' in cfg['task'] else 'wer'
    if metric=='bleu4':
        score = eval_results['bleu']['bleu4']
        best_score = max(best_score, score)
    elif metric=='wer':
        score = eval_results['wer']
        best_score = min(best_score, score)
    logger.info('best_score={:.2f}'.format(best_score))
    ckpt_file = save_model(model=model, optimizer=optimizer, scheduler=scheduler,
        output_file=os.path.join(cfg['training']['model_dir'],'ckpts',tag+'.ckpt'),
        epoch=epoch, global_step=global_step,
        current_score=score)

    best_ckpt_num = cfg['training'].get('bes_ckpt_num',10)
    if best_num_ckpt_queue._qsize() == 0 or best_num_ckpt_queue._qsize() < cfg['training'].get('best_ckpt_num',5):
        best_output_file = os.path.join(cfg['training']['model_dir'],'ckpts','score='+str(score)+'_best.ckpt')
        best_num_ckpt_queue.put((score,best_output_file))
        save_model(model=model, optimizer=optimizer, scheduler=scheduler,
        output_file=best_output_file,
        epoch=epoch, global_step=global_step,
        current_score=score)
    
    else:
        logger.info("best_num_ckpt_queue已满")
        old = best_num_ckpt_queue.get()
        logger.info("最新score:"+str(score))
        logger.info("最小的旧score"+str(old[0]))
        if old[0] <=  score:
            best_output_file = os.path.join(cfg['training']['model_dir'],'ckpts','score='+str(score)+'_best.ckpt')
            best_num_ckpt_queue.put((score,best_output_file))
            save_model(model=model, optimizer=optimizer, scheduler=scheduler,
            output_file=best_output_file,
            epoch=epoch, global_step=global_step,
            current_score=score)
            logger.info("删除"+old[1])
            os.remove(old[1])
        else:
            best_num_ckpt_queue.put(old)

    if ckpt_queue.full():
        to_delete = ckpt_queue.get()
        try:
            os.remove(to_delete)
        except FileNotFoundError:
            logger.warning(
                "Wanted to delete old checkpoint %s but " "file does not exist.",
                to_delete,
            )
    ckpt_queue.put(ckpt_file)        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help='turn on wandb'
    )
    parser.add_argument(
        "--total_epoch",
        default=40,
        type=int,
        help="total epoch"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(seed=cfg["training"].get("random_seed", 42))    
    model_dir = make_model_dir(
        model_dir=cfg['training']['model_dir'], 
        overwrite=cfg['training'].get('overwrite',False),
        if_DDP=False)
    global logger
    logger = make_logger(
        model_dir=model_dir,
        log_file='train.log')
    tb_writer = make_writer(model_dir=model_dir) 
    if args.wandb:
        wandb_run = make_wandb(model_dir=model_dir, cfg=cfg)
    else:
        wandb_run = None
    os.system('cp {} {}/'.format(args.config, model_dir))

    model = build_model(cfg)
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('# Total parameters = {}'.format(total_params))
    logger.info('# Total trainable parameters = {}'.format(total_params_trainable))


    train_dataloader, train_sampler = build_dataloader(cfg, 'train', model.text_tokenizer, model.gloss_tokenizer,if_DDP=False)
    dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', model.text_tokenizer, model.gloss_tokenizer,if_DDP=False)

    discriminator_optimization_cfg = copy.deepcopy(cfg['training']['optimization'])
    optimizer = build_optimizer(config=cfg['training']['optimization'], model=model) 
    if_discriminator = cfg['model']['My'].get('if_discriminator',False)
    scheduler, scheduler_type = build_scheduler(config=cfg['training']['optimization'], optimizer=optimizer)
    
    assert scheduler_type=='epoch'
    start_epoch, total_epoch, global_step = 0, cfg['training']['total_epoch'], 0
    val_unit, val_freq = cfg['training']['validation']['unit'], cfg['training']['validation']['freq']
    global ckpt_queue, best_score, best_num_ckpt_queue
    ckpt_queue = queue.Queue(maxsize=cfg['training']['keep_last_ckpts'])
    best_score = -100 if '2T' in cfg['task'] else 10000 
    best_num_ckpt_queue = PriorityQueue()    
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
    tb_writer = SummaryWriter(log_dir=os.path.join(model_dir,"tensorboard"))
    
    print("start_epoch:",start_epoch)
    print("total_epoch:",total_epoch)
    logger.info("训练开始：epoch:"+str(start_epoch)+",step"+str(global_step))
    
    total_step = total_epoch * math.ceil(train_dataloader.sampler.num_samples / train_dataloader.batch_size)
    model.total_step = total_step
    scaler =  torch.cuda.amp.GradScaler()
    for epoch_no in range(start_epoch, total_epoch):
        
        logger.info('Epoch {}, Training examples {}'.format(epoch_no, len(train_dataloader.dataset)))
        scheduler.step()
        for step, batch in enumerate(train_dataloader):
            model.set_train()
            model.grad_reverse_step = global_step
            output = {}
            batch_temp = {}
            batch_temp['translation_inputs'] = {}
            batch_temp['recognition_inputs'] = {}
            batch_temp['translation_inputs']['labels'] = batch['translation_inputs']['labels'].cuda()
            batch_temp['translation_inputs']['decoder_input_ids'] = batch['translation_inputs']['decoder_input_ids'].cuda()
            batch_temp['translation_inputs']['gloss_ids'] = batch['translation_inputs']['gloss_ids'].cuda()
            batch_temp['translation_inputs']['gloss_lengths'] = batch['translation_inputs']['gloss_lengths'].cuda()
            batch_temp['recognition_inputs']['gls_lengths'] = batch['recognition_inputs']['gls_lengths'].cuda()
            batch_temp['recognition_inputs']['gloss_labels'] = batch['recognition_inputs']['gloss_labels'].cuda()
            batch_temp['recognition_inputs']['head_rgb_input'] = batch['recognition_inputs']['head_rgb_input'].cuda()
            batch_temp['recognition_inputs']['sgn_mask'] = batch['recognition_inputs']['sgn_mask'].cuda()
            batch_temp['recognition_inputs']['sgn_lengths'] = batch['recognition_inputs']['sgn_lengths'].cuda()
           
            
            model.set_train()
            model.grad_reverse_step = global_step
            
            per_grad,origin_logit,batch_origin,sgn_videos_origin,output_origin,per_index,all_keypoint_mask,sgn_ori = model.get_per_gradReverse(batch = batch_temp,data_cfg=cfg['data'],names=batch['name'],num_frames=batch['num_frames'])
            model.recognition_network.fix_sgn_videos = True
            aug_names = []
            aug_num_frames = []
            for _ in per_index:
                aug_names.append(batch['name'][_])
                aug_num_frames.append(batch['num_frames'][_])
            aug_sgn_videos, aug_sgn_keypoints, aug_sgn_lengths = load_batch_video(
            zip_file=cfg['model']['My']['aug_image_path'],
            names=aug_names,
            num_frames=aug_num_frames,
            transform_cfg=cfg['data']['transform_cfg'],
            dataset_name=cfg['data']['dataset_name'],
            pad_length=cfg['data'].get('pad_length', 'pad_to_max'),
            pad=cfg['data'].get('pad', 'replicate'),
            is_train=True,
            name2keypoint=None)
            aug_spatial_ops = []
            aug_spatial_ops.append(torchvision.transforms.CenterCrop(224))
            aug_spatial_ops.append(torchvision.transforms.Resize([224, 224]))
            aug_spatial_ops = torchvision.transforms.Compose(aug_spatial_ops)
            aug_sgn_videos = apply_spatial_ops(aug_sgn_videos, aug_spatial_ops)
            aug_sgn_videos = aug_sgn_videos[:,:,[2,1,0],:,:] 
            aug_sgn_videos = (aug_sgn_videos-0.5)/0.5
            aug_sgn_videos = aug_sgn_videos.permute(0,2,1,3,4).float()
            sgn_videos_origin = aug_sgn_videos.cuda()
            
            sgn_videos_per = sgn_videos_origin + per_grad * cfg['model']['My']['perputation_weight'] * all_keypoint_mask

            batch_per = batch_origin
            batch_per['recognition_inputs']['sgn_videos'] = sgn_videos_per
                    
            if_fp16 = cfg['model']['My'].get('if_fp16',False)
            with autocast(enabled=if_fp16):
                model.recognition_network.input_type = 'video'
                origin_head_rgb_input = batch_temp['recognition_inputs']['head_rgb_input']
                per_features = []
                for _ in batch['recognition_inputs']['head_rgb_input_aug'].cuda():
                    per_features.append(_)

                for i in range(0,len(per_index)):
                    batch_per_temp = {}
                    batch_per_temp['recognition_inputs'] = {}
                    batch_per_temp['recognition_inputs']['gls_lengths'] =  batch_per['recognition_inputs']['gls_lengths'][i].unsqueeze(0)
                    batch_per_temp['recognition_inputs']['gloss_labels'] =  batch_per['recognition_inputs']['gloss_labels'][i].unsqueeze(0)
                    batch_per_temp['recognition_inputs']['sgn_lengths'] =  batch_per['recognition_inputs']['sgn_lengths'][i].unsqueeze(0)
                    batch_per_temp['recognition_inputs']['sgn_videos'] =  batch_per['recognition_inputs']['sgn_videos'][i,:,0:batch_per['recognition_inputs']['sgn_lengths'][i].item()].unsqueeze(0)
                    recognition_per_temp = model.recognition_network(is_train=False, **batch_per_temp['recognition_inputs'])
                    per_features[per_index[i]] = recognition_per_temp['head_rgb_input'].squeeze(0)+1.0e-8
                    del recognition_per_temp
                per_features, _ , _ = load_batch_feature(features=per_features)
                batch_per = batch_temp
                batch_per['recognition_inputs']['head_rgb_input'] = per_features.cuda()
                model.recognition_network.input_type = 'feature'
                output_aug = model(is_train=True, **batch_per)
                logits_aug = output_aug['logits'].float()
                model.recognition_network.fix_sgn_videos = False
                batch_temp['recognition_inputs']['head_rgb_input'] = origin_head_rgb_input
                model.recognition_network.input_type = 'feature'
                output = model(is_train=True, **batch_temp)

                aug_kl_loss_pad = torch.zeros_like(output_aug['logits']).cuda()
                for _ in range(0,output['decoder_input_ids'].shape[0]):
                    one_decoder_input_ids = output['decoder_input_ids'][_]
                    one_decoder_input_ids_len = torch.nonzero(one_decoder_input_ids==2).item()
                    aug_kl_loss_pad[_,0:one_decoder_input_ids_len] = 1
                aug_kl_loss = F.kl_div(F.log_softmax(output_aug['logits'].float(), dim=-1), F.softmax(output['logits'].float(), dim=-1), reduction='none') 
                
                aug_kl_loss = aug_kl_loss * aug_kl_loss_pad
                aug_kl_loss = torch.mean(aug_kl_loss)
                output['total_loss'] += aug_kl_loss * cfg['model']['My']['aug_kl_weight']
                output['aug_kl_loss'] = aug_kl_loss

                output['discriminator_loss'] = torch.tensor(0).cuda()
                z_origin_aug = output['encoder_last_hidden_state']
                z_aug = output_aug['encoder_last_hidden_state']
                discriminator_input_pad_aug = torch.zeros_like(z_aug).cuda()
                for _ in range(0,output['input_lengths'].shape[0]):
                    discriminator_input_pad_aug[_,0:output['input_lengths'][_],:] = 1
                z_origin_aug = z_origin_aug * discriminator_input_pad_aug
                z_origin_aug = torch.sum(z_origin_aug,dim=1)
                z_origin_aug = torch.div(z_origin_aug,output['input_lengths'].unsqueeze(-1))
                target_origin_aug = torch.ones([z_origin_aug.shape[0],1],dtype=torch.int64).cuda()
                z_aug = z_aug * discriminator_input_pad_aug
                z_aug = torch.sum(z_aug,dim=1)
                z_aug = torch.div(z_aug,output['input_lengths'].unsqueeze(-1))
                target_aug = torch.zeros([z_aug.shape[0],1],dtype=torch.int64).cuda()
                
                
                gamma = cfg['model']['My'].get('gamma',10)
                lamb = cal_GradReverse_lambda(step=global_step,total_step=total_step,gamma=gamma)
                output['GradReverse_lambda'] = lamb
                z_aug = GradReverse.apply(z_aug, lamb)
                logit_origin_aug = model.discriminator(z_origin_aug.detach())
                logit_aug = model.discriminator(z_aug)
                origin_aug_discriminator_loss = model.discriminator.loss(logit_origin_aug,target_origin_aug)
                aug_discriminator_loss = model.discriminator.loss(logit_aug,target_aug)
                aug_discriminator_loss_total = (origin_aug_discriminator_loss + aug_discriminator_loss)/2

                output['discriminator_loss'] = aug_discriminator_loss_total
                output['total_loss'] = output['total_loss'] + output['discriminator_loss'] * cfg['model']['My'].get('discriminator_loss_weight',0.01)
                output['total_loss'] += output['multi_translation_loss'] + output['translation_loss']

                        
                
            with torch.autograd.set_detect_anomaly(True):           
                output['total_loss'].backward()
            optimizer.step()
            model.zero_grad()
            

            if tb_writer:
                for k,v in output.items():
                    if '_loss' in k:
                        tb_writer.add_scalar('train/'+k, v, global_step)
                lr = scheduler.optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                if wandb_run!=None:
                    wandb.log({k: v for k,v in output.items() if '_loss' in k})
                    wandb.log({'learning_rate': lr})
            if val_unit=='step' and global_step%val_freq==0 and global_step>0:
                evaluate_and_save(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    val_dataloader=dev_dataloader,
                    cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                    epoch=epoch_no,
                    global_step=global_step,
                    generate_cfg=cfg['training']['validation']['cfg'])
                
                
                for split in ['test']:
                    logger.info('Evaluate on {} set'.format(split))
                    dataloader, sampler = build_dataloader(cfg, split, model.text_tokenizer, model.gloss_tokenizer,if_DDP=False)
                    for _ in range(0,1):
                        generate_cfg = cfg['testing']['cfg']
                        evaluation(model=model, val_dataloader=dataloader, cfg=cfg, 
                                    epoch=epoch_no, global_step=global_step, 
                                    generate_cfg=generate_cfg,
                                    save_dir=os.path.join(model_dir,split),
                                    do_translation=True, do_recognition=True,if_DDP=False)

            global_step += 1
            if pbar:
                pbar(step)
            
            del batch_temp,output, batch
            if_empty = cfg['model']['My'].get('if_empty',True)
            if if_empty:
                torch.cuda.empty_cache()

        if epoch_no == total_epoch-1:
            evaluate_and_save(
                model=model, optimizer=optimizer, scheduler=scheduler,
                val_dataloader=dev_dataloader,
                cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                epoch=epoch_no,
                global_step=global_step,
                generate_cfg=cfg['training']['validation']['cfg'])
         

    logger.info("训练结束：epoch:"+str(epoch_no)+",step"+str(global_step))

    max_score_my = 0
    max_socre_path = ""
    for _ in range(best_num_ckpt_queue._qsize()):
        temp = best_num_ckpt_queue.get()
        if temp[0] > max_score_my:
            max_score_my = temp[0]
            max_socre_path = temp[1]
    load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts',max_socre_path)
    state_dict = torch.load(load_model_path, map_location='cuda')
    model.load_state_dict(state_dict['model_state'])
    epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
    logger.info('Load model ckpt from '+load_model_path)
    do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T' 
    for split in ['dev','test']:
        logger.info('Evaluate on {} set'.format(split))
        dataloader, sampler = build_dataloader(cfg, split, model.text_tokenizer, model.gloss_tokenizer,if_DDP=False)
        for _ in range(0,1):
            generate_cfg = cfg['testing']['cfg']
            evaluation(model=model, val_dataloader=dataloader, cfg=cfg, 
                        epoch=epoch, global_step=global_step, 
                        generate_cfg=generate_cfg,
                        save_dir=os.path.join(model_dir,split),
                        do_translation=do_translation, do_recognition=do_recognition,if_DDP=False)  
  
    if wandb_run!=None:
        wandb_run.finish()    

