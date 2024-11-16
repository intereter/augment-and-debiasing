import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torchvision import transforms
import random
import os
import math
import numpy as np
import pandas as pd
import pickle
import cv2
from modelling.recognition import RecognitionNetwork
from utils.misc import get_logger
from modelling.translation import TranslationNetwork,TranslationNetwork_my
from modelling.translation_ensemble import TranslationNetwork_Ensemble
from modelling.vl_mapper import VLMapper
from modelling.model_my import Discriminator
from dataset.VideoLoader import load_batch_video

class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        if self.task=='S2G':
            self.text_tokenizer = None
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type = 'video',
                transform_cfg=cfg['data']['transform_cfg'],
                input_streams = cfg['data'].get('input_streams','rgb'))
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

            if self.recognition_network.visual_backbone!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())
                
        elif self.task=='G2T':
            self.translation_network = TranslationNetwork(
                input_type='gloss', cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.translation_network.gloss_tokenizer #G2T

        elif self.task=='S2T':
            self.recognition_weight = model_cfg.get('recognition_weight',1)
            self.translation_weight = model_cfg.get('translation_weight',1)
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type = 'feature',
                input_streams = cfg['data'].get('input_streams','rgb'),
                transform_cfg=cfg['data'].get('transform_cfg',{}))
            if model_cfg['RecognitionNetwork'].get('freeze', False)==True:
                self.logger.info('freeze recognition_network')
                self.frozen_modules.append(self.recognition_network)
                for param in self.recognition_network.parameters():
                    param.requires_grad = False
                self.recognition_network.eval()

            input_type = model_cfg['TranslationNetwork'].pop('input_type','feature')
            self.translation_network = TranslationNetwork(
                input_type=input_type, 
                cfg=model_cfg['TranslationNetwork'], 
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer 
            if model_cfg['VLMapper'].get('type','projection') == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features = in_features,
                out_features = self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None), 
            )

        elif self.task=='S2T_Ensemble':
            self.recognition_weight = 0
            self.translation_network = TranslationNetwork_Ensemble(
                cfg=model_cfg['TranslationNetwork_Ensemble']) 
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = None


    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, **kwargs):
        if self.task=='S2G':
            model_outputs = self.recognition_network(is_train=is_train, **recognition_inputs)
            model_outputs['total_loss'] = model_outputs['recognition_loss']            
        elif self.task=='G2T':
            model_outputs = self.translation_network(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
        elif self.task=='S2T':
            recognition_outputs = self.recognition_network(is_train=is_train, **recognition_inputs)
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)
            translation_inputs = {
                **translation_inputs,
                'input_feature':mapped_feature, 
                'input_lengths':recognition_outputs['input_lengths']} 

            translation_outputs = self.translation_network(**translation_inputs)
            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs'] #for latter use of decoding
            model_outputs['total_loss'] = \
                model_outputs['recognition_loss']*self.recognition_weight + \
                model_outputs['translation_loss']*self.translation_weight 
        elif self.task=='S2T_Ensemble':
            assert 'inputs_embeds_list' in translation_inputs and 'attention_mask_list' in translation_inputs
            assert len(translation_inputs['inputs_embeds_list'])==len(self.translation_network.model.model_list)
            model_outputs = self.translation_network(**translation_inputs)
        return model_outputs

    
    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):          
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)  
        return model_outputs
    
    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()
class SignLanguageModel_perturbation_multiHead(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super(SignLanguageModel_perturbation_multiHead, self).__init__()
        self.cfg = cfg
        self.logger = get_logger()
        
        self.task = cfg['task']
        model_cfg = cfg['model']
        self.frozen_modules = []
        self.discriminator = Discriminator(cfg)
        if self.task == 'S2G':
            self.text_tokenizer = None
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type='video',
                transform_cfg=cfg['data']['transform_cfg'],
                input_streams=cfg['data'].get('input_streams', 'rgb'))
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

            if self.recognition_network.visual_backbone != None:
                self.frozen_modules.extend(
                    self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint != None:
                self.frozen_modules.extend(
                    self.recognition_network.visual_backbone_keypoint.get_frozen_layers())

        elif self.task == 'G2T':
            self.translation_network = TranslationNetwork(
                input_type='gloss', cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.translation_network.gloss_tokenizer  

        elif self.task == 'S2T':
            self.recognition_weight = model_cfg.get('recognition_weight', 1)
            self.translation_weight = model_cfg.get('translation_weight', 1)
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type='feature',
                input_streams=cfg['data'].get('input_streams', 'rgb'),
                transform_cfg=cfg['data'].get('transform_cfg', {}),
                ),
            
            if model_cfg['RecognitionNetwork'].get('freeze', False) == True:
                self.logger.info('freeze recognition_network')
                self.frozen_modules.append(self.recognition_network)
                for param in self.recognition_network.parameters():
                    param.requires_grad = False
                self.recognition_network.eval()
            
            input_type = model_cfg['TranslationNetwork'].pop(
                'input_type', 'feature')
            self.translation_network = TranslationNetwork_my(
                input_type=input_type,
                cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
            if model_cfg['VLMapper'].get('type', 'projection') == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features=in_features,
                out_features=self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None),
            )
        elif self.task == 'S2T_Ensemble':
            self.recognition_weight = 0
            self.translation_network = TranslationNetwork_Ensemble(
                cfg=model_cfg['TranslationNetwork_Ensemble'])
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = None

        elif self.task == 'S2T_RGBInput':
            self.recognition_weight = model_cfg.get('recognition_weight', 1)
            self.translation_weight = model_cfg.get('translation_weight', 1)
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type='video',  
                input_streams=cfg['data'].get('input_streams', 'rgb'),
                transform_cfg=cfg['data'].get('transform_cfg', {}),
                mycfg = cfg['model'].get('My',None)
                )
            if model_cfg['RecognitionNetwork'].get('freeze', True):
                self.logger.info('freeze recognition_network')
                self.frozen_modules.append(self.recognition_network)
                for param in self.recognition_network.visual_backbone.parameters():
                    param.requires_grad = False
                self.recognition_network.visual_backbone.eval()

            input_type = model_cfg['TranslationNetwork'].pop(
                'input_type', 'feature')
            self.translation_network = TranslationNetwork_my(
                input_type=input_type,
                cfg=model_cfg['TranslationNetwork'],
                task=self.task,
                mycfg = model_cfg['My'])
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
            if model_cfg['VLMapper'].get('type', 'projection') == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features=in_features,
                out_features=self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None),
            )

    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, **kwargs):
        if self.task == 'S2G':
            model_outputs = self.recognition_network(
                is_train=is_train, **recognition_inputs)
            model_outputs['total_loss'] = model_outputs['recognition_loss']
        elif self.task == 'G2T':
            model_outputs = self.translation_network(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
        elif self.task == 'S2T' or self.task == 'S2T_RGBInput':
            recognition_outputs = self.recognition_network(
                is_train=False, **recognition_inputs)
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)

            translation_inputs = {
                **translation_inputs,
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths']}

            translation_outputs = self.translation_network(
                **translation_inputs)
            translation_encoder_out = translation_outputs['encoder_last_hidden_state']

           
            translation_encoder_mask = translation_outputs['encoder_mask']

            real_input_lenghts = translation_outputs['real_input_lenghts']
            
            translation_decoder_lastLayer_self_att = translation_outputs[
                'decoder_lastLayer_self_att']
        
            translation_decoder_input_ids = translation_outputs['decoder_input_ids']

            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']
            if self.cfg['model'].get('My',None) is not None and self.cfg['model']['My'].get('debias_lm_head',None) is not None:
                model_outputs['total_loss'] = \
                model_outputs['recognition_loss']*self.recognition_weight
                multi_lm_heads = self.translation_network.model.multi_lm_head
                lm_head = self.translation_network.model.lm_head
                l_reg_loss = torch.zeros_like(model_outputs['total_loss']).cuda()
                reg_weight = self.cfg['model']['My']['debias_lm_head'].get('reg_weight',1)
                head_weight = lm_head.weight
                temp_k = head_weight.shape[0]
                for _ in multi_lm_heads:
                    temp_head_weight = _.weight
                    temp_l_reg = temp_head_weight.T @ head_weight
                    temp_l_reg = torch.norm(temp_l_reg)
                    l_reg_loss += temp_l_reg
                l_reg_loss = l_reg_loss / temp_k
                model_outputs['l_reg_loss'] = l_reg_loss
                model_outputs['total_loss'] += l_reg_loss * reg_weight
                    
            else:
                model_outputs['total_loss'] = \
                    model_outputs['recognition_loss']*self.recognition_weight + \
                    model_outputs['translation_loss']*self.translation_weight

            model_outputs['mapped_feature'] = mapped_feature

        elif self.task == 'S2T_Ensemble':
            assert 'inputs_embeds_list' in translation_inputs and 'attention_mask_list' in translation_inputs
            assert len(translation_inputs['inputs_embeds_list']) == len(
                self.translation_network.model.model_list)
            model_outputs = self.translation_network(**translation_inputs)
        return model_outputs

    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
        model_outputs = self.translation_network.generate(
            **transformer_inputs, **generate_cfg)
        return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()


    def get_per_gradReverse(self, batch, data_cfg, names, num_frames):
        self.eval()
        autocast = torch.cuda.amp.autocast
        self.recognition_network.input_type = 'video'
        all_index = range(
            0, batch['translation_inputs']['labels'].shape[0])
        per_index = all_index
        if_keypoint_mask = self.cfg['model']['My'].get('if_keypoint_mask',False)
        if_fp16 = self.cfg['model']['My'].get('if_fp16',False)
        
        names_temp = names
        num_frames_temp = num_frames
        names = []
        num_frames = []
        all_per_grad = []
        all_sgn_videos = []
        all_sgn_lengths = []
        for _ in all_index:
            names = []
            num_frames = []
            names.append(names_temp[_])
            num_frames.append(num_frames_temp[_])
            sgn_videos, sgn_keypoints, sgn_lengths = load_batch_video(
                zip_file=data_cfg['zip_file'],
                names=names,
                num_frames=num_frames,
                transform_cfg=data_cfg['transform_cfg'],
                dataset_name=data_cfg['dataset_name'],
                pad_length=data_cfg.get('pad_length', 'pad_to_max'),
                pad=data_cfg.get('pad', 'replicate'),
                is_train=True,
                name2keypoint=None
            )
        
            all_sgn_lengths.append(sgn_lengths)
            batch_per = {}
            batch_per['translation_inputs'] = {}
            batch_per['recognition_inputs'] = {}
            batch_per['translation_inputs']['labels'] = batch['translation_inputs']['labels'][_].unsqueeze(0)
            batch_per['translation_inputs']['decoder_input_ids'] = batch['translation_inputs']['decoder_input_ids'][_].unsqueeze(0)
            batch_per['translation_inputs']['gloss_ids'] = batch['translation_inputs']['gloss_ids'][_].unsqueeze(0)
            batch_per['translation_inputs']['gloss_lengths'] = batch['translation_inputs']['gloss_lengths'][_].unsqueeze(0)
            batch_per['recognition_inputs']['gls_lengths'] = batch['recognition_inputs']['gls_lengths'][_].unsqueeze(0)
            batch_per['recognition_inputs']['gloss_labels'] = batch['recognition_inputs']['gloss_labels'][_].unsqueeze(0)
            batch_per['recognition_inputs']['sgn_videos'] = sgn_videos.cuda()
            batch_per['recognition_inputs']['sgn_lengths'] = sgn_lengths.cuda()
            
            self.recognition_network.per = True
            with autocast(enabled=if_fp16): 
                output = self.forward(is_train=True, **batch_per)

            output['sgn_videos'].retain_grad()
            all_sgn_videos.append(output['sgn_videos'])
            sgn_ori = output['sgn_ori']
            with torch.autograd.set_detect_anomaly(True):
                output['total_loss'].backward()
            per_grad = output['sgn_videos'].grad
            all_per_grad.append(per_grad)
            origin_logit = output['logits'].detach()
        
        max_sgn_len = 0
        for _ in range(0,len(all_index)):
            max_sgn_len = max(max_sgn_len,all_per_grad[_].shape[2])
        for _ in range(0,len(all_index)):
            if all_per_grad[_].shape[2] < max_sgn_len:
                per_grad_temp = all_per_grad[_]
                per_grad_temp_last = per_grad_temp[:,:,-1,:,:].unsqueeze(2)
                per_grad_temp_last = per_grad_temp_last.repeat(1,1,max_sgn_len - per_grad_temp.shape[2],1,1)
                all_per_grad[_] = torch.cat([per_grad_temp,per_grad_temp_last],dim=2)
                
                sgn_videos_temp = all_sgn_videos[_]
                sgn_videos_temp_last = sgn_videos_temp[:,:,-1,:,:].unsqueeze(2)
                sgn_videos_temp_last = sgn_videos_temp_last.repeat(1,1,max_sgn_len - sgn_videos_temp.shape[2],1,1)
                all_sgn_videos[_] = torch.cat([sgn_videos_temp,sgn_videos_temp_last],dim=2)
    
        all_per_grad = torch.cat(all_per_grad,dim=0)
        all_sgn_videos = torch.cat(all_sgn_videos,dim=0)
        all_sgn_lengths = torch.cat(all_sgn_lengths,dim=0)
        batch_per = {}
        batch_per['translation_inputs'] = {}
        batch_per['recognition_inputs'] = {}
        batch_per['translation_inputs']['labels'] = batch['translation_inputs']['labels']
        batch_per['translation_inputs']['decoder_input_ids'] = batch['translation_inputs']['decoder_input_ids']
        batch_per['translation_inputs']['gloss_ids'] = batch['translation_inputs']['gloss_ids']
        batch_per['translation_inputs']['gloss_lengths'] = batch['translation_inputs']['gloss_lengths']
        batch_per['recognition_inputs']['gls_lengths'] = batch['recognition_inputs']['gls_lengths']
        batch_per['recognition_inputs']['gloss_labels'] = batch['recognition_inputs']['gloss_labels']
        batch_per['recognition_inputs']['sgn_videos'] = all_sgn_videos
        batch_per['recognition_inputs']['sgn_lengths'] = all_sgn_lengths.cuda()

        self.recognition_network.per = False
        self.zero_grad()
        self.recognition_network.per = False
        self.recognition_network.fix_sgn_videos = False
        self.train()

        all_keypoint_mask = None
        if if_keypoint_mask:
            keypoint_index_base_path = "/my/data/keypoint_indexs2/"
            all_keypoint_mask = torch.ones((all_sgn_videos.shape[0],1,all_sgn_videos.shape[2],224,224))
            for index in range(0,len(names)):
                keypoint_index_path = keypoint_index_base_path + names[index] + "/keypoint_index"
                with open(keypoint_index_path,'rb') as f:
                    keypoint_indexs =  pickle.load(f)
                    frame_index = np.arange(len(keypoint_indexs))
                    valid_len = len(keypoint_indexs)
                    for i in range(0,len(frame_index)):
                        keypoint_index = keypoint_indexs[frame_index[i]]
                        keypoint_mask = all_keypoint_mask[index,0,i]
                        keypoint_mask.index_put_(keypoint_index,torch.zeros([1]))
                    if all_sgn_videos[index].shape[1] < all_keypoint_mask.shape[2]:
                        last_keypoint_mask = all_keypoint_mask[index,0,valid_len-1]
                        all_keypoint_mask[index,0,valid_len:] = last_keypoint_mask
            all_keypoint_mask = all_keypoint_mask.cuda()
        return all_per_grad, origin_logit, batch_per, all_sgn_videos.detach(), output ,per_index ,all_keypoint_mask, sgn_ori


def build_model(cfg):
    model = SignLanguageModel_perturbation_multiHead(cfg)
    return model.to(cfg['device'])