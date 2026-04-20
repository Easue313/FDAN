import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from torch.distributions import constraints
from model_cnn import Encoder, Classifier, DomainDiscriminator, Decoder
import itertools
from datetime import datetime
from some_tools import save_training_curves, _logger, rec_show, mmd_loss
from sklearn.metrics import confusion_matrix, accuracy_score
from SDUST_data import (ReadSDUST_S1000_N0, ReadSDUST_S1500_N0, ReadSDUST_S1800_N0, ReadSDUST_S2000_N0, ReadSDUST_S2500_N0, ReadSDUST_S3000_N0,
                        ReadSDUST_S1500_2500_N0, ReadSDUST_S1000_2000_N0, ReadSDUST_S800_1500_N0)
from config_arg import load_args
import matplotlib.pyplot as plt
import itertools
torch.autograd.set_detect_anomaly(True)

def sparsemax(input, dim=-1):
    original_shape = input.shape
    if dim != -1:
        input = input.transpose(dim, -1)
    input_flat = input.reshape(-1, input.shape[-1])  
    D = input_flat.shape[-1]
    z_sorted, _ = torch.sort(input_flat, dim=-1, descending=True)
    cumsum = z_sorted.cumsum(dim=-1) - 1  
    k = torch.arange(1, D + 1, device=input.device, dtype=input.dtype)
    k = k.unsqueeze(0).expand(input_flat.shape[0], -1)  
    threshold = (cumsum - 1) / k.float()  
    greater = threshold >= z_sorted 
    k_max = greater.sum(dim=-1, keepdim=True) 
    tau = threshold.gather(-1, (k_max - 1).clamp(min=0))  
    output_flat = torch.clamp(input_flat - tau, min=0.0)  
    output = output_flat.reshape(original_shape)
    if dim != -1:
        output = output.transpose(dim, -1)
    return output

class FlatSparseAttention(nn.Module):
    def __init__(self, num_features=576, temperature_init=1.5):
        super().__init__()
        self.num_features = num_features
        self.proj = nn.Linear(num_features, num_features)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
    def forward(self, x):
        B, C, L = x.shape
        assert C * L == self.num_features
        x_flat = x.reshape(B, -1)  
        scores = self.proj(x_flat)  
        scores = scores / self.temperature.clamp(min=0.5)
        attn_flat = sparsemax(scores, dim=-1) 
        attn_min = attn_flat.min(dim=1, keepdim=True)[0]
        attn_max = attn_flat.max(dim=1, keepdim=True)[0]
        denominator = (attn_max - attn_min).clamp(min=1e-8)
        attn_flat = (attn_flat - attn_min) / denominator
        attn_map = attn_flat.reshape(B, C, L)  
        return attn_map

class SparseAttention(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_spatial = nn.Linear(seq_len, seq_len)
        self.in_channels = in_channels
        self.seq_len = seq_len
    def forward(self, x):
        B, C, L = x.shape
        x_channel = x.mean(dim=2) 
        channel_score = self.proj(x_channel)  
        channel_attn = sparsemax(channel_score, dim=1).unsqueeze(-1)
        x_spatial = x.mean(dim=1)  
        spatial_score = self.proj_spatial(x_spatial) 
        spatial_attn = sparsemax(spatial_score, dim=1).unsqueeze(1)  
        attn_map = channel_attn * spatial_attn  
        return attn_map

def hsic_loss(z_c, z_d, sigma=1.0):
    B = z_c.size(0)
    z_c = z_c.view(B, -1)
    z_d = z_d.view(B, -1)
    def gaussian_kernel(x, y, sigma):
        xx = x.pow(2).sum(1, keepdim=True)
        yy = y.pow(2).sum(1, keepdim=True)
        xy = torch.mm(x, y.t())
        dist = xx + yy.t() - 2 * xy
        return torch.exp(-dist / (2 * sigma ** 2))
    K_c = gaussian_kernel(z_c, z_c, sigma)
    K_d = gaussian_kernel(z_d, z_d, sigma)
    H = torch.eye(B, device=z_c.device) - 1.0 / B * torch.ones(B, B, device=z_c.device)
    hsic = torch.trace(torch.mm(K_c, H @ K_d @ H)) / ((B - 1) ** 2)
    return hsic

class CFAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_causal = Encoder()   
        self.encoder_domain = Encoder()  
        self.classifier = Classifier(args.num_classes)
        self.decoder = Decoder()
        self.attention = FlatSparseAttention(
            num_features=32 * 18,  
            temperature_init=1.5    
        )
        self.optimizer_main = torch.optim.Adam(
            list(self.encoder_causal.parameters()) +
            list(self.encoder_domain.parameters()) +
            list(self.classifier.parameters()) +
            list(self.decoder.parameters()),
            lr=args.lr
        )
        self.optimizer_att = torch.optim.Adam(
            self.attention.parameters(),
            lr=args.lr1
        )
        self.current_epoch = 0

    def forward(self, x):
        fmap_c, z_c = self.encoder_causal(x) 
        fmap_d, z_d = self.encoder_domain(x)
        f_map = torch.cat([fmap_c, fmap_d], dim=1) 
        x_rec = self.decoder(f_map)
        pred = self.classifier(z_c)
        return z_c, z_d, pred, x_rec, fmap_c
    
    
    def get_attention_map(self, fmap_c):
        return self.attention(fmap_c) 
    
    def transport_augmentation(self, z_c_batch, fmap_c_batch, labels):
        with torch.no_grad():
            attn_map = self.get_attention_map(fmap_c_batch) 
            B, C, L = fmap_c_batch.shape
            device = fmap_c_batch.device
            z_c_reshaped = fmap_c_batch
            aug1_list = []
            aug2_list = []
            for i in range(B):
                y_i = labels[i]
                peer_mask = (labels == y_i) & (torch.arange(B, device=device) != i)
                peers = z_c_reshaped[peer_mask]
                if peers.size(0) == 0:
                    peer_mean = z_c_reshaped[i]
                else:
                    peer_mean = peers.mean(dim=0)
                a = attn_map[i] 
                z_aug1 = a * z_c_reshaped[i] + (1 - a) * peer_mean
                z_aug2 = a * z_c_reshaped[i] + (1 - a) * (2 * z_c_reshaped[i] - peer_mean)
                aug1_list.append(z_aug1)
                aug2_list.append(z_aug2)
            z_aug1_all = torch.stack(aug1_list, dim=0)  
            z_aug2_all = torch.stack(aug2_list, dim=0)  
            z_aug_all = torch.cat([z_aug1_all, z_aug2_all], dim=0)  
            z_aug_flat_all = z_aug_all.flatten(start_dim=1)
            return z_aug_flat_all, z_aug1_all, z_aug2_all 

    def intervention_consistency(self, fmap_c, labels):
        with torch.no_grad():
            attn_map = self.get_attention_map(fmap_c)
            threshold = attn_map.mean() * 0.5  
            mask_low = attn_map < threshold  
        intervened = fmap_c.clone()
        B = fmap_c.size(0)
        for i in range(B):
            y_i = labels[i]
            peer_idx = (labels == y_i) & (torch.arange(B, device=labels.device) != i)
            if peer_idx.any():
                rand_idx = torch.randint(peer_idx.sum(), ())  
                peer = fmap_c[peer_idx][rand_idx]
            else:
                peer = fmap_c[i]  
            intervened[i][mask_low[i]] = peer[mask_low[i]]
        attn_prime = self.get_attention_map(intervened)
        return torch.mean((attn_map - attn_prime) ** 2)
    

    def model_train(self, minibatch_iterator, test_loaders, logger):
        self.to(self.args.device)
        metrics = {
            'total_loss': [], 'cls_loss': [], 'mmd_loss': [], 'ind_loss': [], 'rec_loss': [], 'aug_loss': [], 'att_loss': [],
            'src_acc': [], 'tar_acc': []
        }
        best_tar_acc = 0.0
        best_epoch = 0
        def to_scalar(val):
            return val.item() if hasattr(val, 'item') else float(val)
        def safe_mean(lst):
            return np.mean(lst) if lst else 0.0
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            self.train()
            epoch_losses = {'cls': [], 'mmd': [], 'ind': [], 'rec': [], 'aug': [], 'att': []}
            batch_idx = 0
            steps_per_epoch = 20  
            for _ in range(steps_per_epoch):
                batch_idx += 1
                try:
                    batch = next(minibatch_iterator)
                except StopIteration:
                    minibatch_iterator = balanced_minibatch_generator()
                    batch = next(minibatch_iterator)
                batch_data, batch_label, batch_dlabel = batch
                x = batch_data.to(self.args.device)
                y = batch_label.to(self.args.device)
                dlabel = batch_dlabel.to(self.args.device)
                z_c, z_d, pred, x_rec, fmap_c = self.forward(x)
                cls_loss = F.cross_entropy(pred, y)
                unique_domains = dlabel.unique()
                unique_classes = y.unique()
                mmd_loss_total = torch.tensor(0.0, device=self.args.device)
                pair_class_count = 0
                for di in range(len(unique_domains)):
                    for dj in range(di + 1, len(unique_domains)):
                        domain_i = unique_domains[di]
                        domain_j = unique_domains[dj]
                        mask_i = (dlabel == domain_i)
                        mask_j = (dlabel == domain_j)
                        for c in unique_classes:
                            mask_i_c = mask_i & (y == c)
                            mask_j_c = mask_j & (y == c)
                            if mask_i_c.sum() > 1 and mask_j_c.sum() > 1:
                                mmd_c = mmd_loss(z_c[mask_i_c], z_c[mask_j_c])
                                mmd_loss_total = mmd_loss_total + mmd_c
                                pair_class_count += 1
                if pair_class_count > 0:
                    mmd_loss_total = mmd_loss_total / pair_class_count
                else:
                    mmd_loss_total = torch.tensor(0.0, device=self.args.device)
                ind_loss = hsic_loss(z_c, z_d)
                x_normalized = (x - x.mean(dim=(1,2), keepdim=True)) / (x.std(dim=(1,2), keepdim=True) + 1e-6)
                x_rec_normalized = (x_rec - x_rec.mean(dim=(1,2), keepdim=True)) / (x_rec.std(dim=(1,2), keepdim=True) + 1e-6)
                rec_loss = F.mse_loss(x_rec_normalized, x_normalized)
                total_main_loss = (cls_loss +
                                   self.args.w_ind * ind_loss +
                                   self.args.w_rec * rec_loss +
                                   self.args.w_mmd * mmd_loss_total)
                aug_loss = torch.tensor(0.0, device=self.args.device)
                att_loss = torch.tensor(0.0, device=self.args.device)
                if epoch >= self.args.warmup_epochs:
                    self.optimizer_att.zero_grad()
                    attn_map = self.attention(fmap_c.detach())
                    attended_feat = attn_map * fmap_c.detach()
                    attended_flat = attended_feat.flatten(start_dim=1)
                    att_pred = self.classifier(attended_flat)
                    att_ce = F.cross_entropy(att_pred, y)
                    att_consistency_loss = torch.tensor(0.0, device=self.args.device)
                    if epoch >= self.args.att_start_epoch:
                        att_consistency_loss = self.intervention_consistency(fmap_c.detach(), y)
                        att_loss = att_ce + self.args.w_consistency * att_consistency_loss
                        att_loss.backward()
                        self.optimizer_att.step()
                if epoch >= self.args.aug_start_epoch:
                    z_aug_flat_all, z_aug1_batch, z_aug2_batch = self.transport_augmentation(
                        z_c.detach(), fmap_c.detach(), y
                    )
                    pred_aug_all = self.classifier(z_aug_flat_all)
                    y_repeated = y.repeat(2)
                    aug_loss = F.cross_entropy(pred_aug_all, y_repeated)
                    total_main_loss = total_main_loss + self.args.w_aug * aug_loss
                self.optimizer_main.zero_grad()
                total_main_loss.backward()
                self.optimizer_main.step()
                epoch_losses['cls'].append(to_scalar(cls_loss))
                epoch_losses['mmd'].append(to_scalar(mmd_loss_total))
                epoch_losses['ind'].append(to_scalar(ind_loss))
                epoch_losses['rec'].append(to_scalar(rec_loss))
                epoch_losses['aug'].append(to_scalar(aug_loss))
                epoch_losses['att'].append(to_scalar(att_loss))
            avg_cls = safe_mean(epoch_losses['cls'])
            avg_mmd = safe_mean(epoch_losses['mmd'])
            avg_ind = safe_mean(epoch_losses['ind'])
            avg_rec = safe_mean(epoch_losses['rec'])
            avg_aug = safe_mean(epoch_losses['aug'])
            avg_att = safe_mean(epoch_losses['att'])
            log_str = (f'Epoch {epoch + 1} | '
                       f'Cls: {avg_cls:.4f} | '
                       f'MMD: {avg_mmd:.4f} | '
                       f'Ind: {avg_ind:.4f} | '
                       f'Rec: {avg_rec:.4f} | '
                       f'Aug: {avg_aug:.4f} | '
                       f'Att: {avg_att:.4f}')
            test_acc = self.model_test(test_loaders, logger)
            tar_acc = test_acc[0]
            src_acc_mean = np.mean(test_acc[1:]) if len(test_acc) > 1 else 0.0
            log_str += f' | Tar Acc: {tar_acc:.4f} | Src Mean Acc: {src_acc_mean:.4f}'
            print(log_str)
            logger.info(log_str)
            total_epoch_loss = avg_cls + self.args.w_mmd * avg_mmd + self.args.w_ind * avg_ind + self.args.w_rec * avg_rec + self.args.w_aug * avg_aug
            metrics['total_loss'].append(total_epoch_loss)
            metrics['cls_loss'].append(avg_cls)
            metrics['mmd_loss'].append(avg_mmd)
            metrics['ind_loss'].append(avg_ind)
            metrics['rec_loss'].append(avg_rec)
            metrics['aug_loss'].append(avg_aug)
            metrics['att_loss'].append(avg_att)
            metrics['tar_acc'].append(tar_acc)
            metrics['src_acc'].append(src_acc_mean)
            if tar_acc > best_tar_acc:
                best_tar_acc = tar_acc
                best_epoch = epoch + 1
                torch.save(self.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth'))
        save_training_curves(metrics, self.args.save_dir)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(self.args.save_dir, 'training_metrics.csv'), index=False)
        return best_epoch, best_tar_acc

    def model_test(self, loaders, logger):
        self.eval()
        acc_results = []
        for data_loader in loaders:
            pred_lst, labels_lst = [], []
            with torch.no_grad():
                for batch_data, batch_label, _ in data_loader:
                    _, z_c, pred, _, _ = self.forward(batch_data.to(self.args.device))
                    pred_lst.extend(pred.argmax(1).cpu().numpy())
                    labels_lst.extend(batch_label.numpy())
            acc = accuracy_score(labels_lst, pred_lst)
            acc_results.append(acc)
        return acc_results

#config文件训练
def main_core(args):
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(path_log, time_str)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = _logger(os.path.join(args.save_dir, 'training.log'))
    for k, v in args.__dict__.items():
        logger.info("{}: {}".format(k, v))
    datasets_object_src = [eval('Read' + i + '(args)') for i in args.src_dataset]
    loaders_src = [i.read_data_file() for i in datasets_object_src]
    train_loaders_src = [train for train, _ in loaders_src]
    test_loaders_src = [test for _, test in loaders_src]
    datasets_object_tar = [eval('Read' + i + '(args)') for i in args.tar_dataset]
    loaders_tar = [i.read_data_file() for i in datasets_object_tar]
    test_loaders_tar = [test for _, test in loaders_tar]
    import itertools
    train_iters_src = [itertools.cycle(loader) for loader in train_loaders_src]

    def balanced_minibatch_generator():
        num_sources = len(train_loaders_src)
        if num_sources == 0:
            raise ValueError("没有源域数据！")
        base_bs = args.batch_size // num_sources          
        remainder = args.batch_size % num_sources      
        while True:
            batch_parts = []
            for domain_idx in range(num_sources):
                current_bs = base_bs + (1 if domain_idx == num_sources - 1 and remainder > 0 else 0)
                if current_bs <= 0:
                    continue  
                loader_iter = train_iters_src[domain_idx]
                try:
                    source_batch = next(loader_iter)
                except StopIteration:
                    train_iters_src[domain_idx] = itertools.cycle(train_loaders_src[domain_idx])
                    source_batch = next(train_iters_src[domain_idx])
                src_data, src_label, src_dlabel = source_batch
                if src_data.size(0) >= current_bs:
                    indices = torch.randperm(src_data.size(0))[:current_bs]
                else:
                    repeat_num = (current_bs + src_data.size(0) - 1) // src_data.size(0)
                    indices = torch.arange(src_data.size(0)).repeat(repeat_num)[:current_bs]
                sampled_data   = src_data[indices]
                sampled_label  = src_label[indices]
                sampled_dlabel = src_dlabel[indices]
                batch_parts.append((sampled_data, sampled_label, sampled_dlabel))
            combined_data   = torch.cat([p[0] for p in batch_parts], dim=0)
            combined_label  = torch.cat([p[1] for p in batch_parts], dim=0)
            combined_dlabel = torch.cat([p[2] for p in batch_parts], dim=0)
            perm = torch.randperm(combined_data.size(0))
            combined_data   = combined_data[perm]
            combined_label  = combined_label[perm]
            combined_dlabel = combined_dlabel[perm]
            yield combined_data, combined_label, combined_dlabel
    model = CFAN(args)
    minibatch_iterator = balanced_minibatch_generator()
    best_epoch, best_tar_acc = model.model_train(
        minibatch_iterator,
        test_loaders_tar + test_loaders_src,  # 目标域在前，用于取 tar_acc = test_acc[0]
        logger
    )
    return best_epoch, None, best_tar_acc, None, None

def main_cycle_part(args):
    domain_pairs = [
        {"target": "SDUST_S800_1500_N0", "sources": ["SDUST_S1000_N0", "SDUST_S1500_N0"]},
        {"target": "SDUST_S1000_2000_N0",
          "sources": ["SDUST_S1000_N0", "SDUST_S1500_N0", "SDUST_S1800_N0", "SDUST_S2000_N0"]},
        {"target": "SDUST_S1500_2500_N0",
         "sources": ["SDUST_S1500_N0", "SDUST_S1800_N0", "SDUST_S2000_N0", "SDUST_S2500_N0"]},
        {"target": "SDUST_S1500_N0", "sources": ["SDUST_S1000_2000_N0", "SDUST_S1500_2500_N0"]},
        {"target": "SDUST_S2000_N0", "sources": ["SDUST_S1000_2000_N0", "SDUST_S1500_2500_N0"]}
    ]
    target_label_mapping = {
        'SDUST_S1000_N0': 0,
        'SDUST_S1500_N0': 1,
        'SDUST_S1800_N0': 2,
        'SDUST_S2000_N0': 3,
        'SDUST_S2500_N0': 4,
        'SDUST_S3000_N0': 5,
        'SDUST_S800_1500_N0': 6,
        'SDUST_S1000_2000_N0': 7,
        'SDUST_S1500_2500_N0': 8,
    }
    results = []
    for pair in domain_pairs:
        args.tar_dataset = [pair["target"]]
        args.src_dataset = pair["sources"]
        args.target_domain_label = target_label_mapping[pair["target"]]
        for repeat_idx in range(args.repeat):
            best_epoch, _, best_tar_acc, _, _ = main_core(args)
            results.append({
                "Target": args.tar_dataset[0],
                "Sources": str(args.src_dataset),  # list 转 str 方便保存
                "Best Target Accuracy": round(best_tar_acc, 4),
                "Best Epoch": best_epoch + 1  # epoch 从0开始，显示+1更直观
            })
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.save_dir, 'results.xlsx')
    results_df.to_excel(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    args = load_args()
    path_log = r'./log_files'
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    main_cycle_part(args)
