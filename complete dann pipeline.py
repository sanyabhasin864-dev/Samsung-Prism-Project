import os
import sys
import json
import time
import random
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import torchaudio
import torchaudio.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix,
    balanced_accuracy_score, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class DANNConfig:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_root: str = './data'
    output_dir: str = './dann_results'
    checkpoint_dir: str = './dann_results/checkpoints'
    log_dir: str = './dann_results/logs'
    viz_dir: str = './dann_results/visualizations'
    
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 160
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 8000
    
    max_audio_length: float = 3.0
    max_samples: int = 48000
    
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    
    num_workers: int = 4
    pin_memory: bool = True
    
    gradient_clip: float = 1.0
    use_amp: bool = True
    
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    adversarial_lambda_start: float = 0.0
    adversarial_lambda_end: float = 1.0
    adversarial_gamma: float = 10.0
    
    early_stopping_patience: int = 25
    reduce_lr_patience: int = 10
    min_lr: float = 1e-7
    
    save_top_k: int = 3


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class SincConv(nn.Module):
    def __init__(self, out_channels=64, kernel_size=251, sample_rate=16000):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        low_hz = 30
        high_hz = sample_rate / 2 - 100
        
        mel = torch.linspace(
            2595 * np.log10(1 + low_hz / 700),
            2595 * np.log10(1 + high_hz / 700),
            out_channels + 1
        )
        hz = 700 * (10 ** (mel / 2595) - 1)
        
        self.low_hz_ = nn.Parameter(hz[:-1])
        self.band_hz_ = nn.Parameter(torch.diff(hz))
        
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int((kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size)
        
        n = 2 * np.pi * torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        self.n_ = n / sample_rate
    
    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        
        low = self.low_hz_.clamp(min=50)
        high = (low + self.band_hz_.clamp(min=50))
        
        band = (torch.sin(high.unsqueeze(1) * self.n_.unsqueeze(0)) - 
                torch.sin(low.unsqueeze(1) * self.n_.unsqueeze(0))) / (self.n_.unsqueeze(0) / 2)
        
        band[:, self.kernel_size // 2] = 2 * (high - low)
        
        filters = band * self.window_
        filters = filters / (2 * (high - low).unsqueeze(1))
        
        return F.conv1d(waveforms.unsqueeze(1), filters.unsqueeze(1), 
                       padding=self.kernel_size // 2)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else None
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.sinc = SincConv(out_channels=64, kernel_size=251, sample_rate=config.sample_rate)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            ResBlock(128, 128, stride=1),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512
    
    def forward(self, x):
        x = self.sinc(x)
        x = x.unsqueeze(2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class LabelClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.fc(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512, num_domains=2):
        super().__init__()
        
        self.grl = GradientReversalLayer()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_domains)
        )
    
    def forward(self, x, lambda_=1.0):
        self.grl.set_lambda(lambda_)
        x = self.grl(x)
        return self.fc(x)


class DANN(nn.Module):
    def __init__(self, config, num_domains_dict):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(config)
        self.label_classifier = LabelClassifier(self.feature_extractor.feature_dim)
        
        self.domain_discriminators = nn.ModuleDict()
        for domain_name, num_classes in num_domains_dict.items():
            self.domain_discriminators[domain_name] = DomainDiscriminator(
                self.feature_extractor.feature_dim, num_classes
            )
        
        self.attention = MultiHeadAttention(self.feature_extractor.feature_dim, 8)
        self.layer_norm = nn.LayerNorm(self.feature_extractor.feature_dim)
    
    def forward(self, x, lambda_=1.0, use_attention=True):
        features = self.feature_extractor(x)
        
        if use_attention:
            features_att = self.attention(features.unsqueeze(1))
            features_att = features_att.squeeze(1)
            features = features + features_att
            features = self.layer_norm(features)
        
        class_output = self.label_classifier(features)
        
        domain_outputs = {}
        for domain_name, discriminator in self.domain_discriminators.items():
            domain_outputs[domain_name] = discriminator(features, lambda_)
        
        return class_output, domain_outputs, features


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class AudioDataset(Dataset):
    def __init__(self, df, config, split='train', augment=True):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.sample_rate = config.sample_rate
        self.max_samples = config.max_samples
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        audio_path = row['file_path']
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.shape[1] > self.max_samples:
            start = random.randint(0, waveform.shape[1] - self.max_samples)
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))
        
        if self.augment:
            if random.random() < 0.3:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            
            if random.random() < 0.2:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain
        
        waveform = waveform.squeeze(0)
        
        label = int(row['label'])
        
        domains = {}
        for col in self.df.columns:
            if col.startswith('domain_'):
                domain_name = col.replace('domain_', '')
                domains[domain_name] = int(row[col])
        
        return {
            'waveform': waveform,
            'label': label,
            'domains': domains,
            'file_path': audio_path
        }


def prepare_data(config):
    data_file = Path(config.data_root) / 'processed_data.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    domain_columns = [col for col in df.columns if col.startswith('domain_')]
    
    num_domains_dict = {}
    for col in domain_columns:
        domain_name = col.replace('domain_', '')
        num_classes = df[col].nunique()
        num_domains_dict[domain_name] = num_classes
    
    train_dataset = AudioDataset(train_df, config, 'train', augment=True)
    val_dataset = AudioDataset(val_df, config, 'val', augment=False)
    test_dataset = AudioDataset(test_df, config, 'test', augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader, num_domains_dict


def compute_lambda_p(epoch, total_epochs, gamma=10.0):
    p = float(epoch) / float(total_epochs)
    lambda_ = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    return lambda_


def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'eer': float(eer),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


class Logger:
    def __init__(self, log_dir, name='DANN'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{name}_{timestamp}.log'
        self.metrics_file = self.log_dir / f'metrics_{timestamp}.csv'
        
        self.metrics_buffer = []
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'[{timestamp}] {message}'
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, epoch, metrics, split):
        metrics_dict = {'epoch': epoch, 'split': split}
        metrics_dict.update(metrics)
        self.metrics_buffer.append(metrics_dict)
        
        if len(self.metrics_buffer) >= 10:
            self.flush_metrics()
    
    def flush_metrics(self):
        if self.metrics_buffer:
            df = pd.DataFrame(self.metrics_buffer)
            if not self.metrics_file.exists():
                df.to_csv(self.metrics_file, index=False)
            else:
                df.to_csv(self.metrics_file, mode='a', header=False, index=False)
            self.metrics_buffer = []


class Trainer:
    def __init__(self, model, config, logger, train_loader, val_loader, domain_names):
        self.model = model
        self.config = config
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.domain_names = domain_names
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=True
        )
        
        self.focal_loss = FocalLoss(config.focal_alpha, config.focal_gamma)
        self.domain_loss = nn.CrossEntropyLoss()
        
        self.scaler = GradScaler() if config.use_amp else None
        
        self.best_metric = 0.0
        self.best_models = []
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        lambda_ = compute_lambda_p(epoch, self.config.num_epochs, self.config.adversarial_gamma)
        
        adv_weight = (self.config.adversarial_lambda_start + 
                     (self.config.adversarial_lambda_end - self.config.adversarial_lambda_start) * 
                     (epoch / self.config.num_epochs))
        
        for batch_idx, batch in enumerate(self.train_loader):
            waveforms = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device)
            
            domains_dict = {}
            for domain_name in self.domain_names:
                if domain_name in batch['domains']:
                    domains_dict[domain_name] = batch['domains'][domain_name].to(self.device)
            
            if self.scaler:
                with autocast():
                    class_out, domain_outs, features = self.model(waveforms, lambda_)
                    
                    cls_loss = self.focal_loss(class_out, labels)
                    
                    dom_loss = 0.0
                    for domain_name in self.domain_names:
                        if domain_name in domains_dict and domain_name in domain_outs:
                            dom_loss += self.domain_loss(domain_outs[domain_name], domains_dict[domain_name])
                    
                    loss = cls_loss + adv_weight * dom_loss
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                class_out, domain_outs, features = self.model(waveforms, lambda_)
                
                cls_loss = self.focal_loss(class_out, labels)
                
                dom_loss = 0.0
                for domain_name in self.domain_names:
                    if domain_name in domains_dict and domain_name in domain_outs:
                        dom_loss += self.domain_loss(domain_outs[domain_name], domains_dict[domain_name])
                
                loss = cls_loss + adv_weight * dom_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_class_loss += cls_loss.item()
            total_domain_loss += dom_loss if isinstance(dom_loss, float) else dom_loss.item()
            
            probs = F.softmax(class_out, dim=1)[:, 1]
            preds = torch.argmax(class_out, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_domain_loss = total_domain_loss / len(self.train_loader)
        
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        metrics.update({
            'total_loss': avg_loss,
            'class_loss': avg_class_loss,
            'domain_loss': avg_domain_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        self.logger.log_metrics(epoch, metrics, 'train')
        
        return metrics
    
    def evaluate(self, epoch, loader=None):
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                
                class_out, _, _ = self.model(waveforms, lambda_=0.0)
                
                loss = self.focal_loss(class_out, labels)
                total_loss += loss.item()
                
                probs = F.softmax(class_out, dim=1)[:, 1]
                preds = torch.argmax(class_out, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        metrics['loss'] = avg_loss
        
        self.logger.log_metrics(epoch, metrics, 'val')
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            checkpoint_path = checkpoint_dir / f'best_epoch_{epoch}_acc_{metrics["balanced_accuracy"]:.4f}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            self.best_models.append((metrics['balanced_accuracy'], checkpoint_path))
            self.best_models.sort(key=lambda x: x[0], reverse=True)
            
            if len(self.best_models) > self.config.save_top_k:
                _, path_to_remove = self.best_models.pop()
                if path_to_remove.exists():
                    path_to_remove.unlink()
        
        latest_path = checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def train(self):
        self.logger.log("="*80)
        self.logger.log("Starting DANN Training")
        self.logger.log("="*80)
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.log(f"\nEpoch {epoch}/{self.config.num_epochs}")
            self.logger.log("-"*80)
            
            train_metrics = self.train_epoch(epoch)
            
            self.logger.log(
                f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"Bal Acc: {train_metrics['balanced_accuracy']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}"
            )
            
            val_metrics = self.evaluate(epoch)
            
            self.logger.log(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"Bal Acc: {val_metrics['balanced_accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, "
                f"EER: {val_metrics['eer']:.4f}"
            )
            
            current_metric = val_metrics['balanced_accuracy']
            
            if current_metric > self.best_metric:
                self.logger.log(f"New best balanced accuracy: {current_metric:.4f}")
                self.best_metric = current_metric
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.scheduler.step(current_metric)
            
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.log("Early stopping triggered!")
                break
        
        self.logger.log("\n" + "="*80)
        self.logger.log(f"Training Complete! Best Balanced Accuracy: {self.best_metric:.4f}")
        self.logger.log("="*80)
        
        return {'best_metric': self.best_metric}


class DomainAnalyzer:
    def __init__(self, model, config, logger, test_loader, domain_names):
        self.model = model
        self.config = config
        self.logger = logger
        self.test_loader = test_loader
        self.domain_names = domain_names
        self.device = torch.device(config.device)
    
    def analyze_all_domains(self):
        self.logger.log("="*80)
        self.logger.log("Domain-Specific Analysis")
        self.logger.log("="*80)
        
        self.model.eval()
        
        domain_predictions = defaultdict(lambda: {'labels': [], 'preds': [], 'probs': []})
        
        with torch.no_grad():
            for batch in self.test_loader:
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                
                class_out, _, _ = self.model(waveforms, lambda_=0.0)
                
                probs = F.softmax(class_out, dim=1)[:, 1]
                preds = torch.argmax(class_out, dim=1)
                
                for i in range(len(labels)):
                    for domain_name in self.domain_names:
                        if domain_name in batch['domains']:
                            domain_val = batch['domains'][domain_name][i].item()
                            domain_key = f"{domain_name}_{domain_val}"
                            
                            domain_predictions[domain_key]['labels'].append(labels[i].item())
                            domain_predictions[domain_key]['preds'].append(preds[i].item())
                            domain_predictions[domain_key]['probs'].append(probs[i].item())
        
        results = []
        for domain_key, data in domain_predictions.items():
            if len(data['labels']) < 2:
                continue
            
            metrics = calculate_metrics(
                np.array(data['labels']),
                np.array(data['preds']),
                np.array(data['probs'])
            )
            
            metrics['domain'] = domain_key
            metrics['num_samples'] = len(data['labels'])
            results.append(metrics)
            
            self.logger.log(
                f"{domain_key}: Acc={metrics['accuracy']:.4f}, "
                f"Bal_Acc={metrics['balanced_accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"AUC={metrics['auc']:.4f}, "
                f"Samples={metrics['num_samples']}"
            )
        
        return results
    
    def compute_domain_shifts(self):
        self.model.eval()
        
        domain_features = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.test_loader:
                waveforms = batch['waveform'].to(self.device)
                
                _, _, features = self.model(waveforms, lambda_=0.0)
                
                for i in range(len(features)):
                    for domain_name in self.domain_names:
                        if domain_name in batch['domains']:
                            domain_val = batch['domains'][domain_name][i].item()
                            domain_key = f"{domain_name}_{domain_val}"
                            domain_features[domain_key].append(features[i].cpu().numpy())
        
        domain_centroids = {}
        for domain_key, features_list in domain_features.items():
            if len(features_list) > 0:
                centroid = np.mean(features_list, axis=0)
                domain_centroids[domain_key] = centroid
        
        shifts = {}
        domain_keys = list(domain_centroids.keys())
        for i in range(len(domain_keys)):
            for j in range(i+1, len(domain_keys)):
                key1, key2 = domain_keys[i], domain_keys[j]
                distance = np.linalg.norm(domain_centroids[key1] - domain_centroids[key2])
                shifts[f"{key1}_to_{key2}"] = float(distance)
        
        return shifts


class VisualizationGenerator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bonafide', 'Spoof'],
                   yticklabels=['Bonafide', 'Spoof'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, y_true, y_prob, filename='roc_curve.png'):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"ROC curve saved to {save_path}")
    
    def plot_training_curves(self, metrics_file, filename='training_curves.png'):
        if not Path(metrics_file).exists():
            return
        
        df = pd.read_csv(metrics_file)
        
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].plot(train_df['epoch'], train_df['total_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(val_df['epoch'], val_df['loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(train_df['epoch'], train_df['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(val_df['epoch'], val_df['accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(val_df['epoch'], val_df['f1'], label='F1 Score', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(val_df['epoch'], val_df['auc'], label='AUC', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('AUC', fontsize=12)
        axes[1, 1].set_title('Validation AUC', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"Training curves saved to {save_path}")
    
    def plot_domain_performance(self, domain_results, filename='domain_performance.png'):
        if not domain_results:
            return
        
        df = pd.DataFrame(domain_results)
        df = df.sort_values('balanced_accuracy')
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        axes[0, 0].barh(df['domain'], df['accuracy'], color='steelblue')
        axes[0, 0].set_xlabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Accuracy by Domain', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        axes[0, 1].barh(df['domain'], df['balanced_accuracy'], color='darkgreen')
        axes[0, 1].set_xlabel('Balanced Accuracy', fontsize=12)
        axes[0, 1].set_title('Balanced Accuracy by Domain', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        axes[1, 0].barh(df['domain'], df['f1'], color='coral')
        axes[1, 0].set_xlabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('F1 Score by Domain', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        axes[1, 1].barh(df['domain'], df['auc'], color='purple')
        axes[1, 1].set_xlabel('AUC', fontsize=12)
        axes[1, 1].set_title('AUC by Domain', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"Domain performance plot saved to {save_path}")
    
    def plot_eer_analysis(self, y_true, y_prob, filename='eer_analysis.png'):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, fpr, label='False Positive Rate', linewidth=2)
        plt.plot(thresholds, fnr, label='False Negative Rate', linewidth=2)
        plt.axvline(eer_threshold, color='red', linestyle='--', 
                   label=f'EER Threshold = {eer_threshold:.4f}', linewidth=2)
        plt.axhline(eer, color='red', linestyle='--', 
                   label=f'EER = {eer:.4f}', linewidth=2)
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Error Rate', fontsize=14)
        plt.title('Equal Error Rate Analysis', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"EER analysis plot saved to {save_path}")


class ResultsReporter:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
    
    def generate_summary_report(self, test_metrics, domain_results, training_time):
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DANN MODEL PERFORMANCE SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Training Time: {training_time}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("OVERALL TEST SET PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:          {test_metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Precision:         {test_metrics['precision']:.4f}\n")
            f.write(f"Recall:            {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:          {test_metrics['f1']:.4f}\n")
            f.write(f"AUC:               {test_metrics['auc']:.4f}\n")
            f.write(f"EER:               {test_metrics['eer']:.4f}\n\n")
            
            f.write(f"True Positives:    {test_metrics['tp']}\n")
            f.write(f"True Negatives:    {test_metrics['tn']}\n")
            f.write(f"False Positives:   {test_metrics['fp']}\n")
            f.write(f"False Negatives:   {test_metrics['fn']}\n\n")
            
            if domain_results:
                f.write("-"*80 + "\n")
                f.write("DOMAIN-SPECIFIC PERFORMANCE\n")
                f.write("-"*80 + "\n\n")
                
                for result in sorted(domain_results, key=lambda x: x['balanced_accuracy'], reverse=True):
                    f.write(f"Domain: {result['domain']}\n")
                    f.write(f"  Samples: {result['num_samples']}\n")
                    f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                    f.write(f"  Balanced Accuracy: {result['balanced_accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {result['f1']:.4f}\n")
                    f.write(f"  AUC: {result['auc']:.4f}\n")
                    f.write(f"  EER: {result['eer']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        self.logger.log(f"Summary report saved to {report_path}")
    
    def save_predictions(self, predictions_data, filename='predictions.csv'):
        df = pd.DataFrame(predictions_data)
        save_path = self.output_dir / filename
        df.to_csv(save_path, index=False)
        self.logger.log(f"Predictions saved to {save_path}")
    
    def save_domain_results(self, domain_results, filename='domain_results.csv'):
        if domain_results:
            df = pd.DataFrame(domain_results)
            save_path = self.output_dir / filename
            df.to_csv(save_path, index=False)
            self.logger.log(f"Domain results saved to {save_path}")


class ModelEvaluator:
    def __init__(self, model, config, logger, test_loader):
        self.model = model
        self.config = config
        self.logger = logger
        self.test_loader = test_loader
        self.device = torch.device(config.device)
    
    def evaluate_and_collect_predictions(self):
        self.logger.log("="*80)
        self.logger.log("Evaluating Model on Test Set")
        self.logger.log("="*80)
        
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        all_paths = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                paths = batch['file_path']
                
                class_out, _, _ = self.model(waveforms, lambda_=0.0)
                
                probs = F.softmax(class_out, dim=1)[:, 1]
                preds = torch.argmax(class_out, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(paths)
        
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        self.logger.log("\nTest Set Results:")
        self.logger.log(f"Accuracy:          {metrics['accuracy']:.4f}")
        self.logger.log(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        self.logger.log(f"Precision:         {metrics['precision']:.4f}")
        self.logger.log(f"Recall:            {metrics['recall']:.4f}")
        self.logger.log(f"F1 Score:          {metrics['f1']:.4f}")
        self.logger.log(f"AUC:               {metrics['auc']:.4f}")
        self.logger.log(f"EER:               {metrics['eer']:.4f}")
        
        predictions_data = {
            'file_path': all_paths,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'probability_spoof': all_probs
        }
        
        return metrics, predictions_data, all_labels, all_preds, all_probs


class ThresholdOptimizer:
    def __init__(self, y_true, y_prob, logger):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.logger = logger
    
    def find_optimal_threshold(self, metric='f1'):
        best_threshold = 0.5
        best_score = 0.0
        
        thresholds = np.linspace(0.1, 0.9, 81)
        
        for threshold in thresholds:
            y_pred = (self.y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                _, _, f1, _ = precision_recall_fscore_support(
                    self.y_true, y_pred, average='binary', zero_division=0
                )
                score = f1
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(self.y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(self.y_true, y_pred)
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.logger.log(f"Optimal threshold for {metric}: {best_threshold:.4f} (score: {best_score:.4f})")
        
        return best_threshold, best_score
    
    def evaluate_at_threshold(self, threshold):
        y_pred = (self.y_prob >= threshold).astype(int)
        metrics = calculate_metrics(self.y_true, y_pred, self.y_prob)
        return metrics


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    parser = argparse.ArgumentParser(description='Complete DANN Pipeline for Audio Deepfake Detection')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--output_dir', type=str, default='./dann_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    config = DANNConfig()
    config.data_root = args.data_root
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.seed = args.seed
    config.device = args.device
    config.num_workers = args.num_workers
    
    config.checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
    config.log_dir = os.path.join(config.output_dir, 'logs')
    config.viz_dir = os.path.join(config.output_dir, 'visualizations')
    
    for directory in [config.output_dir, config.checkpoint_dir, config.log_dir, config.viz_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    set_seed(config.seed)
    
    logger = Logger(config.log_dir, name='DANN_Pipeline')
    
    logger.log("")
    logger.log("="*80)
    logger.log("COMPLETE DANN PIPELINE FOR AUDIO DEEPFAKE DETECTION")
    logger.log("="*80)
    logger.log(f"Device: {config.device}")
    logger.log(f"Batch Size: {config.batch_size}")
    logger.log(f"Epochs: {config.num_epochs}")
    logger.log(f"Learning Rate: {config.learning_rate}")
    logger.log(f"Output Directory: {config.output_dir}")
    logger.log("="*80)
    logger.log("")
    
    start_time = time.time()
    
    logger.log("STAGE 1: Data Preparation")
    logger.log("-"*80)
    
    train_loader, val_loader, test_loader, num_domains_dict = prepare_data(config)
    
    logger.log(f"Training samples: {len(train_loader.dataset)}")
    logger.log(f"Validation samples: {len(val_loader.dataset)}")
    logger.log(f"Test samples: {len(test_loader.dataset)}")
    logger.log(f"Number of domains: {len(num_domains_dict)}")
    logger.log(f"Domains: {list(num_domains_dict.keys())}")
    logger.log("")
    
    logger.log("STAGE 2: Model Initialization")
    logger.log("-"*80)
    
    model = DANN(config, num_domains_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    logger.log("")
    
    logger.log("STAGE 3: Model Training")
    logger.log("-"*80)
    
    domain_names = list(num_domains_dict.keys())
    
    trainer = Trainer(
        model=model,
        config=config,
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        domain_names=domain_names
    )
    
    training_results = trainer.train()
    
    training_time = time.time() - start_time
    logger.log(f"\nTraining completed in {format_time(training_time)}")
    logger.log("")
    
    logger.log("STAGE 4: Model Evaluation")
    logger.log("-"*80)
    
    evaluator = ModelEvaluator(model, config, logger, test_loader)
    test_metrics, predictions_data, y_true, y_pred, y_prob = evaluator.evaluate_and_collect_predictions()
    logger.log("")
    
    logger.log("STAGE 5: Threshold Optimization")
    logger.log("-"*80)
    
    threshold_optimizer = ThresholdOptimizer(y_true, y_prob, logger)
    optimal_threshold_f1, _ = threshold_optimizer.find_optimal_threshold('f1')
    optimal_threshold_bal_acc, _ = threshold_optimizer.find_optimal_threshold('balanced_accuracy')
    
    logger.log("\nMetrics at optimal F1 threshold:")
    metrics_at_opt_f1 = threshold_optimizer.evaluate_at_threshold(optimal_threshold_f1)
    logger.log(f"Balanced Accuracy: {metrics_at_opt_f1['balanced_accuracy']:.4f}")
    logger.log(f"F1 Score: {metrics_at_opt_f1['f1']:.4f}")
    logger.log("")
    
    logger.log("STAGE 6: Domain Analysis")
    logger.log("-"*80)
    
    domain_analyzer = DomainAnalyzer(model, config, logger, test_loader, domain_names)
    domain_results = domain_analyzer.analyze_all_domains()
    logger.log("")
    
    logger.log("STAGE 7: Visualization Generation")
    logger.log("-"*80)
    
    viz_generator = VisualizationGenerator(config, logger)
    
    viz_generator.plot_confusion_matrix(y_true, y_pred)
    viz_generator.plot_roc_curve(y_true, y_prob)
    viz_generator.plot_eer_analysis(y_true, y_prob)
    
    metrics_file = logger.metrics_file
    if Path(metrics_file).exists():
        viz_generator.plot_training_curves(metrics_file)
    
    if domain_results:
        viz_generator.plot_domain_performance(domain_results)
    
    logger.log("")
    
    logger.log("STAGE 8: Results Reporting")
    logger.log("-"*80)
    
    reporter = ResultsReporter(config, logger)
    
    total_time = time.time() - start_time
    reporter.generate_summary_report(test_metrics, domain_results, format_time(total_time))
    reporter.save_predictions(predictions_data)
    reporter.save_domain_results(domain_results)
    
    logger.log("")
    logger.log("="*80)
    logger.log("PIPELINE COMPLETED SUCCESSFULLY")
    logger.log("="*80)
    logger.log(f"Total execution time: {format_time(total_time)}")
    logger.log(f"Best validation balanced accuracy: {training_results['best_metric']:.4f}")
    logger.log(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.log(f"Test F1 score: {test_metrics['f1']:.4f}")
    logger.log(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.log(f"Test EER: {test_metrics['eer']:.4f}")
    logger.log("")
    logger.log(f"All results saved to: {config.output_dir}")
    logger.log(f"- Summary report: {config.output_dir}/summary_report.txt")
    logger.log(f"- Predictions: {config.output_dir}/predictions.csv")
    logger.log(f"- Domain results: {config.output_dir}/domain_results.csv")
    logger.log(f"- Visualizations: {config.viz_dir}/")
    logger.log(f"- Checkpoints: {config.checkpoint_dir}/")
    logger.log(f"- Logs: {config.log_dir}/")
    logger.log("="*80)
    
    logger.flush_metrics()


if __name__ == '__main__':
    main()
