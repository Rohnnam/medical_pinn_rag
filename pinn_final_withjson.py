import os, glob, warnings, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import torch
import json

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121, UNet
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose, RandRotated, RandFlipd, RandAffined,
    RandGaussianNoised, RandScaleIntensityd,
    RandShiftIntensityd, ToTensord, RandSpatialCropd,
    RandGridDistortiond
)
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"   Free VRAM: {free_gb:.1f} GB")
torch.backends.cudnn.benchmark = True

# Configuration
ROOT = Path.cwd()
DATA_ROOT = ROOT / "data"
FIGSHARE_DIR = DATA_ROOT / "figshare_brain_tumor"
BRATS_DIR = Path(r"C:\Users\Rohan Nambiar\Downloads\archive(2)\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
SAVED_MODELS_DIR = Path(r"C:\Users\Rohan Nambiar\Documents\Vscode\saved_models")
IMG_SZ = 128
BATCH_2D = 16
EPOCHS_CLS = 15
LR_CLS = 1e-4
NUM_CLS = 3
VOL_SZ = (128, 128, 128)
BATCH_3D = 2
EPOCHS_SEG = 30
LR_SEG = 2e-4
NUM_SEG = 2
PINN_STEPS = 20
PINN_ITERS = 600
PINN_LR = 3e-3
DT = 0.05

# Create saved models directory if it doesn't exist
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

class BrainAtlas:
    """
    Brain anatomical atlas for region identification.
    Uses simplified AAL (Automated Anatomical Labeling) regions mapped to 2D coordinates.
    """
    
    def __init__(self, img_size=128):
        self.img_size = img_size
        self.regions = self._create_anatomical_map()
    
    def _create_anatomical_map(self):
        """
        Create a simplified anatomical region map for axial brain slices.
        This is a simplified version - in practice, you'd use a real atlas like AAL or Harvard-Oxford.
        """
        regions = {}
        
        # Define anatomical regions with approximate boundaries (normalized coordinates 0-1)
        # These are rough approximations for axial slices around the AC-PC line
        
        # Frontal regions
        regions['frontal_left'] = {
            'bounds': [(0.55, 1.0), (0.0, 0.45)],  # [x_range, y_range]
            'name': 'Left Frontal Lobe',
            'subregions': {
                (0.55, 0.75): 'Left Superior Frontal Gyrus',
                (0.75, 0.85): 'Left Middle Frontal Gyrus',
                (0.85, 1.0): 'Left Inferior Frontal Gyrus'
            }
        }
        
        regions['frontal_right'] = {
            'bounds': [(0.0, 0.45), (0.0, 0.45)],
            'name': 'Right Frontal Lobe',
            'subregions': {
                (0.0, 0.25): 'Right Superior Frontal Gyrus',
                (0.25, 0.35): 'Right Middle Frontal Gyrus',
                (0.35, 0.45): 'Right Inferior Frontal Gyrus'
            }
        }
        
        # Parietal regions
        regions['parietal_left'] = {
            'bounds': [(0.55, 1.0), (0.45, 0.75)],
            'name': 'Left Parietal Lobe',
            'subregions': {
                (0.55, 0.75): 'Left Superior Parietal Lobule',
                (0.75, 1.0): 'Left Inferior Parietal Lobule'
            }
        }
        
        regions['parietal_right'] = {
            'bounds': [(0.0, 0.45), (0.45, 0.75)],
            'name': 'Right Parietal Lobe',
            'subregions': {
                (0.0, 0.25): 'Right Superior Parietal Lobule',
                (0.25, 0.45): 'Right Inferior Parietal Lobule'
            }
        }
        
        # Temporal regions
        regions['temporal_left'] = {
            'bounds': [(0.55, 1.0), (0.75, 1.0)],
            'name': 'Left Temporal Lobe',
            'subregions': {
                (0.55, 0.75): 'Left Superior Temporal Gyrus',
                (0.75, 0.85): 'Left Middle Temporal Gyrus',
                (0.85, 1.0): 'Left Inferior Temporal Gyrus'
            }
        }
        
        regions['temporal_right'] = {
            'bounds': [(0.0, 0.45), (0.75, 1.0)],
            'name': 'Right Temporal Lobe',
            'subregions': {
                (0.0, 0.15): 'Right Superior Temporal Gyrus',
                (0.15, 0.3): 'Right Middle Temporal Gyrus',
                (0.3, 0.45): 'Right Inferior Temporal Gyrus'
            }
        }
        
        # Occipital regions
        regions['occipital_left'] = {
            'bounds': [(0.45, 0.55), (0.6, 1.0)],
            'name': 'Left Occipital Lobe',
            'subregions': {
                (0.45, 0.55): 'Left Occipital Cortex'
            }
        }
        
        regions['occipital_right'] = {
            'bounds': [(0.45, 0.55), (0.0, 0.4)],
            'name': 'Right Occipital Lobe',
            'subregions': {
                (0.45, 0.55): 'Right Occipital Cortex'
            }
        }
        
        # Central regions
        regions['central_left'] = {
            'bounds': [(0.45, 0.55), (0.45, 0.6)],
            'name': 'Left Central Region',
            'subregions': {
                (0.45, 0.55): 'Left Precentral/Postcentral Gyrus'
            }
        }
        
        regions['central_right'] = {
            'bounds': [(0.45, 0.55), (0.4, 0.45)],
            'name': 'Right Central Region',
            'subregions': {
                (0.45, 0.55): 'Right Precentral/Postcentral Gyrus'
            }
        }
        
        # Deep structures
        regions['deep_left'] = {
            'bounds': [(0.4, 0.6), (0.5, 0.7)],
            'name': 'Left Deep Structures',
            'subregions': {
                (0.45, 0.55): 'Left Basal Ganglia/Thalamus'
            }
        }
        
        regions['deep_right'] = {
            'bounds': [(0.4, 0.6), (0.3, 0.5)],
            'name': 'Right Deep Structures',
            'subregions': {
                (0.45, 0.55): 'Right Basal Ganglia/Thalamus'
            }
        }
        
        return regions
    
    def pixel_to_normalized(self, x_pixel, y_pixel):
        """Convert pixel coordinates to normalized coordinates (0-1)"""
        x_norm = x_pixel / self.img_size
        y_norm = y_pixel / self.img_size
        return x_norm, y_norm
    
    def get_anatomical_region(self, x_pixel, y_pixel, detailed=True):
        """
        Get anatomical region for given pixel coordinates.
        
        Args:
            x_pixel: X coordinate in pixels
            y_pixel: Y coordinate in pixels  
            detailed: If True, return detailed subregion, else main region
            
        Returns:
            dict: Contains region info, coordinates, and confidence
        """
        x_norm, y_norm = self.pixel_to_normalized(x_pixel, y_pixel)
        
        # Find matching region
        for region_key, region_data in self.regions.items():
            x_range, y_range = region_data['bounds']
            if (x_range[0] <= x_norm <= x_range[1] and 
                y_range[0] <= y_norm <= y_range[1]):
                
                result = {
                    'main_region': region_data['name'],
                    'region_key': region_key,
                    'coordinates': {
                        'pixel': {'x': x_pixel, 'y': y_pixel},
                        'normalized': {'x': x_norm, 'y': y_norm}
                    },
                    'hemisphere': 'Left' if 'left' in region_key else 'Right',
                    'confidence': 0.8  # Base confidence for main regions
                }
                
                if detailed and 'subregions' in region_data:
                    # Find specific subregion
                    for (x_sub_start, x_sub_end), subregion_name in region_data['subregions'].items():
                        if x_sub_start <= x_norm <= x_sub_end:
                            result['detailed_region'] = subregion_name
                            result['confidence'] = 0.9
                            break
                    
                    if 'detailed_region' not in result:
                        result['detailed_region'] = region_data['name']
                
                return result
        
        # Default fallback if no region matched
        hemisphere = 'Left' if x_norm > 0.5 else 'Right'
        return {
            'main_region': f'{hemisphere} Hemisphere (Unspecified)',
            'detailed_region': f'{hemisphere} Hemisphere (Unspecified)',
            'region_key': 'unknown',
            'coordinates': {
                'pixel': {'x': x_pixel, 'y': y_pixel},
                'normalized': {'x': x_norm, 'y': y_norm}
            },
            'hemisphere': hemisphere,
            'confidence': 0.3
        }
    
    def get_region_from_mask(self, mask, method='centroid'):
        """
        Get anatomical region from binary mask.
        
        Args:
            mask: Binary mask (numpy array or torch tensor)
            method: 'centroid', 'largest_component', or 'weighted_average'
            
        Returns:
            dict: Anatomical region information
        """
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        if mask.sum() == 0:
            return {
                'main_region': 'No tumor detected',
                'detailed_region': 'No tumor detected',
                'confidence': 0.0,
                'coordinates': {'pixel': {'x': 0, 'y': 0}, 'normalized': {'x': 0, 'y': 0}},
                'hemisphere': 'Unknown'
            }
        
        if method == 'centroid':
            # Simple centroid
            y_coords, x_coords = np.where(mask > 0)
            center_x = float(np.mean(x_coords))
            center_y = float(np.mean(y_coords))
            
        elif method == 'largest_component':
            # Find largest connected component and use its centroid
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask)
            if num_features == 0:
                center_x = center_y = 0
            else:
                sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                largest_component = (labeled == largest_label)
                y_coords, x_coords = np.where(largest_component)
                center_x = float(np.mean(x_coords))
                center_y = float(np.mean(y_coords))
                
        elif method == 'weighted_average':
            # Weighted average based on intensity values
            y_coords, x_coords = np.where(mask > 0)
            weights = mask[y_coords, x_coords]
            center_x = float(np.average(x_coords, weights=weights))
            center_y = float(np.average(y_coords, weights=weights))
        
        # Get anatomical region
        region_info = self.get_anatomical_region(center_x, center_y, detailed=True)
        
        # Add mask statistics
        region_info['mask_stats'] = {
            'area_pixels': float(mask.sum()),
            'method_used': method,
            'bounding_box': self._get_bounding_box(mask)
        }
        
        return region_info
    
    def _get_bounding_box(self, mask):
        """Get bounding box of the mask"""
        if mask.sum() == 0:
            return [0, 0, 0, 0]
        
        y_coords, x_coords = np.where(mask > 0)
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        return [min_x, min_y, max_x, max_y]
    
    def get_clinical_description(self, region_info):
        """
        Generate clinical description of the anatomical location.
        """
        main_region = region_info['main_region']
        detailed_region = region_info.get('detailed_region', main_region)
        hemisphere = region_info['hemisphere']
        confidence = region_info['confidence']
        
        # Clinical implications based on region
        clinical_notes = {
            'frontal': {
                'functions': ['motor control', 'executive function', 'personality', 'language (if dominant hemisphere)'],
                'clinical_significance': 'May affect motor function, behavior, and cognition'
            },
            'parietal': {
                'functions': ['sensory processing', 'spatial awareness', 'language comprehension'],
                'clinical_significance': 'May affect sensory perception and spatial processing'
            },
            'temporal': {
                'functions': ['memory', 'language', 'auditory processing'],
                'clinical_significance': 'May affect memory formation and language function'
            },
            'occipital': {
                'functions': ['visual processing'],
                'clinical_significance': 'May affect vision and visual perception'
            },
            'central': {
                'functions': ['motor and sensory function'],
                'clinical_significance': 'May affect movement and sensation'
            },
            'deep': {
                'functions': ['motor control', 'cognition', 'emotion regulation'],
                'clinical_significance': 'May affect movement, cognition, and emotional regulation'
            }
        }
        
        # Determine region type
        region_type = None
        for key in clinical_notes.keys():
            if key in main_region.lower():
                region_type = key
                break
        
        if region_type is None:
            region_type = 'unknown'
        
        description = {
            'anatomical_location': detailed_region,
            'hemisphere': hemisphere,
            'confidence_score': confidence,
            'affected_functions': clinical_notes.get(region_type, {}).get('functions', ['Unknown']),
            'clinical_significance': clinical_notes.get(region_type, {}).get('clinical_significance', 'Clinical significance unknown'),
            'laterality_notes': self._get_laterality_notes(hemisphere, region_type)
        }
        
        return description
    
    def _get_laterality_notes(self, hemisphere, region_type):
        """Add notes about hemisphere-specific functions"""
        if hemisphere == 'Left':
            if region_type in ['frontal', 'temporal', 'parietal']:
                return "Typically dominant hemisphere for language function in right-handed individuals"
            else:
                return "Left hemisphere involvement"
        elif hemisphere == 'Right':
            if region_type in ['frontal', 'parietal']:
                return "Typically dominant for spatial processing and attention"
            else:
                return "Right hemisphere involvement"
        return "Hemisphere laterality unclear"

def load_figshare_2d(path: Path, max_samples=None):
    mats = sorted(path.glob("**/*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {path}")
    X, Ymask, Ycls = [], [], []
    skipped = 0
    for p in mats:
        try:
            with h5py.File(p, "r") as f:
                cj = f["cjdata"] if "cjdata" in f else f[list(f.keys())[0]]
                def read_key(g, k):
                    if k not in g: raise KeyError(k)
                    return np.array(g[k])
                img = read_key(cj, "image")
                msk = read_key(cj, "tumorMask")
                lab = int(np.array(cj["label"]).flatten()[0]) - 1
            if img.ndim == 3:
                img = img[:, :, img.shape[-1]//2]
            if msk.ndim == 3:
                msk = msk[:, :, msk.shape[-1]//2]
            img = np.array(img, dtype=np.float32).T
            msk = np.array(msk, dtype=np.uint8).T
            ir = resize(img, (IMG_SZ, IMG_SZ), preserve_range=True, anti_aliasing=True).astype(np.float32)
            mi, ma = ir.min(), ir.max()
            if ma - mi < 1e-6:
                skipped += 1
                continue
            ir = (ir - mi) / (ma - mi)
            mr = resize(msk, (IMG_SZ, IMG_SZ), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            mb = (mr > 0).astype(np.uint8)
            X.append(ir[None, ...])
            Ymask.append(mb)
            Ycls.append(lab)
        except Exception as e:
            print(f"❌ {p.name}: {e}")
            skipped += 1
    if not X:
        raise RuntimeError("No usable Figshare samples")
    X = np.stack(X).astype(np.float32)
    Ymask = np.stack(Ymask).astype(np.int64)
    Ycls = np.array(Ycls, dtype=np.int64)
    if max_samples and len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, Ymask, Ycls = X[idx], Ymask[idx], Ycls[idx]
    cnt = dict(Counter(Ycls.tolist()))
    print(f"📊 Figshare class distribution: {cnt}")
    return X, Ymask, Ycls

class FigshareClsDS(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y = X, y
        self.augment = augment
        if augment:
            self.transform = Compose([
                RandRotated(keys=["image"], range_x=0.3, prob=0.5, mode="bilinear", padding_mode="zeros"),
                RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
                RandAffined(keys=["image"], rotate_range=0.2, scale_range=0.15, translate_range=10, prob=0.6),
                RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
                RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.4),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
                RandGridDistortiond(keys=["image"], num_cells=5, distort_limit=0.2, prob=0.3),
            ])
    def __len__(self): return len(self.X)
   
    def __getitem__(self, i):
        data = {
            "image": torch.from_numpy(self.X[i]).float(),
            "label": torch.tensor(int(self.y[i]), dtype=torch.long),
        }
        if self.augment:
            data = self.transform(data)
        return data

class FigshareSeg2DDS(Dataset):
    def __init__(self, X, m, augment=False):
        self.X, self.m = X, m
        self.augment = augment
        if augment:
            self.transform = Compose([
                RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
                RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
                RandAffined(keys=["image", "label"], rotate_range=0.2, scale_range=0.15, translate_range=10, prob=0.6, mode=["bilinear", "nearest"]),
                RandGridDistortiond(keys=["image", "label"], num_cells=5, distort_limit=0.15, prob=0.3, mode=["bilinear", "nearest"]),
                RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
                RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.4),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
            ])
   
    def __len__(self): return len(self.X)
   
    def __getitem__(self, i):
        data = {
            "image": torch.from_numpy(self.X[i]).float(),
            "label": torch.from_numpy(self.m[i]).long(),
        }
        if self.augment:
            data = self.transform(data)
        return data

def load_brats_3d(brats_root: Path, max_subjects=None, target=VOL_SZ):
    root = Path(brats_root)
    subs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
    subs = sorted(subs)
    if max_subjects: subs = subs[:max_subjects]
    vols = []
    for sd in subs:
        try:
            # Load all modalities
            mods = {}
            for m in ["t1", "t1ce", "t2", "flair"]:
                f = list(sd.glob(f"*_{m}.nii*"))
                if not f: raise FileNotFoundError(f"Missing {m}")
                mods[m] = nib.load(str(f[0])).get_fdata()
            seg = nib.load(str(list(sd.glob("*_seg.nii*"))[0])).get_fdata()
            seg = (seg > 0).astype(np.float32)

            # Resize each modality to target
            img_channels = []
            for m in ["t1", "t1ce", "t2", "flair"]:
                vol = resize(mods[m], target, preserve_range=True, anti_aliasing=True).astype(np.float32)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
                img_channels.append(vol)
            img4 = np.stack(img_channels, axis=0)  # (4, H, W, D)

            seg_r = resize(seg, target, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            seg_r = (seg_r > 0.5).astype(np.float32)

            vols.append({"image": img4, "label": seg_r[None, ...], "id": sd.name})
        except Exception as e:
            print(f"❌ {sd.name}: {e}")
    print(f"✅ Loaded {len(vols)} BraTS multi-channel volumes")
    return vols

class BraTS3DDS(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment
        if augment:
            self.transform = Compose([
                RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, range_z=0.1, prob=0.6, mode=["trilinear", "nearest"]),
                RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
                RandAffined(keys=["image", "label"], rotate_range=[0.15, 0.15, 0.1], scale_range=0.1, translate_range=5, prob=0.7, mode=["trilinear", "nearest"]),
                RandGridDistortiond(keys=["image", "label"], num_cells=3, distort_limit=0.1, prob=0.3, mode=["trilinear", "nearest"]),
                RandGaussianNoised(keys=["image"], std=0.03, prob=0.3),
                RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.5),
            ])
   
    def __len__(self): return len(self.items)
   
    def __getitem__(self, i):
        x = torch.from_numpy(self.items[i]["image"]).float()
        y = torch.from_numpy(self.items[i]["label"]).long()
        data = {"image": x, "label": y}
       
        if self.augment:
            data = self.transform(data)
       
        return data

# Model definitions
clf = DenseNet121(spatial_dims=2, in_channels=1, out_channels=NUM_CLS).to(DEVICE)
opt_clf = optim.AdamW(clf.parameters(), lr=LR_CLS, weight_decay=1e-4)
sch_clf = optim.lr_scheduler.ReduceLROnPlateau(opt_clf, mode='min', patience=3, factor=0.7, verbose=True)

seg3d = UNet(
    spatial_dims=3, in_channels=4, out_channels=NUM_SEG,
    channels=(32, 64, 128, 256, 512), strides=(2,2,2,2), num_res_units=3
).to(DEVICE)

opt_seg = optim.AdamW(seg3d.parameters(), lr=LR_SEG, weight_decay=1e-4)
sch_seg = optim.lr_scheduler.ReduceLROnPlateau(opt_seg, mode='min', patience=5, factor=0.8, verbose=True)
seg_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True,
                      lambda_dice=0.7, lambda_ce=0.3)

def train_classifier(train_loader, val_loader, ce_loss, epochs=EPOCHS_CLS):
    print("\n==== Train 2D Classifier (Figshare) ====")
    history_cls = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best = 1e9; bad=0; patience=6; best_state=None
    for ep in range(1, epochs+1):
        clf.train()
        tr_loss=0; tr_corr=0; tr_tot=0
        for b in train_loader:
            x = b["image"].to(DEVICE); y = b["label"].to(DEVICE)
            opt_clf.zero_grad(set_to_none=True)
            logits = clf(x)
            loss = ce_loss(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt_clf.step()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (logits.argmax(1)==y).sum().item()
            tr_tot += x.size(0)
        clf.eval(); va_loss=0; va_corr=0; va_tot=0
        with torch.no_grad():
            for b in val_loader:
                x=b["image"].to(DEVICE); y=b["label"].to(DEVICE)
                logits=clf(x)
                l=ce_loss(logits,y)
                va_loss += l.item()*x.size(0)
                va_corr += (logits.argmax(1)==y).sum().item()
                va_tot += x.size(0)
        trl=tr_loss/len(train_loader.dataset); val=va_loss/len(val_loader.dataset)
        ta=100*tr_corr/tr_tot; va=100*va_corr/va_tot
        print(f"[CLS] ep {ep:02d}  loss {trl:.4f}  vloss {val:.4f}  acc {ta:.1f}%  vacc {va:.1f}%")
        history_cls['train_loss'].append(trl)
        history_cls['val_loss'].append(val)
        history_cls['train_acc'].append(ta)
        history_cls['val_acc'].append(va)
       
        sch_clf.step(val)
       
        if val < best - 1e-6:
            best=val; bad=0
            best_state={k:v.detach().cpu() for k,v in clf.state_dict().items()}
        else:
            bad+=1
            if bad>=patience:
                print("⏹️ early stop clf")
                break
    if best_state: clf.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    print("✅ Classifier done.")
    return clf, history_cls

def train_segmenter_3d(train_loader, val_loader, epochs=EPOCHS_SEG, patience=8):
    print("\n==== Train 3D Segmenter (BraTS) ====")
    history_seg = {'train_loss':[], 'val_loss':[], 'train_dice':[], 'val_dice':[]}

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs+1):
        seg3d.train()
        tr_loss = 0
        tr_dices = []

        for b in train_loader:
            x = b["image"].to(DEVICE); y = b["label"].to(DEVICE)
            opt_seg.zero_grad(set_to_none=True)
            logits = seg3d(x)
            loss = seg_loss(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(seg3d.parameters(), 1.0)
            opt_seg.step()

            tr_loss += loss.item() * x.size(0)

            # Dice on training batch
            with torch.no_grad():
                pred = logits.argmax(1)
                gt = y[:,0]
                inter = (pred*gt).sum(dim=(1,2,3))
                denom = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
                dice = (2*inter + 1e-6) / (denom + 1e-6)
                tr_dices += dice.detach().cpu().tolist()

        # Validation
        seg3d.eval()
        va_loss = 0
        va_dices = []
        with torch.no_grad():
            for b in val_loader:
                x = b["image"].to(DEVICE); y = b["label"].to(DEVICE)
                logits = seg3d(x)
                loss = seg_loss(logits, y)
                va_loss += loss.item() * x.size(0)
                pred = logits.argmax(1); gt=y[:,0]
                inter=(pred*gt).sum(dim=(1,2,3))
                denom=pred.sum(dim=(1,2,3))+gt.sum(dim=(1,2,3))
                dice=(2*inter+1e-6)/(denom+1e-6)
                va_dices += dice.detach().cpu().tolist()

        # Epoch stats
        trl = tr_loss / len(train_loader.dataset)
        val = va_loss / len(val_loader.dataset)
        td = float(np.mean(tr_dices)) if tr_dices else 0.0
        vd = float(np.mean(va_dices)) if va_dices else 0.0

        print(f"[SEG3D] ep {ep:02d}  loss {trl:.4f}  vloss {val:.4f}  dice {td:.4f}  vdice {vd:.4f}")

        history_seg['train_loss'].append(trl)
        history_seg['val_loss'].append(val)
        history_seg['train_dice'].append(td)
        history_seg['val_dice'].append(vd)

        # Scheduler step
        sch_seg.step(val)

        # Early stopping check
        if val < best_val_loss - 1e-6:
            best_val_loss = val
            best_state = {k: v.detach().cpu() for k, v in seg3d.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⏹️ Early stopping seg3d at epoch {ep}")
                break

    # Load best model
    if best_state:
        seg3d.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print("✅ Segmenter done.")
    return seg3d, history_seg

class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.fc4(x)

class PINN2D(nn.Module):
    def __init__(self, steps=PINN_STEPS, dt=DT):
        super().__init__()
        self.steps = int(steps)
        self.dt = float(dt)
        self.T = self.steps * self.dt
        self.net = MLP()
        self.log_D = nn.Parameter(torch.tensor(-3.0))
        self.log_rho = nn.Parameter(torch.tensor(-3.0))

    @property
    def D(self):
        return torch.nn.functional.softplus(self.log_D) + 1e-6

    @property
    def rho(self):
        return torch.nn.functional.softplus(self.log_rho) + 1e-6

    @property
    def kappa(self):
        return 1.0

    def forward(self, xyt):
        raw = self.net(xyt)
        u = torch.sigmoid(raw)
        return u

def compute_gradients(model, xyt, outputs):
    grads = torch.autograd.grad(outputs=outputs, inputs=xyt, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return grads

def train_pinn_2d(target_mask, steps=PINN_STEPS, iters=PINN_ITERS, lr=PINN_LR, save_path=None):
    H, W = target_mask.shape[-2], target_mask.shape[-1]
    pinn = PINN2D(steps=steps).to(DEVICE)
    opt = optim.AdamW(pinn.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.8)
    pinn_logs = {'total':[], 'data':[], 'phys':[], 'bc':[], 'ic':[], 'D':[], 'rho':[], 'log_D':[], 'log_rho':[]}
    print(f"\n==== Train 2D PINN (collocation-based) on {H}x{W} ====")
    best = 1e9; bad=0; patience=120; best_state=None

    # Prepare data points
    xx, yy = torch.meshgrid(torch.linspace(0, 1, H, device=DEVICE), torch.linspace(0, 1, W, device=DEVICE))
    xx_data = xx.flatten()
    yy_data = yy.flatten()
    tt_data = torch.ones_like(xx_data) * pinn.T
    xyt_data = torch.stack([xx_data, yy_data, tt_data], dim=-1)
    u_target = target_mask.view(-1)

    lambda_data = 1.0
    lambda_phys = 1.0
    lambda_bc = 1.0
    lambda_ic = 1e-3

    N_colloc = 10000
    N_bc = 2000
    N_ic = 1000

    for it in range(1, iters+1):
        opt.zero_grad(set_to_none=True)

        # Data loss
        u_pred_data = pinn(xyt_data)
        data_loss = nn.functional.mse_loss(u_pred_data.squeeze(), u_target)

        # Sample collocation points
        xx_col = torch.rand(N_colloc, device=DEVICE)
        yy_col = torch.rand(N_colloc, device=DEVICE)
        tt_col = torch.rand(N_colloc, device=DEVICE) * pinn.T
        xyt_col = torch.stack([xx_col, yy_col, tt_col], dim=-1)
        xyt_col.requires_grad = True

        u_col = pinn(xyt_col)
        grads_col = compute_gradients(pinn, xyt_col, u_col)
        u_t = grads_col[:, 2]

        u_x = grads_col[:, 0]
        u_xx = compute_gradients(pinn, xyt_col, u_x)[:, 0]

        u_y = grads_col[:, 1]
        u_yy = compute_gradients(pinn, xyt_col, u_y)[:, 1]

        lap = u_xx + u_yy
        growth = pinn.rho * u_col.squeeze() * (pinn.kappa - u_col.squeeze())
        rhs = pinn.D * lap + growth
        residual = u_t - rhs
        phys_loss = torch.mean(residual.pow(2))

        # Boundary loss (Neumann zero flux)
        bc_loss = 0.0
        N_bc_side = N_bc // 4

        # Left x=0
        xx_left = torch.zeros(N_bc_side, device=DEVICE)
        yy_left = torch.rand(N_bc_side, device=DEVICE)
        tt_left = torch.rand(N_bc_side, device=DEVICE) * pinn.T
        xyt_left = torch.stack([xx_left, yy_left, tt_left], -1)
        xyt_left.requires_grad = True
        u_left = pinn(xyt_left)
        grads_left = compute_gradients(pinn, xyt_left, u_left)
        u_x_left = grads_left[:, 0]
        bc_loss += torch.mean(u_x_left.pow(2))

        # Right x=1
        xx_right = torch.ones(N_bc_side, device=DEVICE)
        yy_right = torch.rand(N_bc_side, device=DEVICE)
        tt_right = torch.rand(N_bc_side, device=DEVICE) * pinn.T
        xyt_right = torch.stack([xx_right, yy_right, tt_right], -1)
        xyt_right.requires_grad = True
        u_right = pinn(xyt_right)
        grads_right = compute_gradients(pinn, xyt_right, u_right)
        u_x_right = grads_right[:, 0]
        bc_loss += torch.mean(u_x_right.pow(2))

        # Bottom y=0
        xx_bottom = torch.rand(N_bc_side, device=DEVICE)
        yy_bottom = torch.zeros(N_bc_side, device=DEVICE)
        tt_bottom = torch.rand(N_bc_side, device=DEVICE) * pinn.T
        xyt_bottom = torch.stack([xx_bottom, yy_bottom, tt_bottom], -1)
        xyt_bottom.requires_grad = True
        u_bottom = pinn(xyt_bottom)
        grads_bottom = compute_gradients(pinn, xyt_bottom, u_bottom)
        u_y_bottom = grads_bottom[:, 1]
        bc_loss += torch.mean(u_y_bottom.pow(2))

        # Top y=1
        xx_top = torch.rand(N_bc_side, device=DEVICE)
        yy_top = torch.ones(N_bc_side, device=DEVICE)
        tt_top = torch.rand(N_bc_side, device=DEVICE) * pinn.T
        xyt_top = torch.stack([xx_top, yy_top, tt_top], -1)
        xyt_top.requires_grad = True
        u_top = pinn(xyt_top)
        grads_top = compute_gradients(pinn, xyt_top, u_top)
        u_y_top = grads_top[:, 1]
        bc_loss += torch.mean(u_y_top.pow(2))

        bc_loss /= 4.0

        # IC reg
        xx_ic = torch.rand(N_ic, device=DEVICE)
        yy_ic = torch.rand(N_ic, device=DEVICE)
        tt_ic = torch.zeros(N_ic, device=DEVICE)
        xyt_ic = torch.stack([xx_ic, yy_ic, tt_ic], -1)
        raw_ic = pinn.net(xyt_ic)
        ic_reg = torch.mean(torch.abs(raw_ic))

        total = lambda_data * data_loss + lambda_phys * phys_loss + lambda_bc * bc_loss + lambda_ic * ic_reg
        total.backward()
        nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
        sch.step(total)

        if it % 50 == 0 or it == 1:
            print(f"[PINN] {it:4d}/{iters}  total {total.item():.2e} | data {data_loss.item():.2e} | phys {phys_loss.item():.2e} | bc {bc_loss.item():.2e} | ic {ic_reg.item():.2e} | D {pinn.D.item():.2e} | rho {pinn.rho.item():.2e} | log_D {pinn.log_D.item():.2e} | log_rho {pinn.log_rho.item():.2e}")
            print(f"u_t mean: {u_t.mean().item():.2e}, var: {u_t.var().item():.2e} | lap mean: {lap.mean().item():.2e}, var: {lap.var().item():.2e} | growth mean: {growth.mean().item():.2e}, var: {growth.var().item():.2e}")
            pinn_logs['total'].append(total.item())
            pinn_logs['data'].append(data_loss.item())
            pinn_logs['phys'].append(phys_loss.item())
            pinn_logs['bc'].append(bc_loss.item())
            pinn_logs['ic'].append(ic_reg.item())
            pinn_logs['D'].append(pinn.D.item())
            pinn_logs['rho'].append(pinn.rho.item())
            pinn_logs['log_D'].append(pinn.log_D.item())
            pinn_logs['log_rho'].append(pinn.log_rho.item())

        # Early stopping and best model tracking
        if total.item() < best - 1e-9:
            best = total.item()
            bad = 0
            # Save best state
            best_state = {k: v.detach().cpu() for k, v in pinn.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                print("⏹️ early stop PINN")
                break
    
    # Load best model state
    if best_state:
        pinn.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    # Save the trained PINN model
    if save_path:
        checkpoint = {
            'model_state_dict': pinn.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sch.state_dict(),
            'config': {
                'steps': steps,
                'dt': DT,
                'img_sz': (H, W),
                'iters': it,  # actual iterations completed
                'lr': lr,
                'best_loss': best
            },
            'logs': pinn_logs,
            'final_params': {
                'D': pinn.D.item(),
                'rho': pinn.rho.item(),
                'log_D': pinn.log_D.item(),
                'log_rho': pinn.log_rho.item(),
                'T': pinn.T
            }
        }
        torch.save(checkpoint, save_path)
        print(f"💾 PINN model saved to: {save_path}")
        print(f"   Final D: {pinn.D.item():.3e}")
        print(f"   Final rho: {pinn.rho.item():.3e}")
        print(f"   Best loss: {best:.3e}")
    
    print("✅ PINN done.")
    return pinn, pinn_logs

def load_pinn_2d(load_path, device=DEVICE):
    """Load a trained PINN model from checkpoint"""
    checkpoint = torch.load(load_path, map_location=device)
    
    # Recreate the model with the same configuration
    config = checkpoint['config']
    pinn = PINN2D(steps=config['steps'], dt=DT).to(device)
    
    # Load the trained weights
    pinn.load_state_dict(checkpoint['model_state_dict'])
    pinn.eval()
    
    print(f"📁 PINN model loaded from: {load_path}")
    print(f"   Model trained for {config['iters']} iterations")
    print(f"   Final D: {checkpoint['final_params']['D']:.3e}")
    print(f"   Final rho: {checkpoint['final_params']['rho']:.3e}")
    print(f"   Best loss: {config['best_loss']:.3e}")
    
    return pinn, checkpoint['logs'], checkpoint['config']

def pick_axial_slice_mask(brats_item, size=IMG_SZ):
    vol = brats_item["label"][0]
    z = vol.shape[-1] // 2
    sl = vol[..., z]
    slr = resize(sl, (size, size), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
    return torch.from_numpy(slr[None, ...]).to(DEVICE)

def get_3d_sample_for_visualization(brats_items):
    """Get a sample 3D volume and extract middle slice for visualization"""
    if not brats_items:
        return None, None, None
    
    # Take the first BraTS item
    sample_item = brats_items[0]
    image_3d = torch.from_numpy(sample_item["image"]).float().unsqueeze(0).to(DEVICE)  # Add batch dimension
    label_3d = torch.from_numpy(sample_item["label"]).long().unsqueeze(0).to(DEVICE)   # Add batch dimension
    
    # Extract middle slice for visualization
    mid_slice = image_3d.shape[-1] // 2
    image_2d = image_3d[0, 0, :, :, mid_slice].cpu().numpy()  # Remove batch and channel dimensions
    label_2d = label_3d[0, 0, :, :, mid_slice].cpu().numpy()  # Remove batch and channel dimensions
    
    return image_3d, image_2d, label_2d

def visualize_results(history_cls, history_seg, pinn_logs, sample_batch, clf_model, seg_model, pinn_model, brats_items, DEVICE):
    plt.figure("Learning Curves", figsize=(14, 6))
    plt.subplot(2, 3, 1)
    plt.plot(history_cls['train_loss'], label='Train Loss')
    plt.plot(history_cls['val_loss'], label='Val Loss')
    plt.title("Classifier Loss")
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.plot(history_cls['train_acc'], label='Train Acc')
    plt.plot(history_cls['val_acc'], label='Val Acc')
    plt.title("Classifier Accuracy")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(history_seg['train_loss'], label='Train Loss')
    plt.plot(history_seg['val_loss'], label='Val Loss')
    plt.title("Segmentation Loss")
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(history_seg['train_dice'], label='Train Dice')
    plt.plot(history_seg['val_dice'], label='Val Dice')
    plt.title("Segmentation Dice")
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(pinn_logs['total'], label='Total')
    plt.plot(pinn_logs['data'], label='Data')
    plt.plot(pinn_logs['phys'], label='Physics')
    plt.title("PINN Losses")
    plt.legend()
    plt.subplot(2, 3, 6)
    plt.plot(pinn_logs['D'], label='D (diffusion)')
    plt.plot(pinn_logs['rho'], label='rho (growth)')
    plt.title("PINN Parameters")
    plt.legend()
    plt.tight_layout()

    # Predictions & Overlays
    plt.figure("Predictions & Overlays", figsize=(18, 8))
    imgs, labels, masks = sample_batch
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    
    # 2D Classifier prediction
    with torch.no_grad():
        logits = clf_model(imgs)
        cls_pred = torch.argmax(logits, dim=1).cpu().numpy()
    
    plt.subplot(2, 3, 1)
    plt.imshow(imgs[0, 0].cpu(), cmap="gray")
    plt.title(f"Classifier pred: {cls_pred[0]} (GT: {labels[0]})")
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(imgs[0, 0].cpu(), cmap="gray")
    plt.imshow(masks[0].cpu(), alpha=0.5, cmap="Reds")
    plt.title("GT Mask")
    plt.colorbar()
    
    # 3D Segmentation prediction
    vol_3d, img_slice, gt_slice = get_3d_sample_for_visualization(brats_items)
    if vol_3d is not None and seg_model is not None:
        with torch.no_grad():
            seg3d_logits = seg_model(vol_3d)
            seg3d_pred = torch.argmax(seg3d_logits, dim=1)[0].cpu().numpy()  # Remove batch dimension
            # Extract middle slice from 3D prediction
            mid_slice_idx = seg3d_pred.shape[-1] // 2
            pred_slice = seg3d_pred[:, :, mid_slice_idx]
        
        plt.subplot(2, 3, 3)
        plt.imshow(img_slice, cmap="gray")
        plt.imshow(pred_slice, alpha=0.6, cmap="Greens")
        plt.title("3D Seg Prediction (Mid Axial)")
        plt.colorbar()
    else:
        plt.subplot(2, 3, 3)
        plt.text(0.5, 0.5, "No BraTS data available", 
                 ha='center', va='center', fontsize=12)
        plt.title("3D Seg Prediction (Unavailable)")
    
    # PINN results
    with torch.no_grad():
        H = IMG_SZ
        W = IMG_SZ
        xx, yy = torch.meshgrid(torch.linspace(0, 1, H, device=DEVICE), torch.linspace(0, 1, W, device=DEVICE))
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        
        tt_init = torch.zeros_like(xx_flat)
        xyt_init = torch.stack([xx_flat, yy_flat, tt_init], dim=-1)
        init_u = pinn_model(xyt_init).reshape(H, W).detach().cpu().numpy()
        
        tt_final = torch.ones_like(xx_flat) * pinn_model.T
        xyt_final = torch.stack([xx_flat, yy_flat, tt_final], dim=-1)
        final_u = pinn_model(xyt_final).reshape(H, W).detach().cpu().numpy()
    
    plt.subplot(2, 3, 4)
    plt.imshow(init_u, cmap="hot")
    plt.title("PINN Initial Field")
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.imshow(final_u, cmap="hot")
    plt.title("PINN Final Field")
    plt.colorbar()
    
    # Additional comparison plot if we have 3D segmentation
    if vol_3d is not None and seg_model is not None:
        plt.subplot(2, 3, 6)
        plt.imshow(img_slice, cmap="gray", alpha=0.7)
        plt.imshow(gt_slice, alpha=0.4, cmap="Reds", label="Ground Truth")
        plt.imshow(pred_slice, alpha=0.4, cmap="Greens", label="Prediction")
        plt.title("3D Seg: GT vs Pred Overlay")
        plt.legend()
        plt.colorbar()
    else:
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, "3D Comparison\nNot Available", 
                 ha='center', va='center', fontsize=12)
        plt.title("3D Seg Comparison")
    
    plt.tight_layout()
    plt.show()

def visualize_segmentation(image, mask, atlas_info=None, save_path=None):
    """Enhanced visualization with anatomical region overlay"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original segmentation overlay
    axes[0].imshow(image[0].cpu().numpy(), cmap="gray")
    axes[0].imshow(mask.cpu().numpy(), alpha=0.5, cmap="Reds")
    axes[0].axis('off')
    axes[0].set_title("Segmentation Overlay")
    
    # Anatomical region overlay
    axes[1].imshow(image[0].cpu().numpy(), cmap="gray")
    axes[1].imshow(mask.cpu().numpy(), alpha=0.5, cmap="Reds")
    
    if atlas_info:
        # Add anatomical region annotation
        coords = atlas_info['coordinates']['pixel']
        axes[1].plot(coords['x'], coords['y'], 'yo', markersize=10, markeredgecolor='blue', markeredgewidth=2)
        
        # Add text annotation
        region_text = f"{atlas_info['detailed_region']}\n({atlas_info['hemisphere']})\nConf: {atlas_info['confidence']:.2f}"
        axes[1].text(coords['x'] + 10, coords['y'] - 10, region_text, 
                    color='yellow', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    axes[1].axis('off')
    axes[1].set_title("Anatomical Localization")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

def test_on_sample(image, gt_label, gt_mask, clf_model, atlas, pinn_iters=100):
    """Enhanced test function with anatomical atlas integration"""
    
    # Classify
    with torch.no_grad():
        logit = clf_model(image.unsqueeze(0).to(DEVICE))
        pred_class = torch.argmax(logit).item()
        class_probabilities = torch.softmax(logit, dim=1)[0].cpu().numpy()

    class_names = {0: "meningioma", 1: "glioma", 2: "pituitary"}

    # Get anatomical location using atlas
    mask_np = gt_mask.cpu().numpy()
    
    # Use atlas to determine anatomical region
    atlas_info = atlas.get_region_from_mask(mask_np, method='centroid')
    clinical_description = atlas.get_clinical_description(atlas_info)
    
    # Calculate additional mask statistics
    area = np.sum(mask_np)
    bbox = atlas_info['mask_stats']['bounding_box']
    
    # Calculate tumor characteristics
    tumor_stats = calculate_tumor_characteristics(mask_np)
    
    # Fit PINN on this mask for growth modeling
    target_2d = gt_mask.float().unsqueeze(0).to(DEVICE)  # [1, H, W]
    pinn_test, pinn_logs = train_pinn_2d(target_2d, steps=PINN_STEPS, iters=pinn_iters, lr=PINN_LR)

    D = pinn_test.D.item()
    rho = pinn_test.rho.item()

    # Calculate volume changes and growth metrics
    growth_metrics = calculate_growth_metrics(pinn_test, IMG_SZ, DEVICE)

    # Enhanced visualization with atlas information
    vis_path = "enhanced_segmentation_overlay.png"
    visualize_segmentation(image, gt_mask, atlas_info, vis_path)

    # Create comprehensive JSON output
    json_data = {
        "tumor_classification": {
            "predicted_type": class_names.get(pred_class, "unknown"),
            "predicted_class": pred_class,
            "ground_truth_class": int(gt_label.item()),
            "classification_confidence": {
                "meningioma": float(class_probabilities[0]),
                "glioma": float(class_probabilities[1]),
                "pituitary": float(class_probabilities[2])
            }
        },
        "anatomical_location": {
            "main_region": atlas_info['main_region'],
            "detailed_region": atlas_info.get('detailed_region', atlas_info['main_region']),
            "hemisphere": atlas_info['hemisphere'],
            "coordinates": atlas_info['coordinates'],
            "localization_confidence": atlas_info['confidence'],
            "bounding_box": bbox,
            "clinical_description": clinical_description
        },
        "tumor_characteristics": {
            "size_pixels": float(area),
            "size_mm2_estimated": float(area * 0.25),  # Assuming ~0.5mm pixel spacing
            "morphology": tumor_stats,
            "growth_modeling": {
                "diffusion_coefficient": D,
                "growth_rate": rho,
                "predicted_volume_change_percent": growth_metrics['volume_change_percent'],
                "growth_trajectory": growth_metrics['trajectory'],
                "time_horizon_days": growth_metrics['time_horizon_days']
            }
        },
        "clinical_assessment": {
            "affected_functions": clinical_description['affected_functions'],
            "clinical_significance": clinical_description['clinical_significance'],
            "laterality_notes": clinical_description['laterality_notes'],
            "risk_assessment": assess_tumor_risk(pred_class, atlas_info, area, D, rho)
        },
        "visualization_path": vis_path
    }

    return json_data

def calculate_tumor_characteristics(mask):
    """Calculate morphological characteristics of the tumor"""
    if mask.sum() == 0:
        return {"error": "No tumor detected"}
    
    # Basic shape metrics
    area = mask.sum()
    perimeter = calculate_perimeter(mask)
    
    # Compactness (4π * area / perimeter²)
    compactness = 4 * np.pi * area / (perimeter**2 + 1e-6)
    
    # Eccentricity using moments
    moments = calculate_moments(mask)
    eccentricity = calculate_eccentricity(moments)
    
    # Convex hull ratio
    convex_hull_ratio = calculate_convex_hull_ratio(mask)
    
    return {
        "area_pixels": float(area),
        "perimeter_pixels": float(perimeter),
        "compactness": float(compactness),
        "eccentricity": float(eccentricity),
        "convex_hull_ratio": float(convex_hull_ratio),
        "aspect_ratio": calculate_aspect_ratio(mask),
        "solidity": float(convex_hull_ratio),  # Same as convex hull ratio
        "shape_complexity": categorize_shape_complexity(compactness, eccentricity)
    }

def calculate_perimeter(mask):
    """Calculate tumor perimeter using edge detection"""
    from scipy import ndimage
    # Simple edge detection
    edges = ndimage.binary_erosion(mask) ^ mask
    return edges.sum()

def calculate_moments(mask):
    """Calculate image moments for shape analysis"""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return {"m00": 0, "m10": 0, "m01": 0, "m20": 0, "m11": 0, "m02": 0}
    
    # Central moments
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    
    m00 = len(x_coords)  # area
    m10 = np.sum(x_coords)
    m01 = np.sum(y_coords)
    m20 = np.sum((x_coords - x_mean)**2)
    m11 = np.sum((x_coords - x_mean) * (y_coords - y_mean))
    m02 = np.sum((y_coords - y_mean)**2)
    
    return {"m00": m00, "m10": m10, "m01": m01, "m20": m20, "m11": m11, "m02": m02}

def calculate_eccentricity(moments):
    """Calculate eccentricity from moments"""
    m20, m11, m02 = moments["m20"], moments["m11"], moments["m02"]
    
    if moments["m00"] == 0:
        return 0
    
    # Normalized central moments
    mu20 = m20 / moments["m00"]
    mu11 = m11 / moments["m00"]
    mu02 = m02 / moments["m00"]
    
    # Eigenvalues of covariance matrix
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11**2
    
    if det <= 0:
        return 0
    
    lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
    lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2
    
    if lambda1 <= 0:
        return 0
    
    eccentricity = np.sqrt(1 - lambda2 / lambda1)
    return min(eccentricity, 1.0)

def calculate_convex_hull_ratio(mask):
    """Calculate ratio of area to convex hull area"""
    from scipy.spatial import ConvexHull
    
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 3:
        return 1.0
    
    try:
        points = np.column_stack((x_coords, y_coords))
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        actual_area = len(x_coords)
        return actual_area / hull_area if hull_area > 0 else 1.0
    except:
        return 1.0

def calculate_aspect_ratio(mask):
    """Calculate aspect ratio of bounding box"""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return 1.0
    
    width = np.max(x_coords) - np.min(x_coords) + 1
    height = np.max(y_coords) - np.min(y_coords) + 1
    
    return float(max(width, height) / min(width, height))

def categorize_shape_complexity(compactness, eccentricity):
    """Categorize tumor shape complexity"""
    if compactness > 0.8 and eccentricity < 0.5:
        return "Regular (circular/oval)"
    elif compactness > 0.6 and eccentricity < 0.7:
        return "Moderately irregular"
    elif eccentricity > 0.8:
        return "Elongated"
    else:
        return "Highly irregular"

def calculate_growth_metrics(pinn_model, img_size, device):
    """Calculate comprehensive growth metrics from PINN"""
    with torch.no_grad():
        xx, yy = torch.meshgrid(
            torch.linspace(0, 1, img_size, device=device), 
            torch.linspace(0, 1, img_size, device=device)
        )
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        
        # Calculate volume at different time points
        time_points = np.linspace(0, pinn_model.T, 11)  # 11 points from 0 to T
        volumes = []
        
        for t in time_points:
            tt = torch.ones_like(xx_flat) * t
            xyt = torch.stack([xx_flat, yy_flat, tt], dim=-1)
            u = pinn_model(xyt)
            volume = u.sum().item()
            volumes.append(volume)
        
        # Calculate growth metrics
        initial_volume = volumes[0]
        final_volume = volumes[-1]
        volume_change_percent = (final_volume / max(1e-6, initial_volume) - 1) * 100
        
        # Growth trajectory
        trajectory = []
        for i, (t, v) in enumerate(zip(time_points, volumes)):
            trajectory.append({
                "time": float(t),
                "volume": float(v),
                "relative_change": float((v / max(1e-6, initial_volume) - 1) * 100)
            })
        
        # Estimate time horizon in days (assuming time units are normalized)
        time_horizon_days = float(pinn_model.T * 30)  # Rough conversion to days
        
        return {
            "volume_change_percent": volume_change_percent,
            "trajectory": trajectory,
            "time_horizon_days": time_horizon_days,
            "initial_volume": initial_volume,
            "final_volume": final_volume,
            "diffusion_rate": pinn_model.D.item(),
            "growth_rate": pinn_model.rho.item()
        }

def assess_tumor_risk(tumor_class, atlas_info, size_pixels, diffusion_coeff, growth_rate):
    """Assess tumor risk based on multiple factors"""
    
    # Base risk by tumor type
    type_risk = {
        0: "Low-Moderate",    # meningioma
        1: "High",           # glioma  
        2: "Moderate"        # pituitary
    }
    
    base_risk = type_risk.get(tumor_class, "Unknown")
    
    # Size risk factor
    if size_pixels < 100:
        size_risk = "Low"
    elif size_pixels < 500:
        size_risk = "Moderate"
    else:
        size_risk = "High"
    
    # Location risk factor
    location_risk = "Moderate"  # Default
    region = atlas_info['main_region'].lower()
    
    if any(term in region for term in ['frontal', 'motor', 'central']):
        location_risk = "High"  # Motor areas
    elif any(term in region for term in ['temporal', 'language']):
        location_risk = "High"  # Language areas
    elif 'deep' in region:
        location_risk = "Very High"  # Deep structures
    elif 'occipital' in region:
        location_risk = "Moderate"  # Visual areas
    
    # Growth risk based on PINN parameters
    if growth_rate > 0.1:
        growth_risk = "High"
    elif growth_rate > 0.05:
        growth_risk = "Moderate"
    else:
        growth_risk = "Low"
    
    # Infiltration risk based on diffusion
    if diffusion_coeff > 0.1:
        infiltration_risk = "High"
    elif diffusion_coeff > 0.05:
        infiltration_risk = "Moderate"
    else:
        infiltration_risk = "Low"
    
    return {
        "overall_risk": base_risk,
        "size_risk": size_risk,
        "location_risk": location_risk,
        "growth_risk": growth_risk,
        "infiltration_risk": infiltration_risk,
        "risk_factors": {
            "tumor_type": tumor_class,
            "size_pixels": size_pixels,
            "anatomical_sensitivity": location_risk,
            "growth_rate": growth_rate,
            "diffusion_rate": diffusion_coeff
        }
    }

def main(fig_max=None, brats_max=None, use_brats_for_pinn=True, save_pinn=True, load_existing_pinn=False):
    print("\n================= DATA =================")
    
    # Initialize brain atlas
    atlas = BrainAtlas(img_size=IMG_SZ)
    print(f"🧠 Brain Atlas initialized with {len(atlas.regions)} anatomical regions")
    
    X, M, y = load_figshare_2d(FIGSHARE_DIR, max_samples=fig_max)
    idx = np.arange(len(X))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y if len(np.unique(y))>1 else None, random_state=42)
    
    counts = Counter(y.tolist())
    total = len(y)
    class_weights = []
    for i in range(NUM_CLS):
        cnt = counts.get(i, 1)
        class_weights.append(total / (NUM_CLS * cnt))
    class_weights = np.array(class_weights, dtype=np.float32)
    print(f"📊 Computed classifier class weights: {dict(enumerate(class_weights.tolist()))}")
    
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    
    ds_tr = FigshareClsDS(X[tr], y[tr], augment=True)
    ds_va = FigshareClsDS(X[va], y[va], augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_2D, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_2D, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    
    # Load classifier instead of training
    print("\n==== Loading Pre-trained Classifier ====")
    checkpoint_clf = torch.load(SAVED_MODELS_DIR / "classifier_densenet121.pt")
    clf.load_state_dict(checkpoint_clf['model_state_dict'])
    clf.eval()
    history_cls = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    
    seg2d_tr = DataLoader(FigshareSeg2DDS(X[tr], M[tr], augment=True), batch_size=BATCH_2D, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    seg2d_va = DataLoader(FigshareSeg2DDS(X[va], M[va], augment=False), batch_size=BATCH_2D, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    
    print("\n================= BRATS 3D =================")
    brats_items = load_brats_3d(BRATS_DIR, max_subjects=brats_max, target=VOL_SZ)
    if len(brats_items) == 0:
        print("⚠️ No BraTS volumes found; 3D segmentation & 2D PINN from BraTS will be skipped.")
        history_seg = {'train_loss':[], 'val_loss':[], 'train_dice':[], 'val_dice':[]}
        seg3d_model = None
    else:
        ids = np.arange(len(brats_items))
        tr_b, va_b = train_test_split(ids, test_size=0.2, random_state=42)
        brats_tr = [brats_items[i] for i in tr_b]
        brats_va = [brats_items[i] for i in va_b]
        dlb_tr = DataLoader(BraTS3DDS(brats_tr, augment=True), batch_size=BATCH_3D, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
        dlb_va = DataLoader(BraTS3DDS(brats_va, augment=False), batch_size=BATCH_3D, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
        
        # Load segmenter instead of training
        print("\n==== Loading Pre-trained Segmenter ====")
        checkpoint_seg = torch.load(SAVED_MODELS_DIR / "segmenter_unet3d.pt")
        seg3d.load_state_dict(checkpoint_seg['model_state_dict'])
        seg3d.eval()
        seg3d_model = seg3d
        history_seg = {'train_loss':[], 'val_loss':[], 'train_dice':[], 'val_dice':[]}
    
    print("\n================= 2D PINN =================")
    
    # Define PINN save path
    pinn_save_path = SAVED_MODELS_DIR / "pinn_2d_tumor_growth.pt" if save_pinn else None
    
    # Check if we should load existing PINN or train new one
    if load_existing_pinn and pinn_save_path and pinn_save_path.exists():
        print(f"📁 Loading existing PINN from {pinn_save_path}")
        pinn, pinn_logs, pinn_config = load_pinn_2d(pinn_save_path, DEVICE)
    else:
        # Determine target mask for PINN training
        if use_brats_for_pinn and len(brats_items) > 0:
            target_2d = pick_axial_slice_mask(brats_items[0], size=IMG_SZ)
            print(f"🎯 PINN target: BraTS {brats_items[0]['id']} mid axial slice")
        else:
            mid = len(M) // 2
            target_2d = torch.from_numpy(M[mid][None, ...].astype(np.float32)).to(DEVICE)
            print("🎯 PINN target: Figshare mask")
        
        # Train PINN
        pinn, pinn_logs = train_pinn_2d(target_2d, steps=PINN_STEPS, iters=PINN_ITERS, lr=PINN_LR, save_path=pinn_save_path)
    
    with torch.no_grad():
        vols = []
        xx, yy = torch.meshgrid(torch.linspace(0, 1, IMG_SZ, device=DEVICE), torch.linspace(0, 1, IMG_SZ, device=DEVICE))
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        
        tt_init = torch.zeros_like(xx_flat)
        xyt_init = torch.stack([xx_flat, yy_flat, tt_init], dim=-1)
        init_u = pinn(xyt_init)
        vols.append(init_u.sum().item())
        
        tt_final = torch.ones_like(xx_flat) * pinn.T
        xyt_final = torch.stack([xx_flat, yy_flat, tt_final], dim=-1)
        final_u = pinn(xyt_final)
        vols.append(final_u.sum().item())
        
        print("\n==== PINN RESULTS ====")
        print(f"D (diffusion): {pinn.D.item():.3e}")
        print(f"rho (growth):  {pinn.rho.item():.3e}")
        print(f"Initial volume: {vols[0]:.1f}  Final: {vols[-1]:.1f}  Change: {(vols[-1]/max(1e-6,vols[0])-1)*100:.1f}%")
    
    # Prepare sample data for visualization
    sample_size = min(1, len(ds_va))
    sample_imgs = torch.stack([ds_va[i]['image'] for i in range(sample_size)])
    sample_labels = torch.tensor([ds_va[i]['label'] for i in range(sample_size)])
    ds_seg_va = FigshareSeg2DDS(X[va], M[va], augment=False)
    sample_masks = torch.stack([ds_seg_va[i]['label'] for i in range(sample_size)])
    sample_batch = (sample_imgs, sample_labels, sample_masks)
    
    visualize_results(history_cls, history_seg, pinn_logs, sample_batch, clf, seg3d_model, pinn, brats_items, DEVICE)
    
    # Run on a singular test sample from Figshare validation set with atlas
    print("\n================= Test on Singular Sample with Atlas =================")
    test_index = 7  # First sample in validation set
    test_image = ds_va[test_index]['image']
    test_label = ds_va[test_index]['label']
    test_mask = ds_seg_va[test_index]['label']
    
    json_metrics = test_on_sample(test_image, test_label, test_mask, clf, atlas, pinn_iters=100)
    
    print("\n==== Enhanced Test Sample Metrics with Anatomical Atlas (JSON) ====")
    print(json.dumps(json_metrics, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else o))

    
    # Additional atlas demonstration
    print("\n==== Atlas Demonstration ====")
    demonstrate_atlas_capabilities(atlas, ds_seg_va, n_samples=3)

def demonstrate_atlas_capabilities(atlas, dataset, n_samples=3):
    """Demonstrate atlas capabilities on multiple samples"""
    print("🗺️ Demonstrating anatomical atlas on multiple samples:")
    
    for i in range(min(n_samples, len(dataset))):
        mask = dataset[i]['label'].cpu().numpy()
        
        if mask.sum() == 0:
            continue
            
        # Test different localization methods
        methods = ['centroid', 'weighted_average']
        
        for method in methods:
            try:
                atlas_info = atlas.get_region_from_mask(mask, method=method)
                clinical_desc = atlas.get_clinical_description(atlas_info)
                
                print(f"\n  Sample {i+1} ({method}):")
                print(f"    📍 Region: {atlas_info['detailed_region']}")
                print(f"    🧭 Hemisphere: {atlas_info['hemisphere']}")
                print(f"    🎯 Confidence: {atlas_info['confidence']:.2f}")
                print(f"    ⚕️ Clinical: {clinical_desc['clinical_significance']}")
                print(f"    🧠 Functions: {', '.join(clinical_desc['affected_functions'][:2])}")
                
            except Exception as e:
                print(f"    ❌ Error with {method}: {e}")

def save_comprehensive_results(json_metrics, save_dir=None):
    """Save comprehensive results including atlas information"""
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON report
    json_path = save_dir / "tumor_analysis_with_atlas.json"
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"💾 Comprehensive results saved to: {json_path}")
    
    # Create summary report
    summary = create_clinical_summary(json_metrics)
    summary_path = save_dir / "clinical_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"📋 Clinical summary saved to: {summary_path}")
    
    return json_path, summary_path

def create_clinical_summary(metrics):
    """Create a clinical summary report"""
    tumor_info = metrics['tumor_classification']
    location_info = metrics['anatomical_location']
    characteristics = metrics['tumor_characteristics']
    clinical = metrics['clinical_assessment']
    
    summary = f"""
BRAIN TUMOR ANALYSIS REPORT
===========================

TUMOR CLASSIFICATION:
- Type: {tumor_info['predicted_type'].title()}
- Confidence: {tumor_info['classification_confidence'][tumor_info['predicted_type']]:.3f}

ANATOMICAL LOCATION:
- Region: {location_info['detailed_region']}
- Hemisphere: {location_info['hemisphere']}
- Localization Confidence: {location_info['localization_confidence']:.2f}

TUMOR CHARACTERISTICS:
- Size: {characteristics['size_pixels']:.0f} pixels (~{characteristics['size_mm2_estimated']:.1f} mm²)
- Shape: {characteristics['morphology']['shape_complexity']}
- Compactness: {characteristics['morphology']['compactness']:.3f}
- Eccentricity: {characteristics['morphology']['eccentricity']:.3f}

GROWTH MODELING:
- Diffusion Coefficient: {characteristics['growth_modeling']['diffusion_coefficient']:.2e}
- Growth Rate: {characteristics['growth_modeling']['growth_rate']:.2e}
- Predicted Volume Change: {characteristics['growth_modeling']['predicted_volume_change_percent']:.1f}%

CLINICAL ASSESSMENT:
- Overall Risk: {clinical['risk_assessment']['overall_risk']}
- Location Risk: {clinical['risk_assessment']['location_risk']}
- Growth Risk: {clinical['risk_assessment']['growth_risk']}
- Affected Functions: {', '.join(clinical['affected_functions'])}
- Clinical Significance: {clinical['clinical_significance']}

RECOMMENDATIONS:
- Monitor tumor growth with follow-up imaging
- Consider multidisciplinary team consultation
- Assess functional impact based on anatomical location
- Plan treatment approach considering location-specific risks

Note: This is a computational analysis for research purposes only.
Clinical decisions should always involve qualified medical professionals.
"""
    
    return summary

if __name__ == "__main__":
    # Parameters:
    # save_pinn=True: Save the trained PINN model
    # load_existing_pinn=False: Train new PINN (set to True to load existing one if available)
    main(fig_max=None, brats_max=None, use_brats_for_pinn=True, save_pinn=True, load_existing_pinn=True)