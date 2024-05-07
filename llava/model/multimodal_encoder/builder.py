import os
from .siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', True)
    if is_absolute_path_exists or vision_tower.startswith("google") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            print('using s2!!')
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
