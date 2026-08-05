#!/usr/bin/env python3
"""Replay saved RGB-D bundles with the M37.6 hollow-cylinder joint fit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
REPO_ROOT=Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import _bundle_input, _strip_debug, draw_side_ring_fit_overlay
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_template import SideRingTemplateConfig, fit_side_ring_instance
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig, _associate_ring_mouths_detailed

def args():
 p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--output',type=Path,default=Path('/tmp/m37_6_replay')); p.add_argument('--config',type=Path,default=Path('production/foam_ring_grasp/config/line.yaml')); p.add_argument('--targets',nargs='*',default=[]); return p.parse_args()

def main():
 a=args(); raw=load_yaml(a.config.resolve()); cfg=SideRingTemplateConfig.from_mapping(raw); geo=GeometryConfig(dict(raw)); a.output.mkdir(parents=True,exist_ok=True)
 target_map={}
 for item in a.targets:
  cap,rid=item.split(':',1); target_map[cap]=int(rid)
 rows=[]
 for bundle in sorted([p for p in a.root.resolve().iterdir() if p.is_dir()]):
  cid,rgb,depth,intr,instances,_=_bundle_input(bundle)
  rings=[x for x in instances if x.class_name=='foam_ring']; mouths=[x for x in instances if x.class_name=='ring_mouth']
  matches,_,_,_=_associate_ring_mouths_detailed(rings,mouths,geo); mouth_by={int(r.instance_id):m for r,m,_ in matches}
  selected=[r for r in rings if cid not in target_map or int(r.instance_id)==target_map[cid]]
  fits=[]
  for ring in selected:
   ex=np.zeros_like(ring.mask,dtype=bool)
   for other in rings:
    if int(other.instance_id)!=int(ring.instance_id): ex|=other.mask
   mouth=mouth_by.get(int(ring.instance_id))
   screen=fit_side_ring_instance(ring,depth,intr,cfg,mouth_matched=mouth is not None,mouth_instance=mouth,search_profile='screen',exclusion_mask=ex)
   if screen.get('preliminary_pose_safe'):
    fit=fit_side_ring_instance(ring,depth,intr,cfg,mouth_matched=mouth is not None,mouth_instance=mouth,search_profile='final_verify',initial_axis=np.asarray(screen['axis_toward_camera'],dtype=np.float64),exclusion_mask=ex)
   else: fit=screen
   fit['processing_status']='final_validated' if fit.get('final_pose_safe') is not None else 'screen_evaluated'; fits.append(fit)
   rows.append({'capture_id':cid,'ring_instance_id':int(ring.instance_id),'eligible':fit.get('eligible'),'axis_toward_camera':fit.get('axis_toward_camera'),'near_uv':fit.get('near_opening_center_uv'),'far_uv':fit.get('far_opening_center_uv'),'fit_score':fit.get('fit_score'),'fit_model':fit.get('fit_model'),'surface_counts':fit.get('surface_counts'),'surface_inlier_ratio':fit.get('surface_inlier_ratio'),'surface_residual_median_mm':fit.get('surface_residual_median_mm'),'surface_residual_p90_mm':fit.get('surface_residual_p90_mm'),'rejection_reasons':fit.get('rejection_reasons')})
  selected_id=next((int(f['ring_instance_id']) for f in fits if f.get('eligible')),None)
  overlay=draw_side_ring_fit_overlay(rgb,instances,fits,intr,selected_id)
  cv2.imwrite(str(a.output/f'{cid}_overlay.jpg'),overlay)
  write_json(a.output/f'{cid}_fits.json',{'stage':'M37.6','capture_id':cid,'selected_ring_instance_id':selected_id,'fits':[_strip_debug(f) for f in fits]})
 write_json(a.output/'summary.json',{'stage':'M37.6_hollow_cylinder_multisurface','rows':rows})
 print(json.dumps({'stage':'M37.6','count':len(rows),'output':str(a.output)},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
