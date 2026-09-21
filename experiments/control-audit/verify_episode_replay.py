"""Independently replay saved SemIf/Jevlike/Laya actions in the pinned game.

This verifies physics and trace consistency, not that model inference occurred.
Physics order reproduces JevDash (2026 Sunwood AI Labs, MIT; root LICENSE).
"""
import argparse
import hashlib
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

from audit_controls import PIN, reflex

def verify(args):
    load_started=time.perf_counter()
    actual=subprocess.check_output(['git','-C',str(args.game),'rev-parse','HEAD'],text=True).strip()
    if actual!=PIN: raise ValueError('Game revision mismatch')
    if subprocess.check_output(['git','-C',str(args.game),'status','--porcelain'],text=True).strip():
        raise ValueError('Game checkout must be clean')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    sys.path.insert(0,str(args.game/'src'))
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    raw=args.episode.read_bytes()
    episode=json.loads(raw)
    pre_step_trace=bool(episode.get('control_trace'))
    mode=episode.get('game',{}).get('control_mode','model_only')
    assisted=pre_step_trace and mode=='reflex_assisted'
    if pre_step_trace and mode not in ('model_only','reflex_assisted'):
        raise ValueError('Unsupported Laya controller mode')
    rows=episode.get('control_trace') or episode.get('frame_trace') or episode.get('trajectory',{}).get('frames')
    if not rows: raise ValueError('No supported frame trace found')
    rows=[r for r in rows if not r.get('terminal_hold',False)]
    if not rows: raise ValueError('No simulation frames found')
    level=Level(1); player=Player(*level.start_pos)
    legal={'noop','right','right_run','right_jump','right_run_jump','jump','left'}
    mismatches=[]
    loading_seconds=time.perf_counter()-load_started
    simulation_started=time.perf_counter()
    override_frames=0
    for index,row in enumerate(rows):
        frame=row.get('simulation_frame',row.get('frame'))
        if frame!=index: raise ValueError('Noncontiguous simulation frames')
        raw_action=row.get('raw_action',row.get('raw_model_action'))
        if pre_step_trace:
            obs=TelemetryExtractor.extract(player,level)
            for key in ('x','y','vx','vy','grounded','jumping','airborne_frames','running'):
                if getattr(obs.player,key)!=row['player'][key]:
                    mismatches.append({'frame':index,'field':'pre_step_player.'+key})
            if player.stalled_frames!=row['stalled_frames']:
                mismatches.append({'frame':index,'field':'stalled_frames'})
            action=reflex(obs,raw_action) if assisted else raw_action
        else:
            action=raw_action
        if action!=row['executed_action']:
            mismatches.append({'frame':index,'field':'executed_action'})
        override_frames+=int(raw_action!=action)
        if action not in legal: raise ValueError('Unknown action')
        player.apply_action(action);player.update_physics(level.tiles)
        for enemy in level.enemies:
            enemy.update(level.tiles)
            if enemy.alive and player.rect.colliderect(enemy.rect):
                if player.y+player.height<=enemy.y+enemy.height*.65:
                    enemy.stomp();player.vy=-11.;player.score+=100
                else:player.is_dead=True
        for coin in level.coins:
            if not coin.collected and player.rect.colliderect(coin.rect):
                coin.collected=True;player.coins+=1;player.score+=50
        if player.rect.colliderect(level.goal.rect):player.has_won=True
        if not pre_step_trace:
            for key in ('x','y','vx','vy'):
                if abs(getattr(player,key)-row[key])>=0.011:
                    mismatches.append({'frame':index,'field':key})
            for key in ('has_won','is_dead','score','coins'):
                if key in row and getattr(player,key)!=row[key]:
                    mismatches.append({'frame':index,'field':key})
    if pre_step_trace:
        for key in ('has_won','is_dead','score','coins'):
            if getattr(player,key)!=episode['episode'][key]:
                mismatches.append({'frame':len(rows),'field':'final.'+key})
    result={'kind':'Independent local replay; no model/GPU execution',
        'source_episode_sha256':hashlib.sha256(raw).hexdigest(),
        'game_commit':PIN,'frames_verified':len(rows),
        'control_mode':'original_reflex' if assisted else 'model_only',
        'override_frames':override_frames,
        'raw_equals_executed':override_frames==0,'physics_state_matches_every_frame':not mismatches,
        'mismatches':mismatches,'has_won':player.has_won,'is_dead':player.is_dead,
        'validation_and_loading_seconds':loading_seconds,
        'simulation_seconds':time.perf_counter()-simulation_started}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--game',type=Path,required=True)
    ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    started=time.perf_counter()
    result={'status':'running','environment':{'python':platform.python_version(),
        'os':platform.system(),'architecture':platform.machine(),
        'dependencies':{p:importlib.metadata.version(p) for p in ('pygame','pydantic')}},
        'probabilities':{'applicable':False,'reason':'Replays recorded actions without inference'},
        'peak_vram':{'applicable':False,'reason':'CPU physics replay'},'error':None}
    try:
        result.update(verify(args))
        result['status']='mismatch' if result['mismatches'] else 'success'
    except Exception as exc:
        result['status']='error'
        result['error']={'type':type(exc).__name__,'message':'Replay verification failed; exception text and paths omitted.'}
    result['total_wall_seconds']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
    return int(result['status']!='success')

if __name__=='__main__': sys.exit(main())
