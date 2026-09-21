"""Local physics controls, NOT model/GPU results. Reads an immutable game checkout."""
import argparse
import collections
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import random
import subprocess
import sys
import time

PIN = 'eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480'

def reflex(obs, raw):
    # Faithful copy of the published game's AsyncJevAgent.get_action conditions.
    t, h, p = obs.terrain, obs.hazard, obs.player
    gap = t.gap_ahead and (t.gap_distance_tiles or 99) <= 3.8
    wall = t.obstacle_ahead and (t.obstacle_distance_tiles or 99) <= 2.2
    enemy = h.enemy_ahead and h.nearest_enemy is not None and (
        h.nearest_enemy.distance_pixels <= 130 or h.jump_must_start_now)
    stalled = obs.episode.stalled_frames >= 3
    if (gap or wall or enemy or stalled) and p.grounded:
        return 'right_run_jump'
    near_gap = t.gap_ahead and t.gap_distance_tiles is not None and t.gap_distance_tiles <= 1.5
    if not p.grounded and near_gap:
        return 'right_run_jump'
    return raw

def run(raw_action, with_reflex, cadence, start_offset):
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    random.seed(42)
    level=Level(1)
    player=Player(level.start_pos[0]+start_offset,level.start_pos[1])
    counts=collections.Counter()
    overrides=0
    action=raw_action
    for frame in range(1800):
        if frame % cadence == 0:
            obs=TelemetryExtractor.extract(player,level)
            action=reflex(obs,raw_action) if with_reflex else raw_action
        counts[action]+=1
        overrides+=int(action!=raw_action)
        player.apply_action(action)
        player.update_physics(level.tiles)
        for enemy in level.enemies:
            enemy.update(level.tiles)
            if enemy.alive and player.rect.colliderect(enemy.rect):
                if player.y+player.height <= enemy.y+enemy.height*.65:
                    enemy.stomp(); player.vy=-11.; player.score+=100
                else:
                    player.is_dead=True
        for coin in level.coins:
            if not coin.collected and player.rect.colliderect(coin.rect):
                coin.collected=True; player.coins+=1; player.score+=50
        if level.goal and player.rect.colliderect(level.goal.rect):
            player.has_won=True
        if player.is_dead or player.has_won:
            break
    return dict(raw_action=raw_action,reflex=with_reflex,cadence=cadence,
        start_offset=start_offset,frames=frame+1,has_won=player.has_won,
        is_dead=player.is_dead,max_x=round(player.max_x,2),
        final_x=round(player.x,2),final_y=round(player.y,2),
        override_frames=overrides,override_rate=overrides/(frame+1),
        executed_actions=dict(counts))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--game',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    started=time.perf_counter()
    result={'kind':'local deterministic physics controls; no model inference; no Colab',
        'game_commit':PIN,'seed':42,'max_frames':1800,'cases':[],
        'environment':{'python':platform.python_version(),'os':platform.system(),
            'architecture':platform.machine(),
            'dependencies':{p:importlib.metadata.version(p) for p in ('pygame','pydantic')}},
        'probabilities':{'applicable':False,'reason':'No model inference'},
        'peak_vram':{'applicable':False,'reason':'CPU-only physics; no GPU allocated'},
        'timings':{},'status':'running','error':None}
    phase='validate_game_checkout'
    try:
        actual=subprocess.check_output(['git','-C',str(args.game),'rev-parse','HEAD'],text=True,stderr=subprocess.PIPE).strip()
        if actual!=PIN: raise ValueError('Game revision mismatch')
        if subprocess.check_output(['git','-C',str(args.game),'status','--porcelain'],text=True,stderr=subprocess.PIPE).strip():
            raise ValueError('Game checkout must be clean')
        phase='load_game_modules'
        os.environ['SDL_VIDEODRIVER']='dummy'; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
        sys.path.insert(0,str(args.game/'src'))
        from jev_platformer.engine import entities, world  # noqa: F401
        from jev_platformer.telemetry import extractor  # noqa: F401
        result['timings']['validation_and_loading_seconds']=time.perf_counter()-started
        phase='simulate_controls'
        simulation_start=time.perf_counter()
        for offset in (0,-16,16):
            for raw in ('right_run','right_jump','right_run_jump','left','noop'):
                for assist,cadence in ((False,8),(True,1),(True,8)):
                    case_start=time.perf_counter()
                    row=run(raw,assist,cadence,offset)
                    row['simulation_wall_seconds']=time.perf_counter()-case_start
                    result['cases'].append(row)
        result['timings']['simulation_seconds']=time.perf_counter()-simulation_start
        result['status']='success'
    except Exception as exc:
        result['status']='error'
        # Do not persist exception messages: paths and process output can contain private data.
        result['error']={'phase':phase,'type':type(exc).__name__,
            'message':'Control audit failed; exception text and paths omitted.'}
    finally:
        result['timings']['total_wall_seconds']=time.perf_counter()-started
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if result['status']!='success':
        print(json.dumps(result['error']))
        return 1
    rows=result['cases']
    for row in rows:
        if row['start_offset']==0: print(json.dumps(row))

if __name__=='__main__': sys.exit(main())
