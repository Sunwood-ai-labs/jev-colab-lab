"""Local physics controls, NOT model/GPU results. Reads an immutable game checkout."""
import argparse
import collections
import json
import os
from pathlib import Path
import random
import subprocess
import sys

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
    actual=subprocess.check_output(['git','-C',str(args.game),'rev-parse','HEAD'],text=True).strip()
    if actual!=PIN: raise ValueError('Game revision mismatch')
    if subprocess.check_output(['git','-C',str(args.game),'status','--porcelain'],text=True).strip():
        raise ValueError('Game checkout must be clean')
    os.environ['SDL_VIDEODRIVER']='dummy'; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    sys.path.insert(0,str(args.game/'src'))
    rows=[]
    for offset in (0,-16,16):
        for raw in ('right_run','right_jump','right_run_jump','left','noop'):
            rows.append(run(raw,False,8,offset))
            for cadence in (1,8): rows.append(run(raw,True,cadence,offset))
    result={'kind':'local deterministic physics controls; no model inference; no Colab',
        'game_commit':PIN,'seed':42,'max_frames':1800,'cases':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for row in rows:
        if row['start_offset']==0: print(json.dumps(row))

if __name__=='__main__': main()
