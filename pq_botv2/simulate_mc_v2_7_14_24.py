'''


python simulate_mc_v2_7_14_24.py --run_id 10 --num_battles 1000000000 --run_tag v2_mc_3kmax


# to do before prod
need to create the eval dict

# prod to do stuff
need to sort active each turn
need to load the dict
need to query

'''
import argparse
import time
import copy
import numpy as np
import time
import math
import pprint
import pickle
#from scipy.stats import chi2_contingency
import os


from vgc.datatypes.Objects import PkmTeam, Pkm, GameState, Weather
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator

from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER, MAX_HIT_POINTS, MOVE_MAX_PP, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmMove, Pkm, PkmTeam, GameState, Weather
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, \
    N_TYPES, N_STATUS, N_STATS, N_ENTRY_HAZARD, N_WEATHER, PkmStatus, PkmEntryHazard


def main(args):

    # set up one time things
    run_id = args.run_id
    num_battles = args.num_battles

    time_int = int(time.time())
    run_tag = args.run_tag + '_' + str(run_id) + '_' + str(time_int)

    # execute the loop
    winner_dict, action_state_results_dict = build_train_eval_loop(
        num_battles, run_tag, time_int, is_save=True)



def build_train_eval_loop(
        num_battles, run_tag, time_int, is_save=True):
    '''
    '''
    action_state_results_dict = {}

    # constants
    agent_move_attack_key = 'attack'
    agent_move_swap_party_0_key = 'swap_0'
    agent_move_swap_party_1_key = 'swap_1'
    action_dict_count_key = 'count'
    action_dict_wins_key = 'sum_wins'

    winner_dict = {
        0:0,
        1:0,
        -1:0
    }

    team_generator = RandomTeamGenerator(2)

    # constants for overall loop
    max_episode_steps = 250
    agent_index = 0
    checkpoint_counter = 0
    start_time = time.time()

    max_number_state_keys = 3000

    for battle_idx in range(num_battles):
        
        agent_team = team_generator.get_team().get_battle_team([0,1,2])
        opp_team = team_generator.get_team().get_battle_team([0,1,2])

        # set new environment with teams
        env = PkmBattleEnv((agent_team, opp_team), encode=(False, False)) 

        game_state, info = env.reset()

        # initialize for each new battle
        is_state_to_log_chosen = False
        state_key_to_log = None
        action_to_log = None

        # order to sort the pkm when making the party
        # never use the opp party order unless moves are revealed for both party pkm
        agent_pkm_party_sort_list, _ = get_pkm_id_sort_list(game_state[0].teams[0].party)

        # step through the game for each battle
        for episode_step in range(max_episode_steps):

            num_agent_pkm = get_num_active_pokemon(game_state[0])
            num_opp_pkm = get_num_active_pokemon(game_state[1])

            opp_action, _ = get_best_active_damage_action(game_state[1])

            if num_agent_pkm >= 2 and num_opp_pkm >= 2:

                # get the current state key
                current_step_state_key = get_state_key_from_game_state(game_state[0], agent_pkm_party_sort_list, num_opp_pkm)

                if not is_state_to_log_chosen:
                    # to do: DRY
                    if current_step_state_key in action_state_results_dict:
                        num_attacks = action_state_results_dict[current_step_state_key].get(agent_move_attack_key, {}).get(action_dict_count_key, 0)
                        num_swap_0 = action_state_results_dict[current_step_state_key].get(agent_move_swap_party_0_key, {}).get(action_dict_count_key, 0)

                        if num_agent_pkm == 2:
                            if num_attacks >= max_number_state_keys and num_swap_0 >= max_number_state_keys:
                                # logged this state enough, agent attacks
                                # do not set state or action to log
                                agent_action, _ = get_best_active_damage_action(game_state[0])
                            else:
                                is_state_to_log_chosen = True
                                state_key_to_log = current_step_state_key

                                if num_attacks > num_swap_0:
                                    # agent swaps to party pkm that is not fainted
                                    action_to_log = agent_move_swap_party_0_key
                                    if game_state[0].teams[0].party[0].fainted():
                                        agent_action = 5
                                    else:
                                        agent_action = 4
                                else:
                                    # agent attacks
                                    action_to_log = agent_move_attack_key
                                    agent_action, _ = get_best_active_damage_action(game_state[0])

                        elif num_agent_pkm == 3:
                            num_swap_1 = action_state_results_dict[current_step_state_key].get(agent_move_swap_party_1_key, {}).get(action_dict_count_key, 0)

                            if num_attacks >= max_number_state_keys and num_swap_0 >= max_number_state_keys and \
                                num_swap_1 >= max_number_state_keys:
                                # logged this state enough, agent attacks
                                agent_action, _ = get_best_active_damage_action(game_state[0])
                            else:
                                is_state_to_log_chosen = True
                                state_key_to_log = current_step_state_key

                                if num_attacks > num_swap_0:
                                    # agent swaps 0
                                    action_to_log = agent_move_swap_party_0_key
                                    agent_action = 4
                                elif num_attacks > num_swap_1:
                                    # agent swaps 1
                                    action_to_log = agent_move_swap_party_1_key
                                    agent_action = 5
                                else:
                                    # agent attacks
                                    action_to_log = agent_move_attack_key
                                    agent_action, _ = get_best_active_damage_action(game_state[0])

                    else:
                        # first time seeing this current state choose to attack
                        state_key_to_log = current_step_state_key
                        action_to_log = agent_move_attack_key
                        is_state_to_log_chosen = True
                        agent_action, _ = get_best_active_damage_action(game_state[0])
                else:
                    # already chosen the log state and action. play out rest of the battle
                    # agent attacks
                    agent_action, _ = get_best_active_damage_action(game_state[0])
            else:
                #only 1 pkm left on a side, 
                if not is_state_to_log_chosen:
                    # nothing to log this battle, go to next battle
                    break
                # agent attacks
                agent_action, _ = get_best_active_damage_action(game_state[0])

            # enter action and step the env
            action_list = [agent_action, opp_action]
            game_state, _not_used_reward, terminated, truncated, info = env.step(action_list)  # for inference, we don't need reward

            if episode_step == max_episode_steps - 1:
                print('Warning: max steps reached, ending battle (unlikely to happen)')
                terminated = True
                break

            if terminated:
                winner = env.winner
                if winner == -1:
                    break

                if winner == agent_index:
                    agent_win_int = 1
                else:
                    agent_win_int = 0
                
                if state_key_to_log is not None and action_to_log is not None:
                    action_state_results_dict = add_results_to_action_state_dict(action_state_results_dict,
                                                                                state_key_to_log,
                                                                                action_to_log,
                                                                                agent_win_int, 
                                                                                action_dict_count_key,
                                                                                action_dict_wins_key)
                else:
                    print("Error: not logging anything?")
                
                if winner in winner_dict:
                    winner_dict[winner] += 1
                # end battle
                break

        if battle_idx % 1000000 == 0 and battle_idx > 0:
            checkpoint_counter += 1
            checkpoint_save_int = checkpoint_counter % 3
            save_object_as_pkl(action_state_results_dict,
                f'3v3_results/{run_tag}_action_state_results_dict_checkpoint_{checkpoint_save_int}')

    if is_save:
        save_object_as_pkl(action_state_results_dict, f'3v3_results/{run_tag}_action_state_results_dict')
        save_object_as_pkl(winner_dict, f'3v3_results/{run_tag}_winner_dict')

    end_time = time.time()
    print(f"Time to run {(end_time - start_time) / 60:.3f} minutes")
    print(f"Time to run {(end_time - start_time) / num_battles:.3f} seconds per battle")
    print(f"Time to run {((end_time - start_time) / num_battles / 60 / 60) * 1000000:.3f} hours per million battles")

    print(winner_dict)

    return winner_dict, action_state_results_dict


def get_state_key_from_game_state(agent_game_state, agent_pkm_party_sort_list, num_opp_pkm):
    '''
    '''
    hide_default_value = -1
    fainted_default_value = 0
    max_ttf = 4

    # debugging only
    best_damage_list = []
    hp_list = []

    weather = agent_game_state.weather.condition
    agent_team = agent_game_state.teams[0]
    opp_team = agent_game_state.teams[1]

    if agent_pkm_party_sort_list[0] > agent_pkm_party_sort_list[1]:
        agent_party_list_sorted = [agent_team.party[1], agent_team.party[0]]
    else:
        agent_party_list_sorted = [agent_team.party[0], agent_team.party[1]]

    agent_team_list = [agent_game_state.teams[0].active] + agent_party_list_sorted
    opp_team_list = [agent_game_state.teams[1].active]

    # get agent parts of the state # ugg... DRY
    agent_normalized_hp_list = []
    agent_ttf_list = []

    for agent_pkm_idx, agent_pkm in enumerate(agent_team_list):
        if agent_pkm.fainted():
            # will always know if fainted or not
            agent_normalized_hp_list.append(0)
        else:
            pkm_hp = agent_pkm.hp
            if pkm_hp <= 240.:
                agent_normalized_hp_list.append(0)
            elif pkm_hp <= 336.:
                agent_normalized_hp_list.append(1)
            else:
                agent_normalized_hp_list.append(2)

        for opp_pkm_idx, opp_pkm in enumerate(opp_team_list):
            if opp_pkm.fainted() or agent_pkm.fainted():
                agent_ttf_list.append(fainted_default_value)
            elif not opp_pkm.revealed:
                agent_ttf_list.append(hide_default_value)
                print("error, opp active not revealed")
            else:
                
                best_damage = -np.inf

                if agent_pkm_idx == 0:
                    agent_attack_stage = agent_team.stage[PkmStat.ATTACK]
                else:
                    agent_attack_stage = 0
                
                if opp_pkm_idx == 0:
                    opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
                else:
                    opp_defense_stage = 0

                for move_idx, move in enumerate(agent_pkm.moves):
                    
                    damage = estimate_damage(move.type, agent_pkm.type, move.power, opp_pkm.type, agent_attack_stage,
                                                opp_defense_stage, weather)

                    # Check if the current move has higher damage than the previous best move
                    if damage > best_damage:
                        best_damage = damage

                # used for debugging
                best_damage_list.append(best_damage)
                hp_list.append(opp_pkm.hp)

                if best_damage > 0.:
                    turns_to_faint = math.ceil(opp_pkm.hp / best_damage)

                    # all turns to faint > max value treated as max
                    if turns_to_faint >= max_ttf:
                        turns_to_faint = max_ttf
                else:
                    turns_to_faint = max_ttf

                agent_ttf_list.append(turns_to_faint)

    # get opp parts of the state
    opp_normalized_hp_list = []
    if opp_team_list[0].revealed:
        pkm_hp = opp_team_list[0].hp
        if pkm_hp <= 240.:
            opp_normalized_hp_list.append(0)
        elif pkm_hp <= 336.:
            opp_normalized_hp_list.append(1)
        else:
            opp_normalized_hp_list.append(2)
    else:
        opp_normalized_hp_list.append(hide_default_value)

    
    opp_active_ttf_list = []
    opp_num_moves_revealed_list = []

    for opp_pkm_idx, opp_pkm in enumerate(opp_team_list):
        opp_num_moves_revealed = 0
        for move in opp_team_list[0].moves:
            if move.revealed:
                opp_num_moves_revealed += 1
        opp_num_moves_revealed_list.append(opp_num_moves_revealed)

        for agent_pkm_idx, agent_pkm in enumerate(agent_team_list):
            if agent_pkm.fainted() or opp_pkm.fainted():
                opp_active_ttf_list.append(fainted_default_value)
            elif opp_num_moves_revealed == 0:
                opp_active_ttf_list.append(hide_default_value)
            else:
                best_damage = -np.inf

                if opp_pkm_idx == 0:
                    opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
                else:
                    opp_attack_stage = 0
                
                if agent_pkm_idx == 0:
                    agent_defense_stage = agent_team.stage[PkmStat.DEFENSE]
                else:
                    agent_defense_stage = 0

                for move_idx, move in enumerate(opp_pkm.moves):
                    
                    if move.revealed:
                        damage = estimate_damage(move.type, opp_pkm.type, move.power, agent_pkm.type, opp_attack_stage,
                                                    agent_defense_stage, weather)

                        # Check if the current move has higher damage than the previous best move
                        if damage > best_damage:
                            best_damage = damage

                # used for debugging
                best_damage_list.append(best_damage)
                hp_list.append(agent_pkm.hp)

                if best_damage > 0.:
                    turns_to_faint = math.ceil(agent_pkm.hp / best_damage)

                    # all turns to faint > max value treated as max
                    if turns_to_faint >= max_ttf:
                        turns_to_faint = max_ttf
                else:
                    turns_to_faint = max_ttf

                opp_active_ttf_list.append(turns_to_faint)

    if len(agent_normalized_hp_list) != 3 or len(opp_normalized_hp_list) != 1:
        print("Error: agent or opp hp list not correct length")

    if len(opp_active_ttf_list) != 3 or len(agent_ttf_list) != 3:
        print("Error: agent or opp ttf list not correct length")
    
    if len(opp_num_moves_revealed_list) != 1:
        print("Error: opp num moves revealed list not correct length")

    state_key = tuple(agent_ttf_list + opp_active_ttf_list 
                      +  opp_num_moves_revealed_list + [num_opp_pkm]
                      + agent_normalized_hp_list + opp_normalized_hp_list)

    return state_key


def get_pkm_id_sort_list(team_party_list):
        '''
        Reduce state size by sorting the pkm
        '''
        if len(team_party_list) <= 1:
            print("Error party len is only 1 for sort")
            return [0, 1], ['0', '1']
        
        pkm_id_list = []
        pkm_sort_list = []

        if len(team_party_list) == 2:
            if team_party_list[0].max_hp > team_party_list[1].max_hp:
                pkm_sort_list = [0, 1]
            elif team_party_list[0].max_hp < team_party_list[1].max_hp:
                pkm_sort_list = [1, 0]
            else:
                # hp is equal, sort by move differences
                for i, pkm in enumerate(team_party_list):
                    pkm_id = ''
                    for j, move in enumerate(pkm.moves):
                        pkm_id += str(move.type) + str(move.power)

                    pkm_id_list.append(pkm_id)

                if pkm_id_list[0] > pkm_id_list[1]:
                    pkm_sort_list = [1, 0]
                else:
                    pkm_sort_list = [0, 1]
        else:
            print("Error party len is not 2 for sort")
            return [0, 1], ['0', '1']

        return pkm_sort_list, pkm_id_list


def get_best_active_damage_action(g: GameState):
    '''
    '''
    # Get weather condition
    weather = g.weather.condition

    # Get my Pokémon team
    my_team = g.teams[0]
    my_pkms = [my_team.active] #+ my_team.party

    # Get opponent's team
    opp_team = g.teams[1]
    opp_active = opp_team.active

    opp_active_type = opp_active.type
    opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

    # Iterate over all my Pokémon and their moves to find the most damaging move
    best_dmg_list = []
    best_move_list = []

    for i, pkm in enumerate(my_pkms):
        # Initialize variables for the best move and its damage
        best_damage = -np.inf
        best_move_id = -1

        if i == 0:
            my_attack_stage = my_team.stage[PkmStat.ATTACK]
        else:
            my_attack_stage = 0

        for j, move in enumerate(pkm.moves):
            
            damage = estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                        opp_defense_stage, weather)
            
            # Check if the current move has higher damage than the previous best move
            if damage > best_damage:
                best_move_id = j + i * 4 # think for 2024 j is 0 to 3 for each
                best_damage = damage

        # get best move and dmg for each pokemon
        best_dmg_list.append(best_damage)
        best_move_list.append(best_move_id)

    active_pkm_best_move_id = best_move_list[0]

    if active_pkm_best_move_id < 0 or active_pkm_best_move_id > 3:
        print(f"Error: best move id { active_pkm_best_move_id } not in expected range")
        active_pkm_best_move_id = 0

    return active_pkm_best_move_id, best_dmg_list


def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    '''
    from the repo
    '''
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power

    #print(damage, move_type, pkm_type, move_power, opp_pkm_type, attack_stage, defense_stage, weather)
    return damage


def add_results_to_action_state_dict(action_state_results_dict,
        state_key_to_log,
        move_to_log,
        agent_win_int, 
        action_dict_count_key,
        action_dict_wins_key):

    if state_key_to_log in action_state_results_dict:

        if move_to_log in action_state_results_dict[state_key_to_log]:
            action_state_results_dict[state_key_to_log][move_to_log][action_dict_count_key] += 1
            action_state_results_dict[state_key_to_log][move_to_log][action_dict_wins_key] += agent_win_int
        else:
            action_state_results_dict[state_key_to_log][move_to_log] = {
                action_dict_count_key: 1,
                action_dict_wins_key: agent_win_int
            }
            
    else:

        action_state_results_dict[state_key_to_log] = {}
        action_state_results_dict[state_key_to_log][move_to_log] = {
            action_dict_count_key: 1,
            action_dict_wins_key: agent_win_int
        }

    return action_state_results_dict


def get_num_active_pokemon(game_state):
    num_active_pkm = 0

    team_zero = game_state.teams[0]
    team_list = [team_zero.active] + team_zero.party

    for i, pkm in enumerate(team_list):
        if not pkm.fainted() or pkm.hp > 0.0:
            num_active_pkm += 1

    return num_active_pkm


def save_object_as_pkl(object_to_save, save_tag):
    '''
    Save object a pickle file
    '''
    save_filename = f'{save_tag}.pickle'
    save_path = os.path.join('G:\\', '3v3_vgc_saves_71124', save_filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as handle:
        print("saving: ", save_path)
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set the run_id for args
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--num_battles', type=int, default=10)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    main(args)
