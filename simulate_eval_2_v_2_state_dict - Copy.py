'''
Sim 2 vs 2 games to create a state dict

python simulate_eval_2_v_2_state_dict.py --run_id 0 --num_battles 10 --run_tag test --is_eval 0

working
    TEST add hidden value that can go to state
        only hidden if pkm is still alive
    DONE add hidden randomizer that randomizes hidden at the beginning of the battle
        thinking set it to 0.3. might happen less but state is so much condensed doesn't need to hit it as much

    TEST reveal roll have the hidden stuff be different

    TEST then put those hidden bools into the get_turns_to_faint_list function

    DONE OOOF, I misunderstood the state
        need to be 100% on the revealed or not
        then plan accordingly

        ideas:
            once seen one action against pkm
                if against active, assume best action is taken so know the turns to faint
                for other pkm, could either
                    assume best action (it likely is)
                    or encode as partially known
            revealed now has to parts to it
                if pkm is revealed, turns to faint
                of know moves are revealed, that is hidden
                if partial moves are revealed,
                if full moves are revealed

            I think the compromise is:
                just change my revealed options up front to be
                    for each pkm 2 things that can be revealed or not
                        hp, all moves (not true in practice but assume it is)
                    I think these are the 5 states
                    1,1,1,1: all revealed
                        90% here
                    1,1,1,0: opp party moves hidden
                        3.5% here
                    1,0,1,1: active moves hidden, opp known
                        3.5% here
                    1,0,0,0: only active hp known
                        1% here
                    1,1,0,0: active revealed, opp hidden
                        1% here
                    1,0,1,0: active moves, hidden, opp party moves hidden
                        1% here

                then swaps are allowed at PRD only when
                    one of those key components is revealed
                    if a move with more dmg than the prior best dmg move to active pkm is revealed
                    in training, don't worry about it
                    

    DONE review
        need to review the functions
            get_turns_to_faint_list
            
  
    step through and test
    smoke test
    run small sim
    run large sim

    test and look through this again, so tired

    fix my sim with dmg to actually calculate the dmg turns to faint accurately
    not understanding why some dmg is being taken. not sure hwo to tell wtf is going on
    test actual dmg done vs the function


to do
    use logic to find best action in some end game states
        can use this to test the results

    for eval:
        write the eval code where read from a dict

        write code where don't swap if all things the same
            ie if turns to kill same between active and passive
            and if turns to get killed by active are the same
            and turns to kill backup are the same (or unknown)
            and if turns to get killed by back are the same or unknown

        eval need to consider when more state info is revealed as to when swaps are allowed

	use the function that gets the dmg to get the state tuple in main loop


if doing more self play or iterations
    can allow action selection again after a swap and sometimes store the states and state list

optimize
    can combine state, best action and best dmg into one function
    maybe only call for state for first action. later actions just always attack
        stop storing states when only 1 pkm left maybe as always attacks at that point

for prd
    need to consider revealed when estimating opp dmg

Later
    2v3 and 2v3
    unit tests
    better review
    more alg (think this through)

'''
import argparse
import time
import copy
import numpy as np
import time
import math
import pprint
import pickle


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
    is_eval = bool(args.is_eval)
    eval_dict_path = args.eval_dict_path

    if eval_dict_path is not None and eval_dict_path != '':
        action_dict_to_copy = load_pkl_object(eval_dict_path)
    else:
        action_dict_to_copy = None

    time_int = int(time.time())
    run_tag = "2v2_" + args.run_tag + '_' + str(run_id) + '_' + str(time_int)

    # execute the loop
    winner_dict, action_state_results_dict, pkm_env_action_dict = build_train_eval_loop(
        num_battles, is_eval, run_tag, time_int, action_dict_to_copy, is_save=True)


def build_train_eval_loop(num_battles, is_eval, run_tag, time_int, action_dict_to_copy=None, is_save=True):
    '''
    Build train / eval loop
    print results
    save results
    '''
    action_state_results_dict = {}

    # constants
    agent_first_move_attack_key = 'attack'
    agent_first_move_swap_key = 'swap'
    max_turns_to_faint_value = 8

    if is_eval and action_dict_to_copy is not None:
        action_lookup_dict = copy.deepcopy(action_dict_to_copy)

    winner_dict = {
        0:0,
        1:0,
        -1:0
    }

    pkm_env_action_dict = {
        0:{},
        1:{},
    }

    team_generator = RandomTeamGenerator(2)

    max_episode_steps = 250
    agent_index = 0
    start_time = time.time()

    for battle_idx in range(num_battles):
        
        agent_team = team_generator.get_team().get_battle_team([0, 1, ])
        opp_team = team_generator.get_team().get_battle_team([0, 1, ])

        # set new environment with teams
        env = PkmBattleEnv((agent_team, opp_team), encode=(False, False)) 

        game_state, info = env.reset()

        # initialize for each new battle
        is_first_move = True
        state_list = []

        # opp can have 3 things revealed or not:
        # active moves, party hp, party moves
        # default to all revealed 89% of the time
        # 10% of the time, at least one of the other three is hidden
        reveal_roll = np.random.rand()

        is_reveal_opp_active_moves = True
        is_reveal_opp_party_hp = True
        is_reveal_opp_party_moves = True

        if reveal_roll < .11:
            if reveal_roll >= .075:
                is_reveal_opp_active_moves = False
            elif reveal_roll >= .04:
                is_reveal_opp_party_moves = False
            elif reveal_roll >= .025:
                is_reveal_opp_party_hp = False
                is_reveal_opp_party_moves = False
            elif reveal_roll >= .01:
                is_reveal_opp_active_moves = False
                is_reveal_opp_party_moves = False
            else:
                is_reveal_opp_active_moves = False
                is_reveal_opp_party_hp = False
                is_reveal_opp_party_moves = False

        # step through the game for each battle
        for episode_step in range(max_episode_steps):
            if is_first_move:
                if np.random.rand() < 0.5:
                    agent_pre_env_action = 0
                    agent_first_move = agent_first_move_attack_key
                else:
                    agent_pre_env_action = 1
                    agent_first_move = agent_first_move_swap_key
                is_first_move = False
            else:
                agent_pre_env_action = 0

            # get best actions
            agent_env_action, agent_team_best_damage_list = turn_agent_action_into_env_action(
                agent_pre_env_action, game_state[0])

            opp_action, opp_best_damage_list = get_best_active_damage_action(game_state[1])

            # get state key
            state_list_agent = get_turns_to_faint_list(game_state[0], game_state[1], max_turns_to_faint_value,
                is_reveal_team_1_active_moves=True,
                is_reveal_team_1_party_moves=True,
                is_reveal_team_2_party_hp=is_reveal_opp_party_hp)

            state_list_opp = get_turns_to_faint_list(game_state[1], game_state[0], max_turns_to_faint_value,
                is_reveal_team_1_active_moves=is_reveal_opp_active_moves,
                is_reveal_team_1_party_moves=is_reveal_opp_party_moves,
                is_reveal_team_2_party_hp=True)

            state_key = tuple(state_list_agent + state_list_opp)
            state_list.append(state_key)

            if is_eval:
                print("Not yet implemented")
                assert 1 == 0
                # # only allow swaps under certain criteria, see above and think through
                # is_alowed_to_use_dict = ...
                # if is_allowed_to_use_dict:
                #     # access dict, get statistics to see if allowed
                #     pass
                # else:
                #     # continue to attack with best action
                #     agent_pre_env_action = 0

                # turn the agent action into an env action
                agent_env_action, _ = turn_agent_action_into_env_action(agent_pre_env_action, game_state[0])

                # store actions taken for debugging purposes
                pkm_env_action_dict = add_action_to_pkm_env_action_dict(agent_env_action, pkm_env_action_dict, 0)
                pkm_env_action_dict = add_action_to_pkm_env_action_dict(opp_action, pkm_env_action_dict, 1)

            # check that actions are valid
            check_that_actions_are_valid(agent_pre_env_action, agent_env_action, True, agent_team_best_damage_list)
            check_that_actions_are_valid(0, opp_action, False, opp_best_damage_list)

            # enter action and step the env
            action_list = [agent_env_action, opp_action]
            game_state, _not_used_reward, terminated, truncated, info = env.step(action_list)  # for inference, we don't need reward

            if episode_step == max_episode_steps - 1:
                print('Warning: max steps reached')
                terminated = True

            if terminated:
                winner = env.winner
                if winner == agent_index:
                    agent_win_int = 1
                else:
                    agent_win_int = 0

                action_state_results_dict = add_results_state_list_to_action_dict(
                    action_state_results_dict, state_list, agent_first_move, agent_win_int,
                    agent_first_move_attack_key, agent_first_move_swap_key)

                if winner in winner_dict:
                    winner_dict[winner] += 1
                # end battle
                break

    end_time = time.time()
    print(f"Time to run {(end_time - start_time) / 60:.3f} minutes")
    print(f"Time to run {(end_time - start_time) / num_battles:.3f} seconds per battle")
    print(f"Time to run {((end_time - start_time) / num_battles / 60 / 60) * 1000000:.3f} hours per million battles")

    print(winner_dict)

    if is_save:
        save_object_as_pkl(action_state_results_dict , f'{run_tag}_action_state_results_dict')
        save_object_as_pkl(winner_dict, f'{run_tag}_winner_dict')

        if is_eval:
            save_object_as_pkl(pkm_env_action_dict, f'{run_tag}_pkm_env_action_dict')

    return winner_dict, action_state_results_dict, pkm_env_action_dict


def add_results_state_list_to_action_dict(action_dict, state_list, agent_first_move, agent_win_int,
    agent_first_move_attack_key, agent_first_move_swap_key):
    '''
    '''
    count_key = "count"
    sum_wins_key = "sum_wins"

    for state_list_index, state_key in enumerate(state_list):
        # first move can be swap or attack, then attacking rest of the way
        if state_list_index == 0:
            move_key = agent_first_move
        else:
            move_key = agent_first_move_attack_key
        
        if state_key in action_dict:
            if agent_first_move in action_dict[state_key]:
                action_dict[state_key][move_key][sum_wins_key] += agent_win_int
                action_dict[state_key][move_key][count_key] += 1
            else:
                action_dict[state_key][move_key] = {}
                action_dict[state_key][move_key][sum_wins_key] = agent_win_int
                action_dict[state_key][move_key][count_key] = 1
        else:
            action_dict[state_key] = {}
            action_dict[state_key][move_key] = {}
            action_dict[state_key][move_key][sum_wins_key] = agent_win_int
            action_dict[state_key][move_key][count_key] = 1

    return action_dict


def get_turns_to_faint_list(team_1_game_state, team_2_game_state, max_turns_to_faint_value,
        is_reveal_team_1_active_moves,
        is_reveal_team_1_party_moves,
        is_reveal_team_2_party_hp):
    '''
    put in zero for the fainted
        handle the check for fainted in HP

    '''
    hide_default_value = -1
    fainted_default_value = 0

    # Get weather condition
    weather = team_1_game_state.weather.condition

    # Get my Pokémon team
    team_1 = team_1_game_state.teams[0]
    team_1_pkm_list = [team_1.active] + team_1.party

    # Get opponent's team
    team_2 = team_2_game_state.teams[0]
    team_2_pkm_list = [team_2.active] + team_2.party

    # Iterate over all my Pokémon and their moves to find the most damaging move
    best_damage_list = []
    turns_to_faint_list = []
    hp_list = []

    for team_1_pkm_index, team_1_pkm in enumerate(team_1_pkm_list):

        for team_2_pkm_index, team_2_pkm in enumerate(team_2_pkm_list):

            # reveal the state under these conditions:
            # team 1 active moves revealed and team 2 active HP is revealed (active HP is always revealed)
            # team 1 active moves revaled and team 2 party HP is revealed
            # team 1 party moves revealed and team 2 active HP is revealed (active HP is always revealed)
            # team 1 party moves revealed and team 2 party HP is revealed
            is_reveal_part_of_state = False

            if team_2_pkm_index == 0:
                # team 2 active pkm
                if team_1_pkm_index == 0 and is_reveal_team_1_active_moves:
                    is_reveal_part_of_state = True
                elif team_1_pkm_index == 1 and is_reveal_team_1_party_moves:
                    is_reveal_part_of_state = True
            elif team_2_pkm_index == 1 and is_reveal_team_2_party_hp:
                # team 2 party pkm
                if team_1_pkm_index == 0 and is_reveal_team_1_active_moves:
                    is_reveal_part_of_state = True
                elif team_1_pkm_index == 1 and is_reveal_team_1_party_moves:
                    is_reveal_part_of_state = True

            if team_1_pkm.fainted() or team_1_pkm.hp <= 0.0:
                # will always know if fainted or not
                turns_to_faint_list.append(fainted_default_value)
                continue
            elif not is_reveal_part_of_state:
                # state is not revealed, so hide the value
                turns_to_faint_list.append(hide_default_value)
                continue
            else:
                # Initialize variables for the best move and its damage
                best_damage = -np.inf

                # this part of state is revealed, calculate it and add to turns_to_faint_list
                if team_1_pkm_index == 0:
                    team_1_attack_stage = team_1.stage[PkmStat.ATTACK]
                else:
                    team_1_attack_stage = 0
                
                if team_2_pkm_index == 0:
                    team_2_defense_stage = team_2.stage[PkmStat.DEFENSE]
                else:
                    team_2_defense_stage = 0

                for move_index, move in enumerate(team_1_pkm.moves):
                    
                    damage = estimate_damage(move.type, team_1_pkm.type, move.power, team_2_pkm.type, team_1_attack_stage,
                                                team_2_defense_stage, weather)

                    # Check if the current move has higher damage than the previous best move
                    if damage > best_damage:
                        best_damage = damage

                # get best dmg for each pokemon
                best_damage_list.append(best_damage)
                hp_list.append(team_2_pkm.hp)

                if best_damage > 0.:
                    turns_to_faint = math.ceil(team_2_pkm.hp / best_damage)

                    # all turns to faint > max value treated as max
                    if turns_to_faint >= max_turns_to_faint_value:
                        turns_to_faint = max_turns_to_faint_value
                    elif turns_to_faint > 5:
                        # group all turns to faint between 5 and max value as 5
                        turns_to_faint = 5
                else:
                    turns_to_faint = max_turns_to_faint_value

                turns_to_faint_list.append(turns_to_faint)

    #   {1: 5694845,
    #  2: 1948930,
    #  3: 268534,
    #  4: 61252,
    #  5: 13357,
    #  6: 7598,
    #  7: 2896,
    #  8: 1173,
    #  9: 498,
    #  10: 381,
    #  11: 128,
    #  12: 136,
    #  13: 99,
    #  14: 50,
    #  15: 20,
    #  16: 19,
    #  17: 5,
    #  18: 13,
    #  19: 2,
    #  20: 7,
    #  22: 2,
    #  23: 3,
    #  25: 1,
    #  28: 1,
    #  30: 2,
    #  100: 48}
    # print(turns_to_faint_list)
    # print(best_damage_list)
    # print(hp_list)
  
    return turns_to_faint_list


# def get_opp_active_pkm_id(agent_game_state):
#     '''
#     Tell if opp is same pkm or not
#     AFAIK this is the only way to do it and not 100% accurate as the pkm_id is always -1
#     This won't pick up on swaps between pkm with the exact same characteristics but that is rare
#     Could may be add an HP check as well
#     opp_active_pkm.type will be the type of the first move so don't need to include that
#     only works if all moves are revealed when pkm is active, not sure if that is true
#     '''
#     opp_active_pkm = agent_game_state.teams[1].active

#     opp_active_list = []

#     for move in opp_active_pkm.moves:
#         opp_active_list.append(move.power)
#         opp_active_list.append(move.type)

#     opp_active_id = tuple(opp_active_list)

#     return opp_active_id


def load_pkl_object(pkl_path):
    '''
    Load a pickle object
    '''
    with open(pkl_path, 'rb') as handle:
        return pickle.load(handle)

def save_object_as_pkl(object_to_save, save_tag):
    '''
    Save object a pickle file
    '''
    with open(f'{save_tag}.pickle', 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_that_actions_are_valid(pre_env_action, env_action, is_agent, best_damage_list):
    if pre_env_action == 1 and env_action != 4:
        print("Error pre env action is 1 but env action is not 4 ")
        if not is_agent:
            print("Error opp pre_env action is not always 0")
    
    if pre_env_action == 0:
        if (env_action < 0 or env_action > 3):
            print("Error pre envaction is 0 but env action is not 0 to 3 ")
        
        if len(best_damage_list) == 0:
            print("Error pre_env action is 0 but best damage list is empty")
        elif best_damage_list[0] < 0:
            print("Error agent action is 0 but best damage is negative")


def turn_agent_action_into_env_action(action, agent_game_state):
    '''
    Action values are
    0: select best move
    1: switch to first pkm
    2: switch to second pkm

    Env actions are
    0 to 3: action of active pokm
    4: switch to first pkm
    5: switch to second pkm
    '''
    # always get best move and action dmg list
    best_active_action, best_damage_list = get_best_active_damage_action(agent_game_state)

    if action == 0:
        # get best dmg action
        action = best_active_action
    else:
        # switch to first or second pkm if alive
        if action == 1 or action == 2:
            pkm = agent_game_state.teams[0].party[action-1]
            if pkm.fainted() or pkm.hp <= 0.0:
                action = best_active_action
            else:
                action = action + 3
        else:
            action = best_active_action

    return action, best_damage_list


def get_best_active_damage_action(g: GameState):
    '''
    '''
    # Get weather condition
    weather = g.weather.condition

    # Get my Pokémon team
    my_team = g.teams[0]
    my_pkms = [my_team.active] + my_team.party

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


def add_action_to_pkm_env_action_dict(env_action, my_dict, team_key):
    if env_action in my_dict[team_key]:
        my_dict[team_key][env_action] += 1
    else:
        my_dict[team_key][env_action] = 1

    return my_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set the run_id for args
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--num_battles', type=int, default=10)
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--is_eval', type=int, default=0, choices=[0, 1])
    parser.add_argument('--eval_dict_path', type=str, default='')
    args = parser.parse_args()

    main(args)
