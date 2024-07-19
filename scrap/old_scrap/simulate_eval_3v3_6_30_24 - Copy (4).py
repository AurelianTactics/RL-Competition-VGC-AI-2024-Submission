'''
Sim 3v3, 2v2, or 2v3 games to create a state dict

Can also evaluate the state dict

python simulate_eval_3_v_3_state_dict.py --agent_team_size 2 --opp_team_size 2 --run_id 10 --num_battles 12 --run_tag 3v3test --is_eval 0


# eval run
prior
python simulate_eval_2_v_2_state_dict.py --agent_team_size 2 --opp_team_size 2 --run_id 1000 --num_battles 10000 --run_tag evalsmoke --is_eval 1 --eval_dict_path 2v2_state_dict_results\2v2_test_0_1719717878_action_state_results_dict.pickle

working
    
    TEST works with 2v2 and 3v3
        TEST need to store the 2v3 and 3v2 results differently
            for the collapse to 2v2 need to store that in similar way to 3v3
            for the ones that run to the end, need to store the wins and counts

        TEST need to do the state stuff
            STOPPED HERE
            MEH TOO MANY STATES hp normalized ttf
            TEST hp normalized
            DONE with the new revealed or not

        DONE the 3v3 action dict addition
            DONE write the draft of this code
            review this again, getting a bit tired

        TEST the new add_rsults for non 3v3

        TEST add stuff for num pkm on each side

        MEH need sub directory for savings based on tag then num pkm
        TEST args for num pkm
        
        MEH what size state? shoudl I collapse 2v3 and 3v3 down?
        TEST how to handle the reveal rolls
        TEST IN ACTION ROLL how to code swap 1 and swap 2
        TEST in 3v3 can reset after 1 pkm faints, in 2v3 have to run til end to see how 1v3 and 3v1 work out
            3v1 probably doesn't matter
    
    TEST if 10k in a state and at elast 1k of each action, then can continue


    DONE LOOKS FINE test actual dmg done vs the function
        look through the env code to see if missing anything
        should be easy enough. do a ubnch of new envs and resets, attack from both sides and see if the dmg is as expected
            maybe randomize the moves as well and account for hp norm
        working through it. need to test further

    TEST could sort the party pkm but gets a little complicated
        would reduce states by half. shame that not an ID for this

        TEST can sort the knowns
            could make one by doing power and type concat and then sort the list for active and for party
                gets complicated as hell for the active party and the action though
                maybe it works fine as long as the sort works the same way? would get hte same state

        DONE THIS NEVER OCCURS WHERE IT MATTERS could move fainted to the end
            for agent team, can do that in the loop
                maybe do the list differently
                but would need to account for the action
                    maybe just account for that in prd
                would never do thea gent fainted in 3v3 or 2v2 (where it matters) or 2v3 where it matters
                for opp team, also wouldn't have it where it matters

        could move unknown to the end
            for agent team active never occurs but passive might
            could just make the state space for hidden and reveal symmetric
                then at prod time would organize it that way
                could get tricky but leaning towards this


    review
        think through the hidden and not part again. effing A that is confusing

    smoke test
        2v2
        2v3
        3v2
        3v3
    step through
        2v2
        2v3
        3v2
        3v3

    test and look through this again, so tired




to do


for eval:
    don't do the eval look up in pr
    not important but the swap stuff is being weird

    
    
    
for prd
    how to collapse the results down for 2v2 and 3v3
    do eval with prd set up
    need to calculate the eval look up before
    need to consider revealed when estimating opp dmg
    need to clip the state to match 2v2 state
        ie when it is 2v2, clip the fainted pkm things to just show the 2v2
    sort the party lists the same way
        unknown at the end
        known using the id function and the swap list where applicable
        will ahve to map the swap action correctly

Later
    unit tests
    better review

    if doing more self play or iterations
        meh, I don't think i need to do this
        can allow action selection again after a swap and sometimes store the states and state list

    optimize
        can combine state, best action and best dmg into one function
        maybe only call for state for first action. later actions just always attack
            stop storing states when only 1 pkm left maybe as always attacks at that point
        when enough states in dict and is first move, just move to next simulation

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
from scipy.stats import chi2_contingency


from vgc.datatypes.Objects import PkmTeam, Pkm, GameState, Weather
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator

from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER, MAX_HIT_POINTS, MOVE_MAX_PP, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmMove, Pkm, PkmTeam, GameState, Weather
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, \
    N_TYPES, N_STATUS, N_STATS, N_ENTRY_HAZARD, N_WEATHER, PkmStatus, PkmEntryHazard


def main(args):

    # set up one time things
    agent_team_size = args.agent_team_size
    opp_team_size = args.opp_team_size
    run_id = args.run_id
    num_battles = args.num_battles
    is_eval = bool(args.is_eval)
    eval_dict_path = args.eval_dict_path

    if eval_dict_path is not None and eval_dict_path != '':
        action_dict_to_copy = load_pkl_object(eval_dict_path)
    else:
        action_dict_to_copy = None

    if is_eval and action_dict_to_copy is None:
        print("Error: Eval mode but no dict to copy")
        assert 1 == 0

    time_int = int(time.time())
    run_tag = f"{agent_team_size}_v_{opp_team_size}" + args.run_tag + '_' + str(run_id) + '_' + str(time_int)

    # execute the loop
    winner_dict, action_state_results_dict, pkm_env_action_dict = build_train_eval_loop(
        agent_team_size, opp_team_size,
        num_battles, is_eval, run_tag, time_int, action_dict_to_copy, is_save=True)

    if is_eval:
        pprint.pprint(winner_dict)
        pprint.pprint(pkm_env_action_dict)


def build_train_eval_loop(
        agent_team_size, opp_team_size,
        num_battles, is_eval, run_tag, time_int, action_dict_to_copy=None, is_save=True):
    '''
    Build train / eval loop
    print results
    save results
    '''
    action_state_results_dict = {}

    # constants
    agent_first_move_attack_key = 'attack'
    agent_first_move_swap_party_0_key = 'swap_0'
    agent_first_move_swap_party_1_key = 'swap_1'
    action_dict_count_key = 'count'

    max_turns_to_faint_value = 8

    if agent_team_size == 2:
        agent_team_generator_list = [0,1]
    elif agent_team_size == 3:
        agent_team_generator_list = [0,1,2]
    
    if opp_team_size == 2:
        opp_team_generator_list = [0,1]
    elif opp_team_size == 3:
        opp_team_generator_list = [0,1,2]

    if agent_team_size == 3 and opp_team_size == 3:
        is_3v3_battle = True
    else:
        is_3v3_battle = False

    if agent_team_size == 3 and opp_team_size == 2:
        is_3v2_or_2v3_battle = True
    elif agent_team_size == 2 and opp_team_size == 3:
        is_3v2_or_2v3_battle = True
    else:
        is_3v2_or_2v3_battle = False

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
        
        agent_team = team_generator.get_team().get_battle_team(agent_team_generator_list)
        opp_team = team_generator.get_team().get_battle_team(opp_team_generator_list)

        # set new environment with teams
        env = PkmBattleEnv((agent_team, opp_team), encode=(False, False)) 

        game_state, info = env.reset()

        # initialize for each new battle
        is_first_move = True
        state_list = []
        eval_last_state_key = tuple([-1])

        # order to sort the pkm when making the party
        # never use the opp party order unless moves are revealed for both party pkm
        agent_pkm_party_sort_list, _ = get_pkm_id_sort_list(game_state[0].teams[0].party)
        opp_pkm_party_sort_list, _ = get_pkm_id_sort_list(game_state[1].teams[0].party)

        is_reveal_opp_active_moves, is_reveal_opp_party_0_moves, is_reveal_opp_party_1_moves,\
        is_reveal_opp_party_0_hp, is_reveal_opp_party_1_hp = get_reveal_roll_results(
            agent_team_size, opp_team_size, is_eval)

        # step through the game for each battle
        for episode_step in range(max_episode_steps):
            if is_first_move:
                first_move_roll = np.random.rand()

                if agent_team_size == 2:
                    if first_move_roll < 0.5:
                        agent_pre_env_action = 0
                        agent_first_move = agent_first_move_attack_key
                    else:
                        agent_pre_env_action = 1
                        agent_first_move = agent_first_move_swap_party_0_key
                elif agent_team_size == 3:
                    if first_move_roll < 0.33:
                        agent_pre_env_action = 0
                        agent_first_move = agent_first_move_attack_key
                    elif first_move_roll < 0.66:
                        agent_pre_env_action = 1
                        agent_first_move = agent_first_move_swap_party_0_key
                    else:
                        agent_pre_env_action = 2
                        agent_first_move = agent_first_move_swap_party_1_key
                else:
                    assert 1 == 0
                is_first_move = False
            else:
                agent_pre_env_action = 0

            # get best actions
            agent_env_action, agent_team_best_damage_list = turn_agent_action_into_env_action(
                agent_pre_env_action, game_state[0])

            opp_action, opp_best_damage_list = get_best_active_damage_action(game_state[1])

            # sort the partys to reduce state size
            agent_sorted_party_list, opp_sorted_party_list = get_sorted_team_list(
                game_state[0].teams[0], game_state[1].teams[0],
                agent_pkm_party_sort_list, opp_pkm_party_sort_list,
                is_reveal_opp_party_0_moves, is_reveal_opp_party_1_moves)

            # get state key
            state_list_agent, agent_normalize_hp_list = get_turns_to_faint_list(
                game_state[0],
                game_state[1],
                max_turns_to_faint_value,
                agent_sorted_party_list,
                opp_sorted_party_list,
                hp_normalizer_reveal_list=[True, True, True],
                is_reveal_team_1_active_moves=True,
                is_reveal_team_1_party_0_moves=True,
                is_reveal_team_1_party_1_moves=True,
                is_reveal_team_2_party_0_hp=is_reveal_opp_party_0_hp,
                is_reveal_team_2_party_1_hp=is_reveal_opp_party_1_hp)

            state_list_opp, opp_normalize_hp_list = get_turns_to_faint_list(
                game_state[1],
                game_state[0],
                max_turns_to_faint_value,
                opp_sorted_party_list,
                agent_sorted_party_list,
                hp_normalizer_reveal_list=[True, is_reveal_opp_party_0_hp, is_reveal_opp_party_1_hp],
                is_reveal_team_1_active_moves=is_reveal_opp_active_moves,
                is_reveal_team_1_party_0_moves=is_reveal_opp_party_0_moves,
                is_reveal_team_1_party_1_moves=is_reveal_opp_party_1_moves,
                is_reveal_team_2_party_0_hp=True,
                is_reveal_team_2_party_1_hp=True)

            state_key = tuple(state_list_agent + state_list_opp + agent_normalize_hp_list + opp_normalize_hp_list)
            state_list.append(state_key)

            if is_eval:
                print("TODO: implement this")
                assert 1 == 0
                eval_current_state_key = copy.deepcopy(state_key)

                # allow swaps if state has changed
                # to do: add redundancy confirming num pkm on each side
                if is_swap_allowed_last_current_state(eval_last_state_key, eval_current_state_key):
                    # to do: need to consider how to handle 2v3 and 3v3
                    pass
                    # this was the old 2v2 code
                    # agent_pre_env_action, swap_win_rate_better_rate, is_use_p_value, is_swap_better, p_value = \
                    #     get_chi_square_test_from_action_dict(
                    #         action_lookup_dict, eval_current_state_key)

                    # if agent_pre_env_action == 1:
                    #     print(f"Testing: found a swap. win rate better : {swap_win_rate_better_rate:.3f} | P value {p_value:.5f}")
                    
                else:
                    # don't swap if state is the same
                    agent_pre_env_action = 0

                agent_env_action, _ = turn_agent_action_into_env_action(agent_pre_env_action, game_state[0])

                # store actions taken for debugging purposes
                pkm_env_action_dict = add_action_to_pkm_env_action_dict(agent_env_action, pkm_env_action_dict, 0)
                pkm_env_action_dict = add_action_to_pkm_env_action_dict(opp_action, pkm_env_action_dict, 1)

                eval_last_state_key = copy.deepcopy(eval_current_state_key)
            else:
                # if 10k states and at least 1k of each action, then can continue
                # could nest this but gets harder to read
                if state_key in action_state_results_dict:
                    num_total_visits = 0

                    num_swap_party_0 = action_state_results_dict[state_key].get(
                        agent_first_move_swap_party_0_key, {}).get(action_dict_count_key, 0 )
                    num_total_visits += num_swap_party_0
                    min_visits_to_state = num_swap_party_0
                    
                    num_attacks = action_state_results_dict[state_key].get(
                            agent_first_move_attack_key, {}).get(action_dict_count_key, 0 )
                    num_total_visits += num_attacks
                    min_visits_to_state = min(min_visits_to_state, num_attacks)

                    if agent_team_size == 3:
                        num_swap_party_1 = action_state_results_dict[state_key].get(
                            agent_first_move_swap_party_1_key, {}).get(action_dict_count_key, 0 )
                        num_total_visits += num_swap_party_1
                        min_visits_to_state = min(min_visits_to_state, num_swap_party_1)

                    if num_total_visits >= 10000 and min_visits_to_state >= 1000:
                        # enough states in dict and is first move, just move to next simulation
                        break

            # check that actions are valid
            check_that_actions_are_valid(agent_pre_env_action, agent_env_action, True, agent_team_best_damage_list)
            check_that_actions_are_valid(0, opp_action, False, opp_best_damage_list)

            # enter action and step the env
            action_list = [agent_env_action, opp_action]
            game_state, _not_used_reward, terminated, truncated, info = env.step(action_list)  # for inference, we don't need reward

            if episode_step == max_episode_steps - 1:
                print('Warning: max steps reached')
                terminated = True

            if is_3v3_battle:
                # for 3v3 battles, reset after 1 pkm faints
                # get num active pokemon on each side
                agent_num_active_pkm = get_num_active_pokemon(game_state[0])
                opp_num_active_pkm = get_num_active_pokemon(game_state[1])

                if agent_num_active_pkm < 3 or opp_num_active_pkm < 3:
                    terminated = True
                    if agent_num_active_pkm < 2 or opp_num_active_pkm < 2:
                        # end battle
                        print("Error: 3v3 battle but somehow less than 2 pkm on a side")
            elif is_3v2_or_2v3_battle:
                # if battle collapses to 2v2, then end battle
                agent_num_active_pkm = get_num_active_pokemon(game_state[0])
                opp_num_active_pkm = get_num_active_pokemon(game_state[1])

                if agent_num_active_pkm == 2 and opp_num_active_pkm == 2:
                    terminated = True
                    is_terminated_due_to_2v2 = True
                else:
                    is_terminated_due_to_2v2 = False

            if terminated:
                winner = env.winner
                if winner == agent_index:
                    agent_win_int = 1
                else:
                    agent_win_int = 0

                if is_3v3_battle:
                    action_state_results_dict = add_3v3_results_state_list_to_action_dict(
                        action_state_results_dict, state_list, agent_first_move,
                        agent_first_move_attack_key, action_dict_count_key)
                elif is_3v2_or_2v3_battle:
                    if is_terminated_due_to_2v2:
                        action_state_results_dict = add_3v3_results_state_list_to_action_dict(
                            action_state_results_dict, state_list, agent_first_move,
                            agent_first_move_attack_key, action_dict_count_key)
                    else:
                        action_state_results_dict = add_results_state_list_to_action_dict(
                            action_state_results_dict, state_list, agent_first_move, agent_win_int,
                            agent_first_move_attack_key, action_dict_count_key)
                        
                else:
                    action_state_results_dict = add_results_state_list_to_action_dict(
                        action_state_results_dict, state_list, agent_first_move, agent_win_int,
                        agent_first_move_attack_key, action_dict_count_key)

                if winner in winner_dict:
                    winner_dict[winner] += 1
                # end battle
                break

        if battle_idx % 1000000 == 0 and battle_idx > 0:
            save_object_as_pkl(action_state_results_dict,
                f'3v3_results/{run_tag}_action_state_results_dict_checkpoint_{battle_idx}')

    end_time = time.time()
    print(f"Time to run {(end_time - start_time) / 60:.3f} minutes")
    print(f"Time to run {(end_time - start_time) / num_battles:.3f} seconds per battle")
    print(f"Time to run {((end_time - start_time) / num_battles / 60 / 60) * 1000000:.3f} hours per million battles")

    print(winner_dict)

    if is_save:
        save_object_as_pkl(action_state_results_dict, f'3v3_results/{run_tag}_action_state_results_dict')
        save_object_as_pkl(winner_dict, f'3v3_results/{run_tag}_winner_dict')

        if is_eval:
            save_object_as_pkl(pkm_env_action_dict, f'3v3_results/{run_tag}_pkm_env_action_dict')

    return winner_dict, action_state_results_dict, pkm_env_action_dict


def add_results_state_list_to_action_dict(action_dict, state_list, agent_first_move, agent_win_int,
    agent_first_move_attack_key, action_dict_count_key):
    '''
    '''
    count_key = action_dict_count_key
    sum_wins_key = "sum_wins"

    for state_list_index, state_key in enumerate(state_list):
        # first move can be swap or attack, then attacking rest of the way
        if state_list_index == 0:
            move_key = agent_first_move
        else:
            move_key = agent_first_move_attack_key
        
        if state_key in action_dict:
            if move_key in action_dict[state_key]:
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


def add_3v3_results_state_list_to_action_dict(
    action_dict,
    state_list,
    agent_first_move,
    agent_first_move_attack_key,
    action_dict_count_key):
    '''
    Idea here is not storing a result of wins
    Storing what the final state is when battle is no longer a 3v3
    Want to know the final state and how many times it was reached
    Can later calculate expected win rate from this

    Example Output:
    {
        initial_state: {
            move_key:{
                count_key: count_of_times_reached_this state_with_a_winorloss,
                final_state_value: count_of_times_reached_this state,
                final_state_value_some_other_battle: count_of_times_reached_this state,
                ...
            },
            ...
        }
    }
    '''
    count_key = action_dict_count_key
    sum_wins_key = "sum_wins"

    if len(state_list) > 0:
        final_state_key = state_list[-1]
    else:
        print("Error: state list is empty")

    for state_list_index, state_key in enumerate(state_list):
        # first move can be swap or attack, then attacking rest of the way
        if state_list_index == 0:
            move_key = agent_first_move
        else:
            move_key = agent_first_move_attack_key
        
        if state_key in action_dict:
            if move_key in action_dict[state_key]:
                if final_state_key in action_dict[state_key][move_key]:
                    action_dict[state_key][move_key][final_state_key] += 1
                else:
                    action_dict[state_key][move_key][final_state_key] = 1   
            else:
                action_dict[state_key][move_key] = {
                    final_state_key: 1,
                    # not used here but want to keep the same structure
                    # if are state-move visits with a win-outcome, will increment these in
                    # add_results_state_list_to_action_dict
                    count_key: 0,
                    sum_wins_key: 0
                }

        else:
            action_dict[state_key] = {}
            action_dict[state_key][move_key] = {
                final_state_key: 1,
                # not used here but want to keep the same structure
                # if are state-move visits with a win-outcome, will increment these in
                # add_results_state_list_to_action_dict
                count_key: 0,
                sum_wins_key: 0
            }

    return action_dict


def get_sorted_team_list(agent_team, opp_team, agent_pkm_sort_list, opp_pkm_sort_list,
                         is_reveal_opp_party_0_moves, is_reveal_opp_party_1_moves):
    '''
    Get sorted team list
    '''
    if agent_pkm_sort_list[0] > agent_pkm_sort_list[1]:
        agent_party_list = [agent_team.party[1], agent_team.party[0]]
    else:
        agent_party_list = [agent_team.party[0], agent_team.party[1]]

    if is_reveal_opp_party_0_moves and is_reveal_opp_party_1_moves:
        if opp_pkm_sort_list[0] > opp_pkm_sort_list[1]:
            opp_party_list = [opp_team.party[1], opp_team.party[0]]
        else:
            opp_party_list = [opp_team.party[0], opp_team.party[1]]
    else:
        opp_party_list = [opp_team.party[0], opp_team.party[1]]

    return agent_party_list, opp_party_list


def get_turns_to_faint_list(
        team_1_game_state,
        team_2_game_state,
        max_turns_to_faint_value,
        team_1_party_list,
        team_2_party_list,
        hp_normalizer_reveal_list,
        is_reveal_team_1_active_moves,
        is_reveal_team_1_party_0_moves,
        is_reveal_team_1_party_1_moves,
        is_reveal_team_2_party_0_hp,
        is_reveal_team_2_party_1_hp,):
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
    team_1_pkm_list = [team_1.active] + team_1_party_list

    # Get opponent's team
    team_2 = team_2_game_state.teams[0]
    team_2_pkm_list = [team_2.active] + team_2_party_list

    # Iterate over all my Pokémon and their moves to find the most damaging move
    best_damage_list = []
    turns_to_faint_list = []
    hp_list = []
    normalize_hp_list = []

    for team_1_pkm_index, team_1_pkm in enumerate(team_1_pkm_list):

        if hp_normalizer_reveal_list[team_1_pkm_index]:
            # get normalized HP if allowed to be revealed
            pkm_hp = team_1_pkm.hp
            if pkm_hp <= 240.:
                normalize_hp_list.append(0)
            elif pkm_hp <= 336:
                normalize_hp_list.append(1)
            else:
                normalize_hp_list.append(2)
        else:
            normalize_hp_list.append(hide_default_value)

        for team_2_pkm_index, team_2_pkm in enumerate(team_2_pkm_list):

            # reveal the state under these conditions:
            # team 1 active moves revealed and team 2 active HP is revealed (active HP is always revealed)
            # team 1 active moves revaled and team 2 party 0 HP is revealed
            # team 1 active moves revaled and team 2 party 1 HP is revealed
            # team 1 party 0 moves revealed and team 2 active HP is revealed (active HP is always revealed)
            # team 1 party 0 moves revealed and team 2 party 0 HP is revealed
            # team 1 party 0 moves revealed and team 2 party 1 HP is revealed
            # team 1 party 1 moves revealed and team 2 active HP is revealed (active HP is always revealed)
            # team 1 party 1 moves revealed and team 2 party 0 HP is revealed
            # team 1 party 1 moves revealed and team 2 party 1 HP is revealed

            # see if hp is allowed to be revealed
            if team_2_pkm_index == 0:
                is_reveal_hp_allowed = True
            elif team_2_pkm_index == 1 and is_reveal_team_2_party_0_hp:
                is_reveal_hp_allowed = True
            elif team_2_pkm_index == 2 and is_reveal_team_2_party_1_hp:
                is_reveal_hp_allowed = True
            else:
                is_reveal_hp_allowed = False
        
            # see if moves are allowed to be revealed
            if team_1_pkm_index == 0 and is_reveal_team_1_active_moves:
                is_reveal_move_allowed = True
            elif team_1_pkm_index == 1 and is_reveal_team_1_party_0_moves:
                is_reveal_move_allowed = True
            elif team_1_pkm_index == 2 and is_reveal_team_1_party_1_moves:
                is_reveal_move_allowed = True
            else:
                is_reveal_move_allowed = False

            # will reveal and get ttf if hp for team_2_pkm is allowed to be revealed
            # and if moves for team_1_pkm is allowed to be revealed
            if is_reveal_hp_allowed and is_reveal_move_allowed:
                is_reveal_part_of_state = True
            else:
                is_reveal_part_of_state = False

            if team_1_pkm.fainted() or team_1_pkm.hp <= 0.0 or team_2_pkm.fainted() or team_2_pkm.hp <= 0.0:
                # will always know if fainted or not
                turns_to_faint_list.append(fainted_default_value)
                continue
            elif not is_reveal_part_of_state:
                # state is not revealed, so hide the value
                turns_to_faint_list.append(hide_default_value)
                continue
            else:
                # get TTF for this pkm match up

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

                # used for debugging
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

    return turns_to_faint_list, normalize_hp_list


### HP
# {120.0: 3779517,
#  156.0: 3378332,
#  192.0: 2453402,
#  228.0: 1692512,
#  264.0: 1144966,
#  300.0: 758680,
#  336.0: 484424,
#  372.0: 296296,
#  408.0: 164303,
#  444.0: 75229,
#  480.0: 22651}
### TTF
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
    save_path = f'{save_tag}.pickle'
    with open(save_path, 'wb') as handle:
        print("saving: ", save_path)
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_that_actions_are_valid(pre_env_action, env_action, is_agent, best_damage_list):
    if pre_env_action == 1 and env_action != 4:
        print("Error pre env action is 1 but env action is not 4 ")
        if not is_agent:
            print("Error opp pre_env action is not always 0")
    elif pre_env_action == 2 and env_action != 5:
        print("Error pre env action is 2 but env action is not 5 ")
        if not is_agent:
            print("Error opp pre_env action is not always 0")
    elif pre_env_action == 0:
        if (env_action < 0 or env_action > 3):
            print("Error pre envaction is 0 but env action is not 0 to 3 ")
        
        if len(best_damage_list) == 0:
            print("Error pre_env action is 0 but best damage list is empty")
        elif best_damage_list[0] < 0:
            print("Error agent action is 0 but best damage is negative")
    else:
        print("Error: invalide pre env action")


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


def get_reveal_roll_results(agent_team_size, opp_team_size, is_eval):
    '''
    Parts of the state concerning opp may or may not be revealed in the simmed battle
    due to this roll

    opp can have these things revealed or not:
    opp active moves
    opp party 0 moves
    opp party 1 moves
    opp active HP (always revealed)
    opp party 0 HP
    opp party 1 HP
    '''
    reveal_roll = np.random.rand()

    if is_eval:
        print("TODO: implement this")
        assert 1 == 0
    else:
        is_reveal_opp_active_moves = True
        is_reveal_opp_party_0_moves = True
        is_reveal_opp_party_1_moves = True
        is_reveal_opp_party_0_hp = True
        is_reveal_opp_party_1_hp = True

        if opp_team_size == 2:
            # unknowns should always be at the end

            if reveal_roll < .14:
                if reveal_roll >= .1:
                    is_reveal_opp_active_moves = False
                elif reveal_roll >= .06:
                    is_reveal_opp_party_0_moves = False
                elif reveal_roll >= .04:
                    is_reveal_opp_party_0_hp = False
                    is_reveal_opp_party_0_moves = False
                elif reveal_roll >= .02:
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_0_moves = False
                else:
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_0_hp = False
                    is_reveal_opp_party_0_moves = False
        elif opp_team_size == 3:
            # don't think it is possible for a move to be revealed but not the hp
            # unknowns should always be at the end
            # in prd will sort it that way

            # does not occur as hidden moved to the end
            # if reveal_roll >= :
            #     # party 0 completely hidden
            #     is_reveal_opp_party_0_hp = False
            #     is_reveal_opp_party_0_moves = False
            # elif reveal_roll >= :
            #     # party 0 completely hidden and active moves hidden
            #     is_reveal_opp_active_moves = False
            #     is_reveal_opp_party_0_hp = False
            #     is_reveal_opp_party_0_moves = False
            # elif reveal_roll >= .:
            #     # only party 0 moves hidden
            #     is_reveal_opp_party_0_moves = False
            # elif reveal_roll >= .:
            #         # active and party 0 moves hidden
            #         is_reveal_opp_active_moves = False
            #         is_reveal_opp_party_0_moves = False

            if reveal_roll < .33:

                if reveal_roll >= .29:
                    # party 1 completely hidden
                    is_reveal_opp_party_1_hp = False
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .25:
                    # party 0 and party 1 completely hidden
                    is_reveal_opp_party_0_hp = False
                    is_reveal_opp_party_0_moves = False
                    is_reveal_opp_party_1_hp = False
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .22:
                    # party 1 completely hidden and active moves hidden
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_1_hp = False
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .18:
                    # only actives moves are hidden
                    is_reveal_opp_active_moves = False
                elif reveal_roll >= .14:
                    # only party 1 moves hidden
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .10:
                    # party moves are hidden
                    is_reveal_opp_party_0_moves = False
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .06:
                    # active and party 1 moves hidden
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_1_moves = False
                elif reveal_roll >= .03:
                    # all moves hidden, all hp known
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_0_moves = False
                    is_reveal_opp_party_1_moves = False
                else:
                    # everything that can be hidden is hidden
                    is_reveal_opp_active_moves = False
                    is_reveal_opp_party_0_moves = False
                    is_reveal_opp_party_1_moves = False
                    is_reveal_opp_party_0_hp = False
                    is_reveal_opp_party_1_hp = False

            if agent_team_size == 2:
                # must know opp active moves since one agent pkm has falled (presumably to an attack)
                is_reveal_opp_active_moves = True
        

    return is_reveal_opp_active_moves, is_reveal_opp_party_0_moves, is_reveal_opp_party_1_moves,\
        is_reveal_opp_party_0_hp, is_reveal_opp_party_1_hp


def get_num_active_pokemon(game_state):
    num_active_pkm = 0

    team_zero = game_state.teams[0]
    team_list = [team_zero.active] + team_zero.party

    for i, pkm in enumerate(team_list):
        if not pkm.fainted() or pkm.hp > 0.0:
            num_active_pkm += 1

    return num_active_pkm


def get_pkm_id_sort_list(team_party_list):
    '''
    Reduce state size by sorting the pkm
    '''
    if len(team_party_list) <= 1:
        return [0, 1], ['0', '1']
    
    pkm_id_list = []
    pkm_sort_list = []

    for i, pkm in enumerate(team_party_list):
        pkm_id = ''
        for j, move in enumerate(pkm.moves):
            pkm_id += str(move.type) + str(move.power)

        pkm_id_list.append(pkm_id)

    if pkm_id_list[0] > pkm_id_list[1]:
        pkm_sort_list = [1, 0]
    else:
        pkm_sort_list = [0, 1]

    return pkm_sort_list, pkm_id_list


# Eval Section Functions
def get_chi_square_test_from_action_dict(
    action_dict,
    state_key,
    min_total_count=100,
    min_swap_count=50,
    min_attack_count=50,
    swap_key='swap', attack_key='attack',
    sum_wins_key='sum_wins', count_key='count',
    is_print_statistics=False):

    attack_action = 0
    swap_party_zero_action = 1
    swap_party_one_action = 2

    is_use_p_value = False
    is_swap_better = False
    p_value = None
    swap_win_rate_better_rate = 0.
    recommended_action = attack_action

    try:
        if state_key in action_dict:

            if swap_key in action_dict[state_key] and attack_key in action_dict[state_key]:
                swap_wins = action_dict[state_key][swap_key][sum_wins_key]
                swap_count = action_dict[state_key][swap_key][count_key]
                attack_wins = action_dict[state_key][attack_key][sum_wins_key]
                attack_count = action_dict[state_key][attack_key][count_key]

                total_count = swap_count + attack_count

                if total_count > min_total_count and swap_count > min_swap_count and attack_count > min_attack_count:

                    swap_win_percent = swap_wins / swap_count
                    attack_win_percent = attack_wins / attack_count
                    
                    if swap_win_percent > attack_win_percent:
                        is_swap_better = True
                        swap_win_rate_better_rate = swap_win_percent - attack_win_percent
                    else:
                        is_swap_better = False
                        swap_win_rate_better_rate = 0.

                    # chi squared table breaks down if any 0 values
                    # really should not have less than 5
                    if attack_wins == attack_count:
                        recommended_action = attack_action
                        # choose attack as attack always wins
                        if is_print_statistics:
                            print("Attack always wins")
                            print(f"Swap win rate: {swap_wins / swap_count:.3f} | Count {swap_count}")
                            print(f"Attack win rate: {attack_wins / attack_count:.3f} | Count {attack_count}")
                    elif swap_wins == swap_count:
                        # choose swap
                        is_use_p_value = True
                        is_swap_better = True
                        p_value = 0.
                        recommended_action = swap_party_zero_action
                        if is_print_statistics:
                            print("swap always wins, choosing swap")
                            print(f"Swap win rate: {swap_wins / swap_count:.3f} | Count {swap_count}")
                            print(f"Attack win rate: {attack_wins / attack_count:.3f} | Count {attack_count}")
                    elif swap_wins == 0:
                        recommended_action = attack_action
                        # swap always loses
                        if is_print_statistics:
                            print("Swap always loses")
                            print(f"Swap win rate: {swap_wins / swap_count:.3f} | Count {swap_count}")
                            print(f"Attack win rate: {attack_wins / attack_count:.3f} | Count {attack_count}")
                    elif attack_wins == 0:
                        # attack always loses and swap won at least once so choose swap
                        is_use_p_value = True
                        is_swap_better = True
                        p_value = 0.
                        recommended_action = swap_party_zero_action
                        if is_print_statistics:
                            print("Attack always loses, choosing swap ")
                            print(f"Swap win rate: {swap_wins / swap_count:.3f} | Count {swap_count}")
                            print(f"Attack win rate: {attack_wins / attack_count:.3f} | Count {attack_count}")
                    else:
                        contingency_table = [[swap_wins, swap_count - swap_wins], [attack_wins, attack_count - attack_wins]]
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        is_use_p_value = True

                        if is_swap_better:
                            if p_value < 0.25:
                                recommended_action = swap_party_zero_action
                            elif swap_win_rate_better_rate >= .1:
                                recommended_action = swap_party_zero_action
                            elif swap_win_rate_better_rate >= .05 and p_value < .6:
                                recommended_action = swap_party_zero_action

                        if is_print_statistics:
                            #print(f'Swap Win : { win_loss_draw1[0] / sum(win_loss_draw1):.3f}')
                            print(f"Swap win rate: {swap_wins / swap_count:.3f} | Count {swap_count}")
                            print(f"Attack win rate: {attack_wins / attack_count:.3f} | Count {attack_count}")
                            print(f'Chi-square statistic: {chi2:.3f}')
                            print(f'P-value: {p_value:.5f}')

        else:
            is_use_p_value = False
            is_swap_better = False
            p_value = None
            swap_win_rate_better_rate = 0.
            recommended_action = attack_action
    except Exception as e:
        print("Error: in chi square test ", str(e) )
        is_use_p_value = False
        is_swap_better = False
        p_value = None
        swap_win_rate_better_rate = 0.
        recommended_action = attack_action
    
    return recommended_action, swap_win_rate_better_rate, is_use_p_value, is_swap_better, p_value


def is_swap_allowed_last_current_state(last_state, current_state):
    '''
    If swap is the same for last stae and current state, don't swap as would ahve checked last state
    '''
    if last_state == current_state:
        return False
    else:
        return True


# DEPRECATED. Kind of an explanation to why
# def is_swap_allowed_symmetric_state(current_state, num_agent_pkm, num_opp_pkm):
#     '''
#     Do not allow swaps if the result would mean the same state
#     '''

#     if num_agent_pkm == 2 and num_opp_pkm == 2:
#         # 8 keys
#         # agent active to opp active
#         # agent active to opp party
#         # agent party to opp active
#         # agent party to opp party
#         # opp active to agent active
#         # opp active to agent party
#         # opp party to agent active
#         # opp party to agent party
#         # if a swap occurs
#         #   index 0 becomes index 2 and index 1 becomes index 3
#         #   index 4 becomes index 5 and index 6 becomes index 7

#         if current_state[0] == current_state[2] and current_state[1] == current_state[3] \
#             and current_state[4] == current_state[5] and current_state[6] == current_state[7]:
#             print("Testing: same state, swap not allowed")
#             return False
#         else:
#             return True

#     return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set the run_id for args
    parser.add_argument('--agent_team_size', type=int, choices=[2,3])
    parser.add_argument('--opp_team_size', type=int, choices=[2,3])
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--num_battles', type=int, default=10)
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--is_eval', type=int, default=0, choices=[0, 1])
    parser.add_argument('--eval_dict_path', type=str, default='')
    args = parser.parse_args()

    main(args)
