'''
Trained bot for entry in 2024 VGC AI Battle Track Competition

Given a game state, the bot will return an action to take in the game.

working:

DONE smoke test

review the code
    could be reason for weak performance
        did I filter the state correctly?
    how i buildt the 2v2 dict might be borked. i borked the 2v3 dict

see how it performs in a simple eval loop
    DONE against random
    DONE against nibot
    against itself
    DONE against simplebot

DONE confirm it works in the competition format

confirm the 2v2 state look up and swap is working properly
    check some individual states tos ee if they make sense
    only log the results of when the swap is occuring and see win rate
        though could be biased in favor of bad results
    confirm the swap is actually happening
    maybe log the win % of both actions at the time and the result of the action
    print out the state and then what happened next? maybe the states until the end of hte match?
    maybe be more restrictive on the p values?
        for some reason not working as well as expected
    

join dicts
    2v3 dict
    3v2 dict
    3v3 dict

evaluate dicts


Clean up for prod
    this file in submission
    dicts to load in submission

To do:
current and last state key check maybe
'''

import numpy as np
import pickle
import math

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER, MAX_HIT_POINTS, MOVE_MAX_PP, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmMove, Pkm, PkmTeam, GameState, Weather
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, \
    N_TYPES, N_STATUS, N_STATS, N_ENTRY_HAZARD, N_WEATHER, PkmStatus, PkmEntryHazard


class PequilBot(BattlePolicy):
    '''
    '''
    def __init__(self):
        self.two_vs_two_dict = load_pkl_object('two_vs_two_dict.pkl')

        self.recommended_action_key = 'recommended_action'

        self.max_ttf_value = 8
        
        # self.current_state_key = tuple([19,19])
        # self.last_state_key = tuple([19])

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, game_state: GameState):
        '''
        TEST get num of pkm on each side
            if one on either side
        TEST get best dmg action

        get ttf state
            making state the same way might be a bit tricky
            sorting
                double check this. I am seeing some states in 2v2 dict that should not exist
                    ie unknown moves for active pkm
            reads unkowns properly
                unknown hp
                unknown moves
                    assume if one move known then use TTF
            chekc notes again
        swap 0 and 1 for the 3v2 and 3v3 is complicated
        '''
        #print("in get action -1")
        try:
            num_agent_pkm_non_fainted, agent_fainted_list = self._get_num_non_fainted_pokemon(game_state.teams[0])
            num_opponent_pkm_non_fainted, opp_fainted_list = self._get_num_non_fainted_pokemon(game_state.teams[1])
            #print("in get action -0.5")
            best_active_action, _ = self._get_best_active_damage_action(game_state)
            #print("in get action -0.4")
            try:
                if num_agent_pkm_non_fainted == 1 or num_opponent_pkm_non_fainted == 1:
                    action = best_active_action
                elif num_agent_pkm_non_fainted >= 2 and num_opponent_pkm_non_fainted >= 2:
                    action = best_active_action
                    #print("in get action -0.3")
                    # get opp pkm hp and move revealed status
                    is_reveal_opp_active_hp, is_reveal_opp_active_moves, \
                    is_reveal_opp_party_0_hp, is_reveal_opp_party_0_moves, \
                    is_reveal_opp_party_1_hp, is_reveal_opp_party_1_moves, = self._get_opp_pkm_reveal_status(game_state)
                    #print("in get action -0.25")
                    # get the state
                    agent_pkm_party_sort_list, _ = self._get_pkm_id_sort_list(game_state.teams[0].party)
                    #print("in get action -0.2")
                    if is_reveal_opp_party_0_hp and is_reveal_opp_party_0_moves and is_reveal_opp_party_1_hp and is_reveal_opp_party_1_moves:
                        opp_pkm_party_sort_list, _ = self._get_pkm_id_sort_list(game_state.teams[1].party)
                        is_sort_opp_party = True
                    else:
                        opp_pkm_party_sort_list = [0, 1]
                        is_sort_opp_party = False
                    #print("in get action -0.1")
                    agent_sorted_party_list, opp_sorted_party_list = self._get_sorted_team_list(
                        game_state.teams[0], game_state.teams[1],
                        agent_pkm_party_sort_list, opp_pkm_party_sort_list,
                        is_sort_opp_party)
                    #print("debug pre state 0")
                    # get state info
                    state_list_agent, agent_normalize_hp_list = self._get_state_key_list(
                        game_state,
                        self.max_ttf_value,
                        agent_sorted_party_list,
                        opp_sorted_party_list,
                        hp_normalizer_reveal_list=[True, True, True],
                        is_reveal_team_1_active_moves=True,
                        is_reveal_team_1_party_0_moves=True,
                        is_reveal_team_1_party_1_moves=True,
                        is_reveal_team_2_party_0_hp=is_reveal_opp_party_0_hp,
                        is_reveal_team_2_party_1_hp=is_reveal_opp_party_1_hp,
                        is_agent_team_1=True)

                    state_list_opp, opp_normalize_hp_list = self._get_state_key_list(
                        game_state,
                        self.max_ttf_value,
                        opp_sorted_party_list,
                        agent_sorted_party_list,
                        hp_normalizer_reveal_list=[is_reveal_opp_active_hp, is_reveal_opp_party_0_hp, is_reveal_opp_party_1_hp],
                        is_reveal_team_1_active_moves=is_reveal_opp_active_moves,
                        is_reveal_team_1_party_0_moves=is_reveal_opp_party_0_moves,
                        is_reveal_team_1_party_1_moves=is_reveal_opp_party_1_moves,
                        is_reveal_team_2_party_0_hp=True,
                        is_reveal_team_2_party_1_hp=True,
                        is_agent_team_1=False)
                    #print("debug after state 1")
                    if len(agent_fainted_list) == 3 and len(opp_fainted_list) == 3 \
                        and len(state_list_agent) == 9 and len(state_list_opp) == 9 \
                        and len(agent_normalize_hp_list) == 3 and len(opp_normalize_hp_list) == 3:

                        if num_agent_pkm_non_fainted == 2 and num_opponent_pkm_non_fainted == 2:
                            agent_state_values_to_keep = [True] * len(state_list_agent)
                            opp_state_values_to_keep = [True] * len(state_list_agent)
                            #print("in 2v2 2")
                            if agent_fainted_list[1]:
                                filtered_agent_normalize_hp_list = [agent_normalize_hp_list[0], agent_normalize_hp_list[2]]

                                agent_state_values_to_keep[3:6] = [False]*3

                                opp_state_values_to_keep[1] = False
                                opp_state_values_to_keep[4] = False
                                opp_state_values_to_keep[7] = False
                            elif agent_fainted_list[2]:
                                filtered_agent_normalize_hp_list = [agent_normalize_hp_list[0], agent_normalize_hp_list[1]]

                                agent_state_values_to_keep[6:] = [False]*3

                                opp_state_values_to_keep[2] = False
                                opp_state_values_to_keep[5] = False
                                opp_state_values_to_keep[8] = False
                            else:
                                print("Error: agent active is fainted somehow but 2 pkm in party")
                            #print("in 2v2 3")
                            if opp_fainted_list[1]:
                                filtered_opp_normalize_hp_list = [opp_normalize_hp_list[0], opp_normalize_hp_list[2]]

                                opp_state_values_to_keep[3:6] = [False]*3

                                agent_state_values_to_keep[1] = False
                                agent_state_values_to_keep[4] = False
                                agent_state_values_to_keep[7] = False
                            elif opp_fainted_list[2]:
                                filtered_opp_normalize_hp_list = [opp_normalize_hp_list[0], opp_normalize_hp_list[1]]

                                opp_state_values_to_keep[6:] = [False]*3

                                agent_state_values_to_keep[2] = False
                                agent_state_values_to_keep[5] = False
                                agent_state_values_to_keep[8] = False
                            else:
                                print("Error: opp active is fainted somehow but 2 pkm in party")
                            #print("in 2v2 4")
                            filtered_state_list_agent = [state_list_agent[i] for i in range(len(state_list_agent)) if agent_state_values_to_keep[i]]
                            filtered_state_list_opp = [state_list_opp[i] for i in range(len(state_list_opp)) if opp_state_values_to_keep[i]]

                            state_key = tuple(filtered_state_list_agent + filtered_state_list_opp 
                                            + filtered_agent_normalize_hp_list + filtered_opp_normalize_hp_list)
                            
                            if len(state_key) != 12:
                                print("Error: state key length is not as expected")

                            if state_key in self.two_vs_two_dict:
                                #print("state key in the dict")
                                recommended_action = self.two_vs_two_dict[state_key].get(self.recommended_action_key, 0)
                                #print("in 2v2 5")
                                if recommended_action != 0:
                                    action = self._turn_agent_action_into_env_action(game_state, 
                                        recommended_action, best_active_action, is_allow_fuzzy_swap=True)
                                    #print(f"Testing, recommending a swap {action}, recommended action {recommended_action}")
                                    #print(state_key, self.two_vs_two_dict[state_key])
                    else:
                        print("Error: state length or fainted length is not as expected")

                else:
                    action = best_active_action
            except Exception as e:
                print("Error: getting best active action ", str(e))
                action = best_active_action
            
            if best_active_action < 0 or best_active_action > 3:
                print(f"Warning: best_active_action is not in the range [0, 3] {best_active_action}")
                action = 0

            if action < 0 or action > 5:
                print(f"Warning: action is not in the range [0, 5] {action}")
                action = 0

        except Exception as e:
            print("Error: choosing default action ", str(e))
            action = 0
        
        return action


    def _get_state_key_list(
        self,
        game_state,
        max_turns_to_faint_value,
        team_1_party_list,
        team_2_party_list,
        hp_normalizer_reveal_list,
        is_reveal_team_1_active_moves,
        is_reveal_team_1_party_0_moves,
        is_reveal_team_1_party_1_moves,
        is_reveal_team_2_party_0_hp,
        is_reveal_team_2_party_1_hp,
        is_agent_team_1):
        '''
        put in zero for the fainted
            handle the check for fainted in HP

        '''
        hide_default_value = -1
        fainted_default_value = 0

        weather = game_state.weather.condition

        team_1 = game_state.teams[0]
        team_1_pkm_list = [team_1.active] + team_1_party_list

        team_2 = game_state.teams[1]
        team_2_pkm_list = [team_2.active] + team_2_party_list

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

                # see if hp is revealed
                if not is_agent_team_1:
                    # team 2 is the agent. Agent always knows own HP
                    is_reveal_hp_allowed = True
                elif team_2_pkm.revealed:
                    # team 2 is the opp. only know opp HP if revealed
                    is_reveal_hp_allowed = True
                else:
                    is_reveal_hp_allowed = False
                

                if is_agent_team_1:
                    # team 1 is the agent. Agent always knows own moves
                    is_reveal_move_allowed = True
                else:
                    _, is_current_team_1_move_revealed = self._get_pkm_reveal_status(team_1_pkm)
                    if is_current_team_1_move_revealed:
                        # team 1 is the opp, only know a move if opp revealed a move
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
                        
                        if is_agent_team_1 or move.revealed:
                            damage = self._estimate_damage(move.type, team_1_pkm.type, move.power, team_2_pkm.type, team_1_attack_stage,
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


    def _get_num_non_fainted_pokemon(self, game_state_team):
        num_non_fainted_pkm = 0
        fainted_list = []

        team_list = [game_state_team.active] + game_state_team.party

        for i, pkm in enumerate(team_list):
            if not pkm.fainted() or pkm.hp > 0.0:
                num_non_fainted_pkm += 1
                fainted_list.append(False)
            else:
                fainted_list.append(True)

        return num_non_fainted_pkm, fainted_list

    
    def _get_best_active_damage_action(self, g: GameState):
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
                
                damage = self._estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
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


    def _estimate_damage(self, move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
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
    

    def _get_pkm_id_sort_list(self, team_party_list):
        '''
        Reduce state size by sorting the pkm
        '''
        if len(team_party_list) <= 1:
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

        return pkm_sort_list, pkm_id_list
    

    def _get_opp_pkm_reveal_status(self, game_state):
        '''
        '''
        is_reveal_opp_active_hp, is_reveal_opp_active_moves = self._get_pkm_reveal_status(game_state.teams[1].active)
        is_reveal_opp_party_0_hp, is_reveal_opp_party_0_moves = self._get_pkm_reveal_status(game_state.teams[1].party[0])
        is_reveal_opp_party_1_hp, is_reveal_opp_party_1_moves = self._get_pkm_reveal_status(game_state.teams[1].party[1])

        return is_reveal_opp_active_hp, is_reveal_opp_active_moves, \
            is_reveal_opp_party_0_hp, is_reveal_opp_party_0_moves, \
            is_reveal_opp_party_1_hp, is_reveal_opp_party_1_moves
    

    def _get_pkm_reveal_status(self, pkm):
        '''
        Treating one move revealed like all moves are revealed
        '''
        is_reveal_hp = pkm.revealed
        is_reveal_moves = False

        for move in pkm.moves:
            if move.revealed:
                is_reveal_moves = True
                break

        return is_reveal_hp, is_reveal_moves
        

    def _get_sorted_team_list(self, agent_team, opp_team, agent_pkm_sort_list, opp_pkm_sort_list,
                         is_sort_opp_party):
        '''
        Get sorted team list if allowed
        '''
    
        if len(agent_team.party) == 1:
            agent_party_list = [agent_team.party[0]]
        else:
            if agent_pkm_sort_list[0] > agent_pkm_sort_list[1]:
                agent_party_list = [agent_team.party[1], agent_team.party[0]]
            else:
                agent_party_list = [agent_team.party[0], agent_team.party[1]]

        if len(opp_team.party) == 1:
            opp_party_list = [opp_team.party[0]]
        else:
            if is_sort_opp_party:
                if opp_pkm_sort_list[0] > opp_pkm_sort_list[1]:
                    opp_party_list = [opp_team.party[1], opp_team.party[0]]
                else:
                    opp_party_list = [opp_team.party[0], opp_team.party[1]]
            else:
                opp_party_list = [opp_team.party[0], opp_team.party[1]]

        return agent_party_list, opp_party_list


    def _turn_agent_action_into_env_action(self, 
            agent_game_state, recommended_action, best_active_action, is_allow_fuzzy_swap):
        '''
        Action values are
        0: select best move
        1: switch to first pkm
        2: switch to second pkm

        Env actions are
        0 to 3: action of active pkm
        4: switch to first pkm
        5: switch to second pkm
        '''

        if recommended_action == 0:
            # get best active action
            action = best_active_action
        else:
            # switch to first or second pkm if alive
            if is_allow_fuzzy_swap:
                # allow a swap as long as one of the party pkm is alive
                action = best_active_action

                if recommended_action == 1:
                    pkm_party_0 = agent_game_state.teams[0].party[0]
                    pkm_party_1 = agent_game_state.teams[0].party[1]

                    if pkm_party_0.hp > 0 and not pkm_party_0.fainted():
                        action = 4
                    elif pkm_party_1.hp > 0 and not pkm_party_1.fainted():
                        action = 5
                elif recommended_action == 2:
                    pkm_party_0 = agent_game_state.teams[0].party[0]
                    pkm_party_1 = agent_game_state.teams[0].party[1]

                    if pkm_party_1.hp > 0 and not pkm_party_1.fainted():
                        action = 5
                    elif pkm_party_0.hp > 0 and not pkm_party_0.fainted():
                        action = 4

                if action == best_active_action:
                    print("Warning: recommended action is a swap but no pkm to swap to fuzzy")
            else:
                # only allow a swap if the specific pkm in that slot is alive
                if recommended_action == 1 or recommended_action == 2:
                    pkm = agent_game_state.teams[0].party[recommended_action-1]
                    if pkm.fainted() or pkm.hp <= 0.0:
                        action = best_active_action
                        print("Warning: recommended action is a swap but no pkm to swap to fainted")
                    else:
                        action = recommended_action + 3
                else:
                    action = best_active_action
                    print("Warning: recommended action is a swap but no pkm to swap to")

        return action


def load_pkl_object(pkl_path):
    '''
    Load a pickle object
    '''
    with open(pkl_path, 'rb') as handle:
        return pickle.load(handle)