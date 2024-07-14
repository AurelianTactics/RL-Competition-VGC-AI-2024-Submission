'''
Trained bot for entry in 2024 VGC AI Battle Track Competition

Given a game state, the bot will return an action to take in the game.

'''

import numpy as np
import pickle
import math

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER, MAX_HIT_POINTS, MOVE_MAX_PP, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmMove, Pkm, PkmTeam, GameState, Weather
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, \
    N_TYPES, N_STATUS, N_STATS, N_ENTRY_HAZARD, N_WEATHER, PkmStatus, PkmEntryHazard


class PequilBotV2(BattlePolicy):
    '''
    '''
    def __init__(self, is_debug=False):
        '''
        '''
        self.is_debug = is_debug
        self.action_dict = self._load_pkl_object('action_dict.pickle')
        self.recommended_action_key = 'recommended_action'

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, game_state: GameState):
        '''
        '''

        try:

            best_active_action, _ = self._get_best_active_damage_action(game_state)

            try:
                num_agent_pkm_non_fainted, agent_fainted_list = self._get_num_non_fainted_pokemon(game_state.teams[0])
                num_opponent_pkm_non_fainted, opp_fainted_list = self._get_num_non_fainted_pokemon(game_state.teams[1])

                if num_agent_pkm_non_fainted <= 1 or num_opponent_pkm_non_fainted <= 1:
                    action = best_active_action
                elif num_agent_pkm_non_fainted >= 2 and num_opponent_pkm_non_fainted >= 2:
    
                    action = best_active_action

                    agent_pkm_party_sort_list, _ = self._get_pkm_id_sort_list(game_state.teams[0].party)

                    current_state_key = self._get_state_key_from_game_state(game_state, agent_pkm_party_sort_list,
                                                                            num_opponent_pkm_non_fainted)

                    if current_state_key in self.action_dict:
                        recommended_action = self.action_dict[current_state_key].get(self.recommended_action_key, 0)
                    
                        if recommended_action != 0:
                            if num_agent_pkm_non_fainted == 2:
                                is_allow_fuzzy_swap = True
                            else:
                                is_allow_fuzzy_swap = False

                            action = self._turn_agent_action_into_env_action(game_state, 
                                recommended_action, best_active_action, is_allow_fuzzy_swap=is_allow_fuzzy_swap)

                            if self.is_debug:
                                print("Recommended action taken ", current_state_key)
                                print(self.action_dict[current_state_key])

                    else:
                        action = best_active_action

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


    def _get_state_key_from_game_state(self, agent_game_state, agent_pkm_party_sort_list, num_opp_pkm):
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
                        
                        damage = self._estimate_damage(move.type, agent_pkm.type, move.power, opp_pkm.type, agent_attack_stage,
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
                            damage = self._estimate_damage(move.type, opp_pkm.type, move.power, agent_pkm.type, opp_attack_stage,
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

        if len(fainted_list) != 3:
            print("Error: fainted list length is not as expected, setting to all fainted")
            fainted_list = [True, True, True]
            num_non_fainted_pkm = 0
        else:
            if fainted_list[0]:
                print("Error: active pkm is fainted, setting to all fainted")
                fainted_list = [True, True, True]
                num_non_fainted_pkm = 0

        if sum(fainted_list) != 3 - num_non_fainted_pkm:
            print("Error: fainted list sum is not as expected, setting to all fainted")
            fainted_list = [True, True, True]
            num_non_fainted_pkm = 0

        return num_non_fainted_pkm, fainted_list

    
    def _get_best_active_damage_action(self, g: GameState):
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

        return damage
    

    def _get_pkm_id_sort_list(self, team_party_list):
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


    def _load_pkl_object(self, pkl_path):
        '''
        Load a pickle object
        '''
        with open(pkl_path, 'rb') as handle:
            return pickle.load(handle)
