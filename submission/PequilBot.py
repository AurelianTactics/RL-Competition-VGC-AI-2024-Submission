import numpy as np
import pickle

from vgc.datatypes.Objects import PkmTeam, Pkm, GameState, Weather
from vgc.behaviour import BattlePolicy
# from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER

# from vgc.datatypes.Types import PkmStat, PkmType


class PequilBot(BattlePolicy):
    '''
    '''
    def __init__(self):
        self.two_vs_two_dict = load_pkl_object('two_vs_two_dict.pkl')

        self.recommended_action_key = 'recommended_action'

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, game_state: GameState):
        '''
        TEST get num of pkm on each side
            if one on either side
        get best dmg action
            STOPPED HERE
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
        num_agent_pkm_non_fainted = self._get_num_non_fainted_pokemon(game_state.teams[0])
        num_opponent_pkm_non_fainted = self._get_num_non_fainted_pokemon(game_state.teams[1])
        
        best_active_action = self._get_best_active_action(game_state)

        if num_agent_pkm_non_fainted == 1 or num_opponent_pkm_non_fainted == 1:
            action = best_active_action
        elif num_agent_pkm_non_fainted >= 2 and num_opponent_pkm_non_fainted >= 2:
            action = best_active_action
            # get the state
            ttf_state = ...

            if num_agent_pkm_non_fainted == 2 and num_opponent_pkm_non_fainted == 2:
                if ttf_state in self.two_vs_two_dict:
                    recommended_action = self.two_vs_two_dict[ttf_state].get(self.recommended_action_key, 0)
                    if recommended_action != 0:
                        action = self._turn_agent_action_into_env_action(game_state, 
                            recommended_action, best_active_action, is_allow_fuzzy_swap=True)

        else:
            action = best_active_action

        if best_active_action < 0 or best_active_action > 3:
            print(f"Warning: best_dmg_action is not in the range [0, 3] {best_active_action}")

        if action < 0 or action > 5:
            print(f"Warning: action is not in the range [0, 5] {action}")

        return action


    def _get_num_non_fainted_pokemon(self, game_state_team):
        num_non_fainted_pkm = 0

        team_list = [game_state_team.active] + game_state_team.party

        for i, pkm in enumerate(team_list):
            if not pkm.fainted() or pkm.hp > 0.0:
                num_non_fainted_pkm += 1

        return num_non_fainted_pkm
    

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

            else:
                # only allow a swap if the specific pkm in that slot is alive
                if recommended_action == 1 or recommended_action == 2:
                    pkm = agent_game_state.teams[0].party[recommended_action-1]
                    if pkm.fainted() or pkm.hp <= 0.0:
                        action = best_active_action
                    else:
                        action = recommended_action + 3
                else:
                    action = best_active_action

        return action


def load_pkl_object(pkl_path):
    '''
    Load a pickle object
    '''
    with open(pkl_path, 'rb') as handle:
        return pickle.load(handle)