'''
Wraps PkmBattleEnv into something expected by most RL algorithms

working
    sheeprl and cleanrl
        can be discrete or box based on an arg
        has id
        removes the needed teams
        arg for which type of obs to use
            simple
            medium
            full
'''

import gymnasium as gym
import numpy as np
from typing import List, Union

from vgc.datatypes.Objects import PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator


from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition

from vgc.datatypes.Constants import MAX_HIT_POINTS, MOVE_MAX_PP, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmMove, Pkm, PkmTeam, GameState, Weather
from vgc.datatypes.Types import N_TYPES, N_STATUS, N_STATS, N_ENTRY_HAZARD, N_WEATHER, PkmStat, PkmType, \
    PkmStatus, WeatherCondition, PkmEntryHazard
from vgc.engine.HiddenInformation import null_pkm, null_pkm_move



class PkmBattleEnvWrapper(gym.Wrapper):
    '''
    '''
    def __init__(self, id: str, obs_type: str = 'simple', **kwargs):
        '''
        '''
        self.num_resets = -1
        self.current_episode_steps = 0
        self.max_episode_steps = 250

        self.player_index, self.opponent_index = self._get_player_opp_index()

        self.opponent_agent = NiBot()
        self.obs_nibot = ObsFromNiBot()

        # competition rules
        # need to test and disable the set team when set team shows it works
        self.team_generator = RandomTeamGenerator(2)
        self.env = self._get_random_team_env(self.team_generator, self.player_index)

        # set team for testing
        # if doing non random mode
        # team0, team1 = PkmTeam(), PkmTeam()
        # self.env = PkmBattleEnv((team0, team1),
        #            # encode Fasle for forward env
        #            #encode=(agent0.requires_encode(), agent1.requires_encode()))  # set new environment with teams
        #            encode=(True, True))
        # self.opponent_agent = RandomPlayer()

        
        # if this is needed then do a reset call here. shouldn't be needed
        #self.current_obs_list = [np.zeros((self.env.observation_space.n,))]
        self.current_obs_list = []

        self.reward_range = (-1, 1)
        self.action_space = self.env.action_space
        

        self.obs_type = obs_type
        if self.obs_type == 'simple':
            self.num_obs =
        elif self.obs_type == 'medium':
            self.num_obs = 
        elif self.obs_type == 'full':
            self.num_obs =

        self.observation_space = gym.spaces.Dict(
            {
                #"state": self.env.observation_space,
                # sheeprl wants a box space
                "state": gym.spaces.Box(low=-1.0, high=2.0, shape=(self.num_obs,), dtype=np.float32)
            }
        )

    def step(self, action):
        '''
        Get opponent action then step through env
        '''

        # if opponent needs the non-bool obs can do this
        # may be an unnecessary step
        #opponent_obs = self._change_obs_bool_to_float(self.current_obs_list[self.opponent_index])
        opponent_obs = self.current_obs_list[self.opponent_index]
        opponent_action = self.opponent_agent.get_action(opponent_obs)
        if opponent_action >= self.action_space.n or opponent_action < 0:
            print(f"Error: opponent action {opponent_action} | action space {self.action_space.n}")
            opponent_action = 0

        if self.player_index == 0:
            action_list = [action, opponent_action]
        else:
            action_list = [opponent_action, action]

        self.current_obs_list, reward_list, terminated, truncated, info = self.env.step(action_list)

        obs = self.current_obs_list[self.player_index]
        obs = self._change_obs_bool_to_float(obs)
        obs_dict = self._put_obs_in_dict(obs)

        # increment number of steps
        self.current_episode_steps += 1
        if self.current_episode_steps >= self.max_episode_steps:
            terminated = True

        # get custom reward
        reward = self._win_loss_reward(terminated, self.player_index)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ''''''
        self.num_resets += 1
        self.current_episode_steps = 0
        self.player_index, self.opponent_index = self._get_player_opp_index()

        # if doing random teams
        self.env = self._get_random_team_env(self.team_generator, self.player_index)

        self.current_obs_list, info = self.env.reset()

        obs = self.current_obs_list[self.player_index]
        obs = self._change_obs_bool_to_float(obs)
        obs_dict = self._put_obs_in_dict(obs)

        return obs_dict, info

    def render(self, mode='console'):
        ''''''
        self.env.render(mode)

    def close(self):
        self.env.close()

    def _get_player_opp_index(self):
        '''
        Get the player and opponent index
        return 0 or 1 depending on which team you are
        Used for accessing obs indices, reward indices,
        telling who winner is etc
        '''
        player_index = self.num_resets % 2
        opp_index = (player_index + 1) % 2
        # for testing do 0 and 1
        # player_index = 0
        # opp_index = 1

        return player_index, opp_index
        
    def _win_loss_reward(self, terminated, player_index):
        '''
        Does a reward for winning or losing
        winner is -1 unless a winner has been picked
        '''
        reward = 0.
        if terminated:
            winner = self.env.winner

            if winner == 0 or winner == 1:
                if winner == player_index:
                    reward = 1.
                else:
                    reward = -1.
            #print(f"reward {reward} | terminated {terminated} | winner {self.env.winner} | player_index {player_index}|")
        return reward

    def _change_obs_bool_to_float(self, obs_list):
        '''
        Some values in the obs are bools
        '''
        obs_list = [float(x) if isinstance(x, bool) else x for x in obs_list]
        # if min(obs_list) < -.01 or max(obs_list) > 2.01:
        #     print("value outside expected obs range found")
        #     # for testing purposes
        #     obs_array = np.array(obs_list, dtype=np.float32)
        #     print('min, max', min(obs_array), max(obs_array))
        #     # print all values in obs less than 0 or greater than 1
        #     for i, x in enumerate(obs_array):
        #         if x < 0 or x > 1:
        #             print(i, x)
        obs_array = np.array(obs_list, dtype=np.float32).clip(-1.0, 2.0)

        return obs_array

    def _put_obs_in_dict(self, obs):
        '''
        Put the obs in a dict
        '''
        obs_dict = {
            "state": obs
        }

        return obs_dict

    def _get_random_team_env(self, team_generator, player_index):
        '''
        '''
        team0 = team_generator.get_team().get_battle_team([0, 1, 2])
        team1 = team_generator.get_team().get_battle_team([0, 1, 2])

        if player_index == 0:
            encode_state_tuple = (True, False)
        else:
            encode_state_tuple = (False, True)

        env = PkmBattleEnv((team0, team1), encode=encode_state_tuple)

        return env

    
    def _get_obs_from_game_state(self, game_state: GameState, obs_type: str, num_obs: int,
                                 is_debug_mode: bool = False) -> np.ndarray:
        '''
        Get obs from game state
        put into list then turn into numpy array

        to do:
        new nibot stuff
        maybe scale the % of HP and dmg by a larger number (the same) and then clip so in 0 to 1 range
        maybe try to make sure max range is 1?
        '''
        obs_list = []
        min_range = -1.0
        max_range = 2.0
        default_value_not_revealed = -0.5
        default_value_fainted = -1.0

        agent_team = game_state.teams[0]
        opp_team = game_state.teams[1]

        obs_list = self._get_surviving_revealed_obs(agent_team, obs_list)
        obs_list = self._get_surviving_revealed_obs(opp_team, obs_list)

        # add nibot values
        obs_nibot_list = self.obs_nibot.get_action(game_state, is_non_active_obs=False, party_index=None)
        obs_list.extend(obs_nibot_list)

        # add weather
        obs_list += self.one_hot(game_state.weather.condition, N_WEATHER)
        obs_list += [game_state.weather.n_turns_no_clear / 5]

        # team obs that are public information
        obs_list = self._encode_team_public(agent_team, obs_list)
        obs_list = self._encode_team_public(opp_team, obs_list)

        # type chart values
        obs_list = self._get_type_chart_values(agent_team, opp_team, obs_list,
            default_value_not_revealed)
        
        # to do
        # get the remaining non public team and move obs
            
        # nibot dmg for agent team against other revealed pkm
        for party_index in range(len(opp_team.party)):
            if opp_team.party[party_index].revealed:
                obs_nibot_non_active_list = self.obs_nibot.get_action(game_state, is_non_active_obs=True,
                    party_index=party_index)
            else:
                # not revealed, so populate the default values
                obs_nibot_non_active_list = [default_value_not_revealed] * 12

            obs_list.extend(obs_nibot_non_active_list)


        obs_list = [float(x) if isinstance(x, bool) else x for x in obs_list]

        if is_debug_mode:
            # for testing purposes
            if min(obs_list) < min_range or max(obs_list) > max_range:
                print("value outside expected obs range found")
                obs_array = np.array(obs_list, dtype=np.float32)
                print('min, max', min(obs_array), max(obs_array))
                # print all values in obs less than 0 or greater than 1
                for i, x in enumerate(obs_array):
                    if x < 0 or x > 1:
                        print(i, x)

        obs_array = np.array(obs_list, dtype=np.float32).clip(min_range, max_range)

        return obs_array
    
    def _get_type_chart_values(self, agent_team: PkmTeam, opp_team: PkmTeam, obs_list: list,
                               default_value: float) -> list:
        '''
        '''
        agent_team_list = [agent_team.active] + agent_team.party
        opp_team_list = [opp_team.active] + opp_team.party

        for agent_pkm in agent_team_list:
            for opp_pkm in opp_team_list:
                if opp_pkm.revealed:
                    # get type chart value
                    type_chart_value = TYPE_CHART_MULTIPLIER[agent_pkm.type][opp_pkm.type] / 2.0
                    obs_list.append(type_chart_value) 
                else:
                    obs_list.append(default_value)

        return obs_list

    def _encode_team_public(self, team: PkmTeam, obs_list: list):
        '''
        Encode the public information of a team
        '''
        obs_list += [team.confused]
        obs_list += team.entry_hazard
        for stat in range(N_STATS):
            obs_list += [team.stage[stat] / 5]

        return obs_list

    def _get_surviving_revealed_obs(self, team: PkmTeam, obs_list: list):
        '''
        Get the surviving pokemon, hp percent left and if they are revealed
        '''
        team_list = [team.active] + team.party

        for pkm in team_list:
            if pkm.fainted():
                # surviving
                obs_list.append(0.)
                # hp
                obs_list.append(0.)
                
            else:
                # surviving
                obs_list.append(1.)
                # hp remaining
                obs_list.append(pkm.hp / MAX_HIT_POINTS)

            # revealed
            if pkm.revealed:
                obs_list.append(1.)
            else:
                obs_list.append(0.)

        return obs_list
    
    def one_hot(self, p, n):
        b = [0] * n
        b[p] = 1
    
        return b
    
    def encode_team(self, obs_list, team, obs_type,):
        '''
        '''
        # STOPPED HERE
        for pkm in team.party:
            encode_pkm(obs_list, pkm)
        encode_pkm(obs_list, team.active)

    



# Nizar's Bot
class NiBot(BattlePolicy):
    '''
    Bot from 2023 VGC AI competition with some modifications
    https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/blob/master/competition/vgc2023/NiBot_Submission%20-%20Nizar%20Haimoud/BattlePolicies.py
    '''

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
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

        # Initialize variables for the best move and its damage
        best_move_id = -1
        best_damage = -np.inf

        # Iterate over all my Pokémon and their moves to find the most damaging move
        for i, pkm in enumerate(my_pkms):
            if i == 0:
                my_attack_stage = my_team.stage[PkmStat.ATTACK]
            else:
                my_attack_stage = 0
            #print("pkm moves is ", pkm.moves)
            for j, move in enumerate(pkm.moves):
                
                if pkm.hp == 0.0:
                    continue

                # Estimate the damage of the move
                damage = self.estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                         opp_defense_stage, weather)

                # Check if the current move has higher damage than the previous best move
                if damage > best_damage:
                    best_move_id = j + i * 4 # think for 2024 j is 0 to 3 for each
                    best_damage = damage
                #print(i, "Move", j, best_move_id, "Damage", best_damage, 'Poke', pkm)
            # print("Pokemon", i, pkm, "Move", best_move_id, "Damage", best_damage)
        # Decide between using the best move, switching to the first party Pokémon, or switching to the second party Pokémon
        if best_move_id < 4:
            return best_move_id  # Use the current active Pokémon's best damaging move
        elif 4 <= best_move_id < 8:
            return 4  # Switch to the first party Pokémon
        else:
            return 5  # Switch to the second party Pokémon

    def estimate_damage(self, move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        '''
        Not from original code. from updated repo
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


class ObsFromNiBot(BattlePolicy):
    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState, is_non_active_obs: bool, party_index: int):

        obs_nibot_list = []
        # Get weather condition
        weather = g.weather.condition

        # Get my Pokémon team
        my_team = g.teams[0]
        my_pkms = [my_team.active] + my_team.party

        # Get opponent's team
        opp_team = g.teams[1]
        if is_non_active_obs:
            # do revealed check outside of this function
            opp_active = opp_team.party[party_index]
        else:
            opp_active = opp_team.active
            
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # Initialize variables for the best move and its damage
        best_move_id = -1
        best_damage = -np.inf

        # Iterate over all my Pokémon and their moves to find the most damaging move
        for i, pkm in enumerate(my_pkms):
            if i == 0:
                my_attack_stage = my_team.stage[PkmStat.ATTACK]
            else:
                my_attack_stage = 0
            #print("pkm moves is ", pkm.moves)
            for j, move in enumerate(pkm.moves):
                
                if pkm.hp == 0.0:
                    scaled_damage = 0.
                else:
                    # Estimate the damage of the move
                    damage = self.estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                            opp_defense_stage, weather)

                    scaled_damage = damage / MAX_HIT_POINTS
                
                # can clip this to desired range outside of this function
                obs_nibot_list.append(scaled_damage)

                # Check if the current move has higher damage than the previous best move
                if damage > best_damage:
                    best_move_id = j + i * 4 # think for 2024 j is 0 to 3 for each
                    best_damage = damage
                
                #print(i, "Move", j, best_move_id, "Damage", best_damage, 'Poke', pkm)
            # print("Pokemon", i, pkm, "Move", best_move_id, "Damage", best_damage)
        # Decide between using the best move, switching to the first party Pokémon, or switching to the second party Pokémon
        if best_move_id < 4:
            nibot_action = best_move_id
        elif 4 <= best_move_id < 8:
            nibot_action = 4
        else:
            nibot_action = 5

        # get the NiBot recommended action
        if is_non_active_obs:
            expected_num_obs = 12
        else:
            expected_num_obs = 18
            nibot_action_one_hot_list = self.one_hot(nibot_action, 6) 
            obs_nibot_list.extend(nibot_action_one_hot_list)

        if len(obs_nibot_list) != expected_num_obs:
            print(f"ObsNiBot obs_list not {expected_num_obs} long. It is {len(obs_nibot_list)} long. Is non active obs: {is_non_active_obs}")

        return obs_nibot_list

    def estimate_damage(self, move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        '''
        Not from original code. from updated repo
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
    
    def one_hot(self, p, n):
        b = [0] * n
        b[p] = 1
    
        return b