from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.competition.Competitor import Competitor
from pequil_bot_battle_policy import PequilBot


class PequilBotCompetitor(Competitor):

    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = PequilBot()
        # self._battle_policy = RandomPlayer()
        # self._team_selection_policy = FirstEditionTeamSelectionPolicy()
        # self._team_build_policy = RandomTeamBuilder()

    @property
    def name(self):
        return self._name

    # @property
    # def team_build_policy(self) -> TeamBuildPolicy:
    #     return self._team_build_policy

    # @property
    # def team_selection_policy(self) -> TeamSelectionPolicy:
    #     return self._team_selection_policy

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
