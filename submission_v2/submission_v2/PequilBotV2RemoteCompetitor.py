import argparse

from PequilV2_Competitor import PequilBotV2Competitor
from vgc.network.RemoteCompetitorManager import RemoteCompetitorManager


def main(args):
    competitorId = args.id
    competitor = PequilBotV2Competitor(name=f"PequilBotV2 {competitorId}")
    server = RemoteCompetitorManager(competitor, port=5000 + competitorId,
                                     authkey=f'Competitor {competitorId}'.encode('utf-8'))
    server.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    main(args)
