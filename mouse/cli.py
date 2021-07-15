from pathlib import Path

import click

from mouse.agent.agent import Agent
from mouse.environment.steinmetz import Steinmetz
from mouse.model import Stimulus
from mouse.service.agent_service import AgentService


@click.group()
def cli():
    pass

@cli.command()
@click.option("--path", required=True)
@click.option("--direction", required=True)
def infer(path: str, direction: str):
    print(direction)
    agent = AgentService.load(Path(path))
    env = Steinmetz()
    env.reset(stimulus=Stimulus.left())
    agent.infer()
    print(agent)
    print(agent._policy_net)

@cli.command()
@click.option("--episodes", "-e", required=True, type=int)
@click.option("--save", required=True)
def train(episodes: int, save: str):
    agent = Agent()
    agent.train(num_episodes=episodes)

    agent.save(Path(save))
