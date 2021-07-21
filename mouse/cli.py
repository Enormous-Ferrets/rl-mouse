from pathlib import Path

import click

from mouse.agent.agent import Agent
from mouse.environment.steinmetz import Steinmetz
from mouse.model.experiment import Stimulus
from mouse.service.agent_service import AgentService


@click.group()
def cli():
    pass

@cli.command()
@click.option("--path", required=True)
@click.option("--direction", required=True)
def infer(path: str, direction: str):
    agent = AgentService.load(Path(path))
    env = Steinmetz()
    env.reset(stimulus=Stimulus.from_direction(direction))
    _input = agent.get_screen(env)
    agent.infer(_input)

@cli.command()
@click.option("--episodes", "-e", required=True, type=int)
@click.option("--save", required=True)
def train(episodes: int, save: str):
    agent = Agent()
    agent.train(num_episodes=episodes)

    agent.save(Path(save))

@cli.command()
@click.option("--version", required=True)
@click.option("--sessions", "-s", required=True, type=int)
@click.option("--model", required=False, default="v0.1")
def generate(version: str, sessions: int, model: str):
    print(f"Generating dataset {version}")
    agent = AgentService.load(Path(f"models/{model}"))

    dataset = agent.generate_data(sessions)
    output = Path(".") / "datasets" / f"dataset-{version}.npy"
    print(f"\n\tSaving dataset to {output}\n")
    dataset.save(output)
