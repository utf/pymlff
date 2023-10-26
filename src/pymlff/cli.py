"""Command line interface for pymlff."""
import sys

import click


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """Command line utility for the pymlff package."""


@cli.command()
@click.argument("inputs", type=click.Path(exists=True), nargs=-1)
@click.argument("output", type=click.Path(exists=False))
def merge(inputs, output):
    """Merge multiple ML_AB files into one."""
    from pymlff import MLAB

    if len(inputs) < 2:
        click.echo("ERROR: At least 2 ML_AB files are needed as input")
        sys.exit()

    new_ml_ab = None
    for filename in inputs:
        try:
            ml_ab = MLAB.from_file(filename)
            if new_ml_ab is None:
                new_ml_ab = ml_ab
            else:
                new_ml_ab += ml_ab
        except ValueError:
            click.echo(f"ERROR: Could not read ML_AB file: {filename}")
            sys.exit()

    new_ml_ab.write_file(output)


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def write_extxyz(input, output):
    """Convert an ML_AB file to extended xyz format."""
    from pymlff import MLAB

    try:
        ml_ab = MLAB.from_file(input)
    except ValueError:
        click.echo(f"ERROR: Could not read ML_AB file: {input}")
        sys.exit()

    ml_ab.write_extxyz(output)
