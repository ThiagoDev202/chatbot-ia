"""Configuracao de logging da aplicacao."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configura o logger raiz com formato padrao."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
