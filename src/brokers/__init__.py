"""Broker abstraction + concrete adapters."""
from .base import BrokerBase, Order, OrderSide, OrderType, Position, OrderStatus
from .paper_broker import PaperBroker


def get_broker(name: str, cfg: dict) -> BrokerBase:
    """Factory — instantiate the broker chosen in config.yaml."""
    name = name.lower()
    if name == "paper":
        return PaperBroker(
            starting_cash=cfg.get("starting_cash", 10_000),
            slippage_bps=cfg.get("slippage_bps", 5),
        )
    if name == "alpaca":
        from .alpaca_broker import AlpacaBroker
        return AlpacaBroker()
    if name == "robinhood":
        from .robinhood_broker import RobinhoodBroker
        return RobinhoodBroker()
    raise ValueError(f"Unknown broker: {name}")
