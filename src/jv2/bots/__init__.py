"""JV Boting v2 – Bot Registry"""

from src.jv2.config import SYMBOLS
from src.jv2.bots.trend_rider import TrendRider
from src.jv2.bots.mean_reverter import MeanReverter
from src.jv2.bots.breakout_hunter import BreakoutHunter
from src.jv2.bots.contrarian import Contrarian
from src.jv2.bots.flow_tracker import FlowTracker
from src.jv2.bots.momentum_surfer import MomentumSurfer
from src.jv2.bots.level_bouncer import LevelBouncer
from src.jv2.bots.volatility_fader import VolatilityFader

BOT_CLASSES = [
    TrendRider, MeanReverter, BreakoutHunter, Contrarian,
    FlowTracker, MomentumSurfer, LevelBouncer, VolatilityFader,
]


def create_all_bots():
    """Erstellt 8 Bots × N Symbole = alle Bot-Instanzen."""
    bots = []
    for symbol in SYMBOLS:
        for cls in BOT_CLASSES:
            bots.append(cls(symbol=symbol))
    return bots
