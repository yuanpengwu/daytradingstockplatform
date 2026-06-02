"""FinRL-style PPO reinforcement-learning signal.

Architecture
────────────
A PPO agent (Stable-Baselines3 + PyTorch) trained on historical 1-min bar
features.  During inference the policy's action-probability distribution is
read directly, giving a smooth score/confidence pair rather than a hard
argmax decision.

State space  : FEATURE_DIM technical features (same as local ML model — see ml_model.py)
Action space : Discrete(3) — 0=SELL, 1=HOLD, 2=BUY
Reward       : next-bar return × position_direction − transaction_cost

Inference
─────────
  obs → policy network → action probs [p_sell, p_hold, p_buy]
  score      = p_buy − p_sell     ∈ [-1, +1]
  confidence = max(p_buy, p_sell)  ∈ [ 0,  1]

GPU
───
PyTorch CUDA is used automatically when available.  Set config device: cpu
to force CPU inference (useful when the GPU is shared with other processes).

Dependencies
────────────
  pip install stable-baselines3 torch

If stable-baselines3 or torch are not installed, all evaluate() calls return
null signals (score=0, confidence=0) — the bot continues normally with the
remaining signal stack.

Config
──────
finrl:
  model_path: models/finrl_ppo.zip
  retrain_days: 7
  min_confidence: 0.55
  tech_threshold: 0.25        # skip symbols with weak tech signal
  total_timesteps: 100000     # PPO training steps (increase for better AUC)
  device: auto                # auto | cpu | cuda
"""
from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .base import Signal, SignalSource
from .ml_model import FEATURE_DIM, _compute_features  # reuse identical feature engineering

log = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _null_signal(symbol: str, reason: str) -> Signal:
    return Signal(
        symbol=symbol,
        source=SignalSource.FINRL,
        score=0.0,
        confidence=0.0,
        metadata={"reason": reason},
    )


def _detect_device(cfg_device: str) -> str:
    """Return 'cuda' or 'cpu' based on config + hardware availability."""
    if cfg_device == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log.info("FinRL: GPU detected — %s", gpu_name)
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── Gymnasium-compatible trading environment ───────────────────────────────

import gymnasium as _gym


class TradingEnv(_gym.Env):
    """Single-stock episodic environment for PPO training.

    Observation : FEATURE_DIM technical features (float32)
    Action      : Discrete(2) — 0=SELL, 1=BUY
                  (No HOLD option — forcing a directional decision prevents the
                   common RL pathology where the agent collapses to always-HOLD
                   because its reward=0 is "safe" vs noisy BUY/SELL.)
    Reward      : direction × next_bar_return × 100 − tc_cost
                  Normalised by rolling return std so rewards are comparable
                  across symbols with different volatility.
    Episode     : one full sequence of bars (reset restarts from bar 0)

    Inference mapping
    ─────────────────
      probs = [p_sell, p_buy]
      score      = p_buy − p_sell          ∈ [-1, +1]
      confidence = |p_buy − p_sell|        ∈ [ 0,  1]
        → high confidence when policy strongly prefers one direction
        → confidence ≈ 0 when policy is near 50/50 (uncertain)
    """

    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, returns: np.ndarray, tc_pct: float = 0.0005):
        super().__init__()

        self.observation_space = _gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = _gym.spaces.Discrete(2)   # 0=SELL, 1=BUY

        self.features = features.astype(np.float32)

        # Normalise returns so reward scale is consistent across symbols.
        # Target std ≈ 1.0 so reward is dimensionless.
        ret_std = float(np.std(returns))
        self.returns  = (returns / ret_std).astype(np.float32) if ret_std > 1e-9 else returns.astype(np.float32)
        self.tc_pct   = tc_pct / (ret_std if ret_std > 1e-9 else 1.0)   # scale tc too
        self.n        = len(features)
        self._step    = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self.features[0].copy(), {}

    def step(self, action: int):
        ret = float(self.returns[self._step]) if self._step < len(self.returns) else 0.0

        direction = 1.0 if action == 1 else -1.0   # BUY=+1, SELL=-1
        reward    = float(direction * ret - self.tc_pct)

        self._step += 1
        done = self._step >= self.n - 1
        obs  = self.features[min(self._step, self.n - 1)].copy()
        return obs, reward, done, False, {}

    def render(self):
        pass  # headless environment


# ── Main class ─────────────────────────────────────────────────────────────

class FinRLSignal:
    """PPO reinforcement-learning signal.

    Wraps a Stable-Baselines3 PPO agent trained on historical bar features.
    The agent's action distribution is used directly at inference time to
    produce a directional score and confidence, consistent with the rest of
    the signal stack.
    """

    def __init__(self, cfg: dict):
        self.cfg              = cfg
        self.model_path       = Path(cfg.get("model_path", "models/finrl_ppo.zip"))
        self.retrain_days     = int(cfg.get("retrain_days", 1))
        self.min_conf         = float(cfg.get("min_confidence", 0.55))
        self.tech_threshold   = float(cfg.get("tech_threshold", 0.25))
        self.total_timesteps  = int(cfg.get("total_timesteps", 100_000))
        self.daily_timesteps  = int(cfg.get("daily_timesteps", 50_000))
        self._cfg_device      = cfg.get("device", "auto")

        self._model      = None
        self._trained_at: Optional[date] = None
        self._sym_cache: Dict[str, Tuple[datetime, Signal]] = {}

        self._load_model()

        if self._model is not None:
            log.info(
                "FinRLSignal: model loaded from %s (trained %s).",
                self.model_path, self._trained_at,
            )
        else:
            log.info(
                "FinRLSignal: no saved model found — will train at startup. "
                "(requires: pip install stable-baselines3 torch)"
            )

    # ── Model persistence ──────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self.model_path.with_suffix(".meta.pkl")

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(self.model_path))
            meta = self._meta_path()
            if meta.exists():
                with open(meta, "rb") as f:
                    obj = pickle.load(f)
                self._trained_at = obj.get("trained_at")
        except Exception as e:
            log.warning("FinRL: could not load model: %s", e)
            self._model = None

    def _save_model(self) -> None:
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(self.model_path))
            with open(self._meta_path(), "wb") as f:
                pickle.dump({"trained_at": self._trained_at}, f)
            log.info("FinRL PPO model saved → %s", self.model_path)
        except Exception as e:
            log.warning("FinRL: could not save model: %s", e)

    # ── Training ───────────────────────────────────────────────────────────

    def _needs_retrain(self) -> bool:
        if self._model is None or self._trained_at is None:
            return True
        return (date.today() - self._trained_at).days >= self.retrain_days

    def train(self, bars_by_sym: Dict[str, pd.DataFrame]) -> bool:
        """Train (or incrementally update) the PPO agent on historical bars.

        First call (no saved model)
        ───────────────────────────
        A new PPO agent is built and trained for ``total_timesteps`` steps
        pooled across all symbols (~4 min CPU / ~2 min GPU).

        Subsequent daily calls (model already saved)
        ────────────────────────────────────────────
        The existing model is loaded, its environment is updated to the
        fresh daily bars, and training continues for ``daily_timesteps``
        steps (``reset_num_timesteps=False`` preserves the learning-rate
        schedule and total-step counter).  This is ~3-4× faster and lets
        the agent retain knowledge from prior days while adapting to the
        latest market data.

        Returns True if training succeeded.
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            log.warning(
                "FinRL train skipped — stable-baselines3 not installed. "
                "Run:  pip install stable-baselines3 torch"
            )
            return False

        # ── Build pooled feature + next-bar-return sequences ──────────────
        all_feats, all_rets = [], []
        for sym, bars in bars_by_sym.items():
            if bars is None or len(bars) < 50:
                continue
            feats = _compute_features(bars)
            if feats is None or len(feats) < 20:
                continue
            close = bars["Close"].astype(float).values
            rets  = np.diff(close) / np.where(close[:-1] == 0, np.nan, close[:-1])
            rets  = np.nan_to_num(rets, nan=0.0)
            n = min(len(feats) - 1, len(rets))
            if n < 20:
                continue
            all_feats.append(feats[:n])
            all_rets.append(rets[:n])

        if not all_feats:
            log.warning("FinRL train: no usable symbol data — skipping.")
            return False

        X      = np.vstack(all_feats).astype(np.float32)
        R      = np.concatenate(all_rets).astype(np.float32)
        n_syms = len(all_feats)

        device  = _detect_device(self._cfg_device)
        env     = TradingEnv(X, R)
        vec_env = DummyVecEnv([lambda: env])

        incremental = self._model is not None and self._trained_at is not None
        timesteps   = self.daily_timesteps if incremental else self.total_timesteps

        log.info(
            "FinRL %s: %d steps across %d symbols (timesteps=%d) …",
            "incremental update" if incremental else "initial training",
            len(X), n_syms, timesteps,
        )

        if incremental:
            # ── Incremental update: swap in fresh env, continue learning ──
            # set_env() replaces the rollout buffer's environment without
            # resetting the policy weights or optimiser state.
            try:
                self._model.set_env(vec_env)
                self._model.learn(
                    total_timesteps=timesteps,
                    reset_num_timesteps=False,   # preserve step counter & LR schedule
                )
                model = self._model
            except Exception as inc_err:
                # Rare: architecture mismatch if FEATURE_DIM changed.  Fall
                # back to a full retrain so the bot is never left without a model.
                log.warning(
                    "FinRL incremental update failed (%s) — falling back to full retrain.",
                    inc_err,
                )
                incremental = False
                timesteps   = self.total_timesteps

        if not incremental:
            # ── Full retrain from scratch ──────────────────────────────────
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps       = min(512, max(64, len(X) // 8)),
                batch_size    = 64,
                n_epochs      = 10,
                learning_rate = 1e-4,
                gamma         = 0.99,
                gae_lambda    = 0.95,
                clip_range    = 0.20,
                ent_coef      = 0.05,   # higher entropy → prevents always-HOLD collapse
                policy_kwargs = dict(net_arch=[dict(pi=[128, 64], vf=[128, 64])]),
                verbose       = 0,
                device        = device,
            )
            model.learn(total_timesteps=timesteps)

        self._model      = model
        self._trained_at = date.today()
        self._save_model()

        log.info(
            "FinRL training complete | mode=%s  device=%s  samples=%d  syms=%d",
            "incremental" if incremental else "full",
            device, len(X), n_syms,
        )
        return True

    # ── Inference ──────────────────────────────────────────────────────────

    def _get_action_probs(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Return [p_sell, p_buy] from the Discrete(2) policy network.

        Uses the policy's action distribution directly (avoids argmax
        information loss).  Falls back to a hard one-hot vector if the
        distribution API is unavailable.
        """
        try:
            import torch
            from stable_baselines3.common.utils import obs_as_tensor

            obs_t = obs_as_tensor(obs, self._model.device)
            with torch.no_grad():
                # get_distribution returns a SB3 CategoricalDistribution
                dist  = self._model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy()[0]   # shape (2,)
            return probs

        except Exception as e:
            log.debug("FinRL proba inference failed (%s) — falling back to argmax.", e)

        # Argmax fallback: one-hot at the predicted action
        try:
            action, _ = self._model.predict(obs, deterministic=True)
            a  = int(action[0]) if hasattr(action, "__len__") else int(action)
            ph = np.zeros(2, dtype=np.float32)
            ph[a] = 1.0
            return ph
        except Exception as e2:
            log.debug("FinRL argmax fallback failed: %s", e2)
            return None

    def _predict(self, bars: pd.DataFrame) -> Tuple[float, float]:
        """Return (score, confidence) for the most recent bar.

        Discrete(2) formulation:
          probs  = [p_sell, p_buy]
          score  = p_buy − p_sell      ∈ [-1, +1]
          conf   = |p_buy − p_sell|    ∈ [ 0,  1]
            → confidence near 0 when policy is uncertain (≈ 50/50)
            → confidence near 1 when policy strongly prefers one direction
        """
        if self._model is None:
            return 0.0, 0.0

        feats = _compute_features(bars)
        if feats is None or len(feats) == 0:
            return 0.0, 0.0

        obs   = feats[-1:].astype(np.float32)   # shape (1, 8)
        probs = self._get_action_probs(obs)
        if probs is None:
            return 0.0, 0.0

        p_sell, p_buy = float(probs[0]), float(probs[1])

        score      = float(np.clip(p_buy - p_sell, -1.0, 1.0))
        confidence = float(np.clip(abs(p_buy - p_sell), 0.0, 1.0))
        return score, confidence

    # ── Backtest batch pre-computation ────────────────────────────────────

    def precompute_backtest_scores(
        self,
        bars_by_symbol: Dict[str, "pd.DataFrame"],
    ) -> None:
        """Pre-compute FinRL scores for every bar of every symbol in one GPU batch.

        Call once before the backtest loop.  Subsequent ``evaluate()`` calls
        check ``_bt_cache`` first and return instantly, eliminating the O(N²)
        feature recomputation and the per-bar GPU dispatch overhead.

        Results are stored in ``self._bt_cache[sym]`` as a dict mapping each
        bar timestamp → (score, confidence).
        """
        if self._model is None:
            log.warning("FinRL: precompute skipped — model not trained.")
            return

        import torch
        from stable_baselines3.common.utils import obs_as_tensor

        self._bt_cache: Dict[str, Dict] = {}
        total_bars = sum(len(df) for df in bars_by_symbol.values())
        log.info(
            "FinRL: pre-computing scores for %d symbols / %d bars …",
            len(bars_by_symbol), total_bars,
        )

        for sym, df in bars_by_symbol.items():
            feats = _compute_features(df)
            if feats is None or len(feats) == 0:
                self._bt_cache[sym] = {}
                continue

            # Single GPU batch for all bars of this symbol
            obs_batch = torch.tensor(
                feats.astype(np.float32), device=self._model.device
            )
            try:
                with torch.no_grad():
                    dist  = self._model.policy.get_distribution(obs_batch)
                    probs = dist.distribution.probs.cpu().numpy()  # (N, 2)
            except Exception as exc:
                log.debug("FinRL batch inference failed for %s (%s) — zeros.", sym, exc)
                self._bt_cache[sym] = {}
                continue

            cache: Dict = {}
            for i, ts in enumerate(df.index):
                p_sell, p_buy = float(probs[i, 0]), float(probs[i, 1])
                score = float(np.clip(p_buy - p_sell, -1.0, 1.0))
                conf  = float(np.clip(abs(p_buy - p_sell), 0.0, 1.0))
                cache[ts] = (score, conf)
            self._bt_cache[sym] = cache

        log.info("FinRL: pre-computation complete.")

    # ── Public evaluate ────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        tech_signal: Optional[Signal] = None,
    ) -> Optional[Signal]:
        """Evaluate the PPO policy for *symbol* and return a Signal.

        Returns a null signal (score=0, conf=0) rather than None so the
        aggregator's dead-signal detection can track this source correctly.
        """
        if self._model is None:
            return _null_signal(symbol, "model not trained")

        # Tech gate: skip symbols where the technical signal is too weak
        # to bother running the heavier RL policy
        if tech_signal is None or abs(tech_signal.score) < self.tech_threshold:
            return _null_signal(symbol, "tech score below threshold")

        # Backtest pre-computation cache: O(1) lookup, zero GPU dispatch
        bt_sym_cache = getattr(self, "_bt_cache", {}).get(symbol)
        if bt_sym_cache is not None and not bars.empty:
            ts = bars.index[-1]
            entry = bt_sym_cache.get(ts)
            if entry is not None:
                score, conf = entry
                if conf >= self.min_conf:
                    return Signal(
                        symbol=symbol,
                        source=SignalSource.FINRL,
                        score=score,
                        confidence=conf,
                        metadata={"provider": "finrl_ppo"},
                    )
                return _null_signal(symbol, f"low confidence ({conf:.2f})")

        # Live path: per-call inference with 5-min wall-clock cache
        cached = self._sym_cache.get(symbol)
        if cached and (datetime.now() - cached[0]) < timedelta(minutes=5):
            return cached[1]

        score, conf = self._predict(bars)

        if conf >= self.min_conf:
            sig = Signal(
                symbol=symbol,
                source=SignalSource.FINRL,
                score=score,
                confidence=conf,
                metadata={"provider": "finrl_ppo"},
            )
            log.info("FinRL %s | score=%+.2f conf=%.2f", symbol, score, conf)
        else:
            sig = _null_signal(symbol, f"low confidence ({conf:.2f})")

        self._sym_cache[symbol] = (datetime.now(), sig)
        return sig
