"""
Market Regime Detection and Adaptation Module
Core system for identifying financial market regimes and adapting strategies.
Architecture: Ensemble detector (HMM + Volatility Clustering + Trend Analysis) → Firebase state → Strategy Adapter
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials, firestore

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Defined market regimes based on volatility and trend characteristics"""
    STRONG_BULL = "strong_bull"      # High returns, low volatility, strong uptrend
    MODERATE_BULL = "moderate_bull"  # Moderate returns, medium volatility
    SIDEWAYS = "sideways"           # Low returns, low volatility, no trend
    VOLATILE = "volatile"           # High volatility, no clear direction
    MODERATE_BEAR = "moderate_bear" # Moderate negative returns
    STRONG_BEAR = "strong_bear"     # Strong negative returns, high volatility
    CRASH = "crash"                 # Extreme negative returns, extreme volatility
    UNKNOWN = "unknown"             # Insufficient data or error state

@dataclass
class RegimeMetrics:
    """Comprehensive metrics for regime characterization"""
    volatility: float
    trend_strength: float
    returns_mean: float
    returns_skew: float
    volume_trend: float
    regime_confidence: float
    duration_days: int
    detected_at: datetime

class FirebaseManager:
    """Handles all Firebase operations with connection pooling and error recovery"""
    
    _initialized = False
    
    @classmethod
    def initialize(cls, credential_path: Optional[str] = None) -> None:
        """Initialize Firebase with proper error handling and singleton pattern"""
        if cls._initialized:
            logger.debug("Firebase already initialized")
            return
            
        try:
            if credential_path and os.path.exists(credential_path):
                cred = credentials.Certificate(credential_path)
                firebase_admin.initialize_app(cred)
            else:
                # For environments with GOOGLE_APPLICATION_CREDENTIALS set
                firebase_admin.initialize_app()
            
            cls._initialized = True
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise RuntimeError(f"Firebase initialization failed: {str(e)}")
    
    @classmethod
    def get_firestore_client(cls) -> firestore.Client:
        """Get Firestore client with validation"""
        if not cls._initialized:
            cls.initialize()
        
        try:
            return firestore.client()
        except Exception as e:
            logger.error(f"Failed to get Firestore client: {str(e)}")
            raise

class FeatureEngineer:
    """Engineers features for regime detection from raw price data"""
    
    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 200]
        self.scaler = StandardScaler()
        logger.info(f"FeatureEngineer initialized with lookbacks: {self.lookback_periods}")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns with validation"""
        if prices is None or len(prices) < 2:
            logger.warning("Insufficient data for returns calculation")
            return pd.Series(dtype=float)
        
        try:
            returns = np.log(prices / prices.shift(1)).dropna()
            logger.debug(f"Calculated returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
            return returns
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def calculate_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility with