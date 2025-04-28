from abc import ABC, abstractmethod
from enum import Enum

from scipy.optimize import minimize

import numpy as np
import pandas as pd
from typing import Union

class WeightingSchemeType(Enum):
    EQUAL_WEIGHT = "Equal Weight"
    SHARPE = "Sharpe Ratio"
    RANK = "Ranking"

class WeightingScheme(ABC):
    @abstractmethod

    def compute_weights(self, signals: Union[pd.Series, pd.DataFrame]):
        pass

    @staticmethod
    def _clean_signals(data:pd.DataFrame):
        """
        Méthode permettant d'exclure les 10% de valeurs extrêmes de l'univers investissable
        :param data:
        :return:
        """

class MaxSharpeWeighting(WeightingScheme):
    """
    Classe permettant de construire les poids d'un portefeuille
    en maximisant le sharpe ratio
    """
    def compute_weights(self, signals: pd.DataFrame) -> list:
        """
        Méthode permettant de construire le portefeuille qui maximise
        le ratio de Sharpe
        :param signals: rendements à utiliser
        :return: liste des poids après optimisation
        """
        # Récupération du nombre d'actifs
        n_assets: int = signals.shape[1]
 
        # Récupération du rendement moyen de chaque actif
        returns: list = signals.describe().iloc[1,:].values.tolist()
    
        """
        On régularise la matrice de covariance. En effet, étant donné le nombre 
        considérable d'actifs considérés, cette régularisation est nécessaire
        pour que l'optimisation fonctionne. 
        """
        epsilon = 0.01 
        cov: np.ndarray = np.cov(signals.T) + np.eye(n_assets) * epsilon
    
        # On vérifie que les données (en particulier les covariances) sont bien calculées
        if np.any(np.isnan(returns)) or np.any(np.isnan(cov)):
            print("Données avec NaN détectées, retour à équipondération")
            return [1.0/n_assets] * n_assets
    
        # On considère deux méthodes d'optimisation, au cas où la première échoue
        methods = ["SLSQP", "trust-constr"]
    
        for method in methods:
            try:
                # Poids initiaux équipondérés
                x0 = np.ones(n_assets)/n_assets
            
                # Contrainte sur la sommation à 1 des poids
                constraints = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})

                # Bornes pour les poids (long-only)
                bounds = tuple((0,1) for _ in range(n_assets))
            
                # Options d'optimisation 
                options = {
                    'maxiter': 2000, 
                    'disp': True,
                    'ftol': 1e-6,  
                }
            
                # Lancement 
                result = minimize(
                    self._calc_portfolio_sharpe, 
                    x0,
                    args=(returns, cov),
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options=options
                )
            
                # Si l'optmisation réussit, on récupère les nouveaux poids
                if result.success:
                    weights = result.x
                    # On normalise pour s'assurer que la somme est 1
                    weights = weights / np.sum(weights)
                    return weights.tolist()
                else:
                    print(f"Méthode {method} n'a pas convergé: {result.message}")
                
            except Exception as e:
                print(f"Méthode {method} a échoué: {e}")
    
    @staticmethod
    def _calc_portfolio_sharpe(weights:np.ndarray, returns:list, cov:np.ndarray) -> float:
        """
        Méthode permettant de calculer l'opposée du sharpe ratio du portefeuille que l'on minimise
        pour déterminer les poids
        :param returns: rendements à utiliser
        :param cov: matrice de variance covariance des rendements
        :param weights: poids des titres
        :return: sharpe ratio annualisé du portefeuille
        """
        try:
            # calcul du rendement moyen du portefeuille
            ptf_ret:float = np.sum(np.multiply(weights.T, returns))  * 12

            # Calcul de la volatilité
            ptf_vol: float = np.sqrt(weights.T @ (cov @ weights) * 12)

        except():
            raise Exception("Les dimensions ne correspondent pas")

        # calcul du ratio de sharpe (==> voir pour ajouter un facteur d'annualisation dans la méthode...)
        ptf_sharpe: float = ptf_ret/ptf_vol

        return -ptf_sharpe

class EquallyWeighting(WeightingScheme):
    def __init__(self, quantile:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile:float = quantile

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille équipondéré
        Return : liste de poids associés à chaque actif du portefeuille
        """

        # Initialisation de la liste contenant les poids du portefeuille
        weights_ptf: list = []

        # Récupération
        nb_buy_signals: int = signals.loc[signals > 0].shape[0]
        if nb_buy_signals==0:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals(self.quantile)
            weights_ptf: list = ranking_instance.compute_weights(signals)
            return weights_ptf

        weight_long: float = 1 / nb_buy_signals

        for i in range(len(signals)):
            # Premier cas : portefeuille Long Only
                if signals[i] > 0:
                    weights_ptf.append(weight_long)
                else:
                    weights_ptf.append(0)

        return weights_ptf

class RankingWeightingSignals(WeightingScheme):
    def __init__(self, quantile:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile:float = quantile

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille Long-Only en adoptant une méthodologie de ranking
        list_signals : liste contenant les valeurs des signaux renvoyés par la stratégie
        """

        # Création d'une liste pour stocker les poids des titres sur lesquels on prend une position
        weights_ptf: list = [0] * signals.shape[0]

        # Ranking des titres selon les signaux
        ranks: pd.Series = signals.rank(method="max", ascending=True).astype(float)

        # Calcul du nombre de titres à conserver selon le quantile
        nb_stocks: int = int(np.ceil(ranks.shape[0] * self.quantile))

        # Récupération des titres sur lesquels on prend position
        top_ranks:pd.Series = ranks.nlargest(nb_stocks+1)

        # somme des rangs
        top_ranks_sum:pd.Series = top_ranks.sum()
        index_ranks = ranks.isin(top_ranks)

        # boucle pour calculer le poids associé à chaque titre selon le rang
        for i in range(len(ranks)):
            if index_ranks[i]:
                weights_ptf[i] = ranks[i]/top_ranks_sum
        return weights_ptf