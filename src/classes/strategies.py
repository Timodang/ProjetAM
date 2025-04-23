import abc
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy.dialects.postgresql import array

from src.classes.weighting_scheme import EquallyWeighting, RankingWeightingSignals, MaxSharpeWeighting

class Strategy(metaclass=abc.ABCMeta):
    EQUALWEIGHT_LABEL = "equalweight"
    RANKING_LABEL = "ranking"

    @abc.abstractmethod
    def __init__(self, returns: pd.DataFrame, universe: pd.DataFrame, weight_scheme: str, quantile:float) -> None:
        """
        :param returns: Rendement à utiliser pour calculer les signaux d'achat
        :param universe: Composition de l'univers d'investissement sur la période considérée
        :param weight_scheme: schéma de pondération à utiliser
        :param quantile: quantile à utiliser dans le cas d'un ranking
        """
        self.returns: pd.DataFrame = returns
        self.universe: pd.DataFrame = universe
        self.weight_scheme: str = weight_scheme
        self.quantile: float = quantile

    @abc.abstractmethod
    def get_position(self) -> list:
        """
        Méthode pour récupérer les positions du portefeuilles à la suite de la mise en oeuvre de la stratégie
        :return:
        """
        pass

    @staticmethod
    def _clean_returns(returns: pd.DataFrame, universe:pd.DataFrame) -> (pd.DataFrame, array[bool]):
        """
        Méthode permettant de ne conserver qu'une partie des actifs à la date de mise en oeuvre de la stratégie, à savoir :
        - Les actifs qui figurent toujours dans l'univers d'investissement
        - Les actifs pour lesquelles toutes les données nécessaires au calcul des signaux sont disponibles
        :return: Un dataframe contenant les rendements qui sont utilisés pour la mise en oeuvre de la stratégie et
        un vecteur permettant de savoir si un actif est utilisé ou non
        """

        # Dans le cas où il y a des valeurs infinies, on les remplace à 0
        returns.replace(np.inf, 0, inplace=True)

        # Matching des indices de l'univers avec ceux des rendements
        universe = universe[universe.index.isin(returns.index)]

        # Les titres qui ne figurent pas dans l'indice à la date de rebalancement sont exclus
        index_data_not_in_index: array[bool] = universe.iloc[-1, :].eq(0)

        # Les titres pour lesquels ils manquent des rendements dans la fenêtre glissante sont également exclus de l'univers
        index_missing_datas: array[bool] = returns.isna().any()

        # Récupération de l'indice des données qui ne figurent pas dans l'univers
        index_data_not_in_strat: array[bool] = np.logical_or(index_data_not_in_index, index_missing_datas)

        # Seules les colonnes dans données manquantes sont conservées
        returns:pd.DataFrame = returns.loc[:, index_data_not_in_strat==False]

        return returns, index_data_not_in_strat

class Momentum(Strategy):
    """
    Classe qui implémente une stratégie Momentum basée sur les rendements des actifs.
    """

    def __init__(self, returns: pd.DataFrame, universe: pd.DataFrame, weight_scheme: str, quantile:float) -> None:
        self.returns: pd.DataFrame = returns
        self.universe: pd.DataFrame = universe
        self.weight_scheme: str = weight_scheme
        self.quantile: float = quantile

    def get_position(self) -> list:
        """
        Calcule les parts dans le portefeuille à une date donnée.
        """

        # Création d'un array vide pour stocker les poids
        weights_array = np.zeros(self.universe.shape[1])

        # Retraitement des rendements
        returns, index_missing_data = self._clean_returns(self.returns, self.universe)

        # Dans le cas d'un momentum qui n'est pas mean-revert, on ne conserve pas la dernière valeur (données mensuelles)
        if returns.shape[0] > 1:
            returns = returns.iloc[:-1,:]

        # Calcul du rendement sur la période
        signals_momentum: pd.Series = (1 + returns).prod() - 1

        # Dans le cas d'un momentum mean-revert, le signal est inversé
        if returns.shape[0] == 1:
            signals_momentum = -signals_momentum

        # Cas où l'utilisateur souhaite réaliser une allocation par ranking
        if self.weight_scheme == Strategy.RANKING_LABEL:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals(self.quantile)
            weights_momentum: list = ranking_instance.compute_weights(signals_momentum)
        # Cas où il souhaite réaliser une allocation équipondérée
        elif self.weight_scheme == Strategy.EQUALWEIGHT_LABEL:
            equalweight_instance: EquallyWeighting = EquallyWeighting(self.quantile)
            weights_momentum: list = equalweight_instance.compute_weights(signals_momentum)
        else:
            raise Exception("Méthode non implémentée")

        # Récupération des poids sur l'ensemble des actifs et transformation en liste
        weights_array[index_missing_data == False] = weights_momentum
        list_weights:list = weights_array.tolist()

        # check pour vérifier que la somme des poids est approximativement égale à 1
        check_weight = np.sum(list_weights)
        if round(check_weight, 5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
        return list_weights

class IdiosyncraticMomentum(Strategy):
    """
    Classe qui implémente une stratégie de Momentum idiosyncratique
    """
    def __init__(self, returns: pd.DataFrame, universe: pd.DataFrame, bench_ret:pd.DataFrame,
                 weight_scheme: str, quantile:float, window:int) -> None:
        self.returns: pd.DataFrame = returns
        self.bench_ret:pd.DataFrame = bench_ret
        self.universe: pd.DataFrame = universe
        self.weight_scheme: str = weight_scheme
        self.quantile: float = quantile
        self.window: int = window

    @staticmethod
    def _import_ff_factors():
        """
        Méthode permettant d'importer et de retraiter le jeu de données
        contenant les facteurs de Fama & French pour calculer le momentum idiosyncratique
        :return:
        """
        # Importation des facteurs Mkt, HML, SMB
        df_factors: pd.DataFrame = pd.read_excel("data/facteurs_ff.xlsx")

        # Transformation de l'indice en date
        df_factors["Dates"] = pd.to_datetime(df_factors['Dates'])
        df_factors.set_index("Dates", inplace=True)

        return df_factors


    def get_position(self) -> list:
        """
        Méthode permettant de déterminer les poids associés à chaque titre
        dans le cadre d'une stratégie de Momentum Idiosyncratique
        :return:
        """

        # Création d'un array vide pour stocker les poids de chaque actif dans le portefeuille
        weights_array = np.zeros(self.universe.shape[1])

        # Retraitement des rendements
        returns, index_missing_data = self._clean_returns(self.returns, self.universe)

        # Première étape : Importation du benchmark (éventuellement des série de facteurs FF d'AQR)
        df_factors: pd.DataFrame = self.bench_ret
        # df_factors: pd.DataFrame = self._import_ff_factors()

        # Deuxième étape : Alignement des rendements par rapport au portefeuille
        df_factors = df_factors[df_factors.index.isin(returns.index)]

        # Troisième étape : Régression pour calculer le beta de marché
        x = sm.add_constant(df_factors)
        model = sm.OLS(returns, x)
        result = model.fit()
        res = result.resid

        # Quatrième étape : récupération des signaux (= les résidus)
        if self.window == 1:
            # Sur du mean revert : on n'a qu'un résidu et on multiplie par -1 pour être long sur les loser
            signals_id_momentum = -1 * res.iloc[-1]
        else:
            # Calcul du signal mean-revert
            signals_id_momentum =  np.sum(res.iloc[-12:-1])/np.std(res.iloc[-12:-1])

        # Cinquième étape : calcul des poids
        # Cas où l'utilisateur souhaite réaliser une allocation par ranking
        if self.weight_scheme == Strategy.RANKING_LABEL:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals(self.quantile)
            weights: list = ranking_instance.compute_weights(signals_id_momentum)
        # Cas où l'utilisateur souhaite réaliser une allocation équipondérée
        elif self.weight_scheme == Strategy.EQUALWEIGHT_LABEL:
            equalweight_instance: EquallyWeighting = EquallyWeighting(self.quantile)
            weights: list = equalweight_instance.compute_weights(signals_id_momentum)
        else:
            raise Exception("Méthode non implémentée")

        # récupération des poids pour les titres qui figuraient dans l'univers d'investissement
        weights_array[index_missing_data == False] = weights
        list_weights:list = weights_array.tolist()

        # Vérification que la somme des poids est bien égale à 1
        check_weight = np.sum(list_weights)
        if round(check_weight, 5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
        return list_weights

class MaxSharpe(Strategy):
    """
    Classe qui implémente une stratégie Max Sharpe ratio basée sur les rendements des actifs.
    """

    def __init__(self, returns: pd.DataFrame, universe: pd.DataFrame, weight_scheme: str, quantile:float) -> None:
        self.returns: pd.DataFrame = returns
        self.universe: pd.DataFrame = universe
        self.weight_scheme: str = weight_scheme
        self.quantile: float = quantile

    def get_position(self) -> list:
        """
        Calcule les parts dans le portefeuille à une date donnée.
        """
        # Création d'un array vide pour stocker les poids
        weights_array = np.zeros(self.universe.shape[1])

        # Retraitement des rendements
        returns, index_missing_data = self._clean_returns(self.returns, self.universe)

        # Réalisation d'une allocation par max sharpe
        max_sharpe_instance: MaxSharpeWeighting = MaxSharpeWeighting()
        weights = max_sharpe_instance.compute_weights(returns)

        # récupération des poids pour les titres qui figuraient dans l'univers d'investissement
        weights_array[index_missing_data == False] = weights
        list_weights: list = weights_array.tolist()

        # Vérification que la somme des poids est bien égale à 1
        check_weight = np.sum(list_weights)
        if round(check_weight, 5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
        return weights
