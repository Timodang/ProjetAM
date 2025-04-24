import pandas as pd
import numpy as np
import datetime as datetime
from operator import add

from src.classes.strategies import  Momentum, MaxSharpe, IdiosyncraticMomentum
from src.classes.utilitaire import Utils


class Portfolio:
    #Labels relatifs aux méthodes de calcul des rendements
    METHODS_LABELS = ["discret", "continu"]

    # Labels relatifs à la périodicité
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"
    PERIODICITY_LABELS = [MONTHLY_LABEL, QUARTERLY_LABEL, YEARLY_LABEL]

    # Labels relatifs aux stratégies
    MOMENTUM_LABEL = "momentum"
    MOMENTUM_IDIOSYNCRATIC_LABEL = "momentum idiosyncratique"
    MAX_SHARPE_LABEL = "max sharpe"

    # Labels relatifs aux schémas de pondération
    WEIGHTING_LABELS = ["equalweight", "ranking"]

    # Labels relatifs à la segmentation
    SECTOR_LABEL = "sectorial"

    """
    Classe utilisée pour construire un portefeuille et effectuer un backtest sur 
    une stratégie donnée
    """

    def __init__(self, df_prices: pd.DataFrame, universe:pd.DataFrame, bench:pd.DataFrame, dict_sector: dict,
                 periodicity: str, rebalancement: str, method: str,
                 strat: str, weighting: str, calculation_window: int, quantile: float, segmentation:str = None):
        """
        :param df_prices: DataFrame contenant les prix de tous les actifs sur la période d'étude
        :param universe: DataFrame contenant la composition de l'univers d'investissement à chaque date
        :param bench: DataFrame contenant la valeur du benchmark à chaque date
        :param dict_sector: Dictionnaire contenant la segmentation sectorielle du portefeuille
        :param periodicity: Fréquence des données pour le calcul des rendements
        :param rebalancement: Fréquence de rebalancement de la stratégie
        :param method: Méthode pour le calcul des rendements
        :param strat: Nom de la stratégie à backtester
        :param weighting: Méthode de pondération utilisée
        :param calculation_window: taille de la fenêtre
        :param quantile: quantile utilisé pour le calcul des poids dans un schéma de ranking
        :param segmentation: segmentation sectorielle du portefeuille
        """

        # Checks de cohérence
        if periodicity.lower() not in Portfolio.PERIODICITY_LABELS:
            raise Exception(
                f"La périodicité pour le calcul des rendements doit être l'une des suivantes : {Portfolio.PERIODICITY_LABELS}.")
        if method.lower() not in Portfolio.METHODS_LABELS:
            raise Exception(
                f"La méthode pour le calcul des rendements doit être l'une des suivantes : {Portfolio.METHODS_LABELS}.")
        if rebalancement.lower() not in Portfolio.PERIODICITY_LABELS:
            raise Exception(
                f"La périodicité pour la date de rebalancement doit être l'une des suivantes : {Portfolio.PERIODICITY_LABELS}.")

        # instance de la classe util
        self.utils: Utils = Utils()

        # récupération des prix
        self.asset_prices: pd.DataFrame = df_prices

        # Récupération de la composition de l'univers d'investissement et du benchmark
        self.universe: pd.DataFrame = universe
        self.bench: pd.DataFrame = bench

        # Récupération de la segmentation sectorielle du portefeuille
        self.dict_sectors: dict = dict_sector
        self.segmentation: str = segmentation

        # caractéristiques de la stratégie à utiliser
        self.rebalancement: str = rebalancement
        self.periodicity: str = periodicity.lower()
        self.method: str = method.lower()
        self.calculation_window: int = calculation_window
        self.quantile: float = quantile
        self.strategy: str = strat
        self.weighting: str = weighting

        # Calcul des rendements
        self.returns:pd.DataFrame = self.utils.compute_asset_returns(self.asset_prices, self.periodicity, self.method)

        # Composition du portefeuille à chaque date
        self.positions: pd.DataFrame = pd.DataFrame()

        # NAV du portefeuille à chaque date
        self.portfolio_value: pd.Series = pd.Series()

    def run_backtest(self):
        """
        Méthode permettant de réaliser le backtest en calculant les positions de chaque titre
        et le valeur du portefeuille
        :return:
        """

        # 1ere étape : calcul des poids
        portfolio_positions: pd.DataFrame = self.build_portfolio()
        self.positions = portfolio_positions

        # 2eme étape : Calcul de la NAV du portefeuille
        portfolio_value: list = self.compute_portfolio_value()
        self.portfolio_value = pd.DataFrame(portfolio_value, columns = ["Nav"], index = self.positions.index)

    def rebalancing_date(self, serie_date:pd.Series, index:int) -> datetime:
        """
        Méthode permettant de déterminer la prochaine date de rebalancement à partir d'une liste de données mensuelles
        contenant toutes les dates de la période d'étude
        :param serie_date: ensemble de dates de la période d'étude
        :param index: indice de la date actuelle
        :return: la prochaine date de rebalancement
        """

        if self.rebalancement == Portfolio.MONTHLY_LABEL:
            new_date: datetime = serie_date.values[index + 1]
        elif self.rebalancement == Portfolio.QUARTERLY_LABEL:
            new_date: datetime = serie_date.values[index + 3]
        elif self.rebalancement == Portfolio.YEARLY_LABEL:
            new_date: datetime = serie_date.values[index + 12]
        else:
            raise Exception("Fréquence de rebalancement non implémentée")
        return new_date

    def _rebal_to_index(self) -> int:
        """
        Méthode permettant de convertir une fréquence de rebalancement en nombre à partir de données mensuelles
        (ex : rebalancement annuelle pour données mensuelles => 12)
        :return:
        """
        if self.rebalancement == Portfolio.MONTHLY_LABEL:
            index_rebal:int  = 1
        elif self.rebalancement == Portfolio.QUARTERLY_LABEL:
            index_rebal:int = 3
        elif self.rebalancement == Portfolio.YEARLY_LABEL:
            index_rebal:int = 12
        else:
            raise Exception("La fréquence de rebalancement souhaitée n'est pas implémentée")
        return index_rebal

    def build_portfolio(self) -> pd.DataFrame:
        """
        Méthode  permettant de déterminer les positions du portefeuilles à chaque date
        """

        # récupération de l'indice des rebalancements
        int_rebal_freq: int = self._rebal_to_index()

        # On récupère le nombre de rendements disponibles
        length: int = self.returns.shape[0]

        # premier indice à partir duquel on réalise les calculs (on commence arbitrairement au bout de 3 ans dans un soucis de comparaison p/r
        # à la stratégie de Momentum idiosyncratique)
        index_strat: int = 36
        # Récupération de la première date de rebalancement
        rebalancing_date: datetime = self.returns.index[index_strat]

        # Initialisation du dataframe pour stocker les positions / poids
        positions: pd.DataFrame = pd.DataFrame(0.0, index=self.returns.index,
                                               columns=self.returns.columns)

        # Boucle pour construire le portefeuille à partir de la première date de rebalancement
        for idx in range(index_strat, length):

            # Récupération de la date courante
            date: datetime = self.returns.index[idx]

            # Calcul des poids du portefeuille
            # 1er cas : date >= date de rebalancement ==> on rebalance
            if date >= rebalancing_date:

                # Récupération du premier indice pour le calcul des rendements
                begin_idx: int = idx - 36

                # Calcule de la nouvelle date de rebalancement (si disponible)
                if idx < length - int_rebal_freq:
                    rebalancing_date = self.rebalancing_date(self.returns.index, idx)

                # Récupération des signaux associés à cette date selon les stratégies / schémas implémentés
                positions.iloc[idx, :] = self._compute_portfolio_position(
                    self.returns.iloc[begin_idx:idx, :], self.strategy,
                    self.weighting)

            # 2e cas : la date n'est pas une date de rebalancement
            # Récupération des poids précédents et calcul des nouveaux poids (dérive du portefeuille)
            else:
                prec_weights: list = positions.iloc[idx - 1, :]
                positions.iloc[idx, :] = self.compute_portfolio_derive(idx, prec_weights)

        # On ne conserve pas les dates antérieures à la première date de rebalancement
        positions = positions.iloc[index_strat:, ]
        return positions

    def _compute_portfolio_position(self, returns_to_use: pd.DataFrame, strategy:str, weighting:str) -> list:
        """
        Méthode permettant de déterminer le poids du portefeuille à chaque période de rebalancement
        en appliquant une éventuelle segmentation sectorielle
        :param returns_to_use: DataFrame contenant les rendements pertiennts pour le calcul des poids
        :param strategy: nom de la stratégie à utiliser
        :param weighting: schéma de pondération à utiliser
        :return: Une liste contenant les poids du portefeuille pour la date courante
        """

        # Gestion des erreurs pour déterminer si l'utilisateur souhaite réaliser une segmentation sectorielle ou non
        if self.segmentation != Portfolio.SECTOR_LABEL and self.segmentation is not None:
            raise Exception("Segmentation non implémentée")
        elif self.segmentation is None:
            # Calcul des poids sans segmentation sectorielle
            return self._get_weight(returns_to_use, strategy, weighting)
        else:
            # Récupération du nombre de secteurs
            nb_sect: int = len(self.dict_sectors)

            # Calcul du coefficient multiplicateur qui sera utilisé pour construire le portefeuille (équipondération : chaque secteur a le même poids final)
            scaling_factor: float = 1 / nb_sect

            # Liste pour stocker les poids du portefueille
            list_weights_ptf: list = [0] * returns_to_use.shape[1]

            # boucle sur les secteurs
            for key, value in self.dict_sectors.items():
                # Récupération des titres à utiliser pour le secteur étudier
                df_prices_sector: pd.DataFrame = value

                # Calcul des rendements et alignement pour ne conserver que les dates de la période d'étude
                ret_sectors : pd.DataFrame = self.utils.compute_asset_returns(df_prices_sector, self.periodicity, self.method)
                ret_sectors = ret_sectors[ret_sectors.index.isin(returns_to_use.index)]

                # Calcul des poids pour la stratégie
                list_weights_sector: list = self._get_weight(ret_sectors, strategy, weighting)

                # Ajout à la liste des poids finaux du portefeuille pour la date considérée
                list_weight_sector = [weight * scaling_factor for weight in list_weights_sector]
                list_weights_ptf = list(map(add, list_weights_ptf, list_weight_sector))

            return list_weights_ptf

    def _get_weight(self, returns_to_use: pd.DataFrame, strategy: str, weighting: str) -> list:
        """
        Méthode permettant de générer les poids d'un portefeuille à une date t, pour une stratégie donnée
        :param returns_to_use: rendement à utiliser pour mettre en oeuvre la stratégie
        :param strategy: strategie que l'utilisateur souhaite utiliser (parmi celles implémentées)
        :param weighting: schéma de pondération à utiliser (parmi ceux implémentés)
        :return: une liste contenant les poids de chaque actif à la date de rebalancement pour la stratégie souhaitée
        """
        # Tests usuels pour vérifier que les méthodes sont implémentés
        if strategy not in [Portfolio.MOMENTUM_LABEL, Portfolio.MOMENTUM_IDIOSYNCRATIC_LABEL, Portfolio.MAX_SHARPE_LABEL]:
            raise Exception("Stratégie non implémentée")
        if weighting not in Portfolio.WEIGHTING_LABELS:
            raise Exception("Schéma de pondération non implémenté")

        # distinction selon les différentes stratégies implémentées : Momentum
        if strategy == Portfolio.MOMENTUM_LABEL:
            strategy_instance: Momentum = Momentum(returns_to_use.iloc[-self.calculation_window:], self.universe,
                                                    weighting, self.quantile)
        # Stratégie Momentum idiosyncratique
        elif strategy == Portfolio.MOMENTUM_IDIOSYNCRATIC_LABEL:
            # Calcul des rendements du benchmark (= facteur de risque du momentum idiosyncratique)
            ret_bench: pd.DataFrame = self.utils.compute_asset_returns(self.bench, self.periodicity, self.method)
            strategy_instance: IdiosyncraticMomentum = IdiosyncraticMomentum(returns_to_use, self.universe,ret_bench,
                                                    weighting, self.quantile, self.calculation_window)
        # Stratégie Max Sharpe
        elif strategy == Portfolio.MAX_SHARPE_LABEL:
            strategy_instance: MaxSharpe = MaxSharpe(returns_to_use.iloc[-self.calculation_window:], self.universe,
                                                    weighting, self.quantile)
        # Erreur pour toute autre stratégie
        else:
            raise Exception("Aucune stratégie n'est implémentée en dehors du momentum, du momentum idiosyncratique et du max sharpe")

        # Récupération des poids associés à la stratégie
        list_weights_ptf: list = strategy_instance.get_position()
        return list_weights_ptf

    def compute_portfolio_derive(self, idx: int, list_prec_weights: list) -> list:
        """
        Méthode permettant de calculer la dérive des poids entre deux dates de rebalancement dans le cas d'un
        portefeuille long only
        :param idx: indice de la date en cours
        :param list_prec_weights: liste contenant les positions du portefeuille à la date précédente
        :return: liste contenant les positions du portefeuille à la date courante
        """

        # récupération des rendements en remplaçant les valeurs manquantes / -Inf par des 0 pour les calculs
        # Retraitements nécessaires en raison des entrées / sorties de titres de l'univers d'investissement
        returns:pd.DataFrame = self.returns.fillna(0)
        returns.replace(-np.inf, 0, inplace=True)

        # Test sur les dimensions pour vérifier que les calculs sont réalisables
        if len(list_prec_weights) != len(returns.iloc[idx - 1, :]):
            raise Exception("Les listes de poids et de rendement doivent avoir la même taille pour réaliser le calcul")

        # Liste pour stocker les nouveaux poids
        weights: list = []

        # Etape 1 : récupération des rendements à la période précédente
        list_prec_ret: list = returns.iloc[idx - 1, :].values.tolist()

        # Etape 2 : calcul du rendement du portefeuille à la fin de la période précédente
        ptf_ret: float = np.sum(np.multiply(list_prec_weights, list_prec_ret))
        tot_ret: float = 1 + ptf_ret

        # Etape 3 : calcul des nouveaux poids
        for i in range(len(list_prec_weights)):
            old_weight: float = list_prec_weights[i]
            tot_ret_asset: float = 1 + list_prec_ret[i]
            weights.append(old_weight * tot_ret_asset / tot_ret)

        return weights

    def compute_portfolio_value(self, initial_value: float = 100.0) -> list:
        """
        Méthode permettant de calculer la NAV du portefeuille à chaque période
        :param initial_value: valeur initiale du portefeuille (base 100)
        :return: Liste contenant la valeur du portefeuille à chaque date
        """

        # Copie des rendements et des positions
        returns: pd.DataFrame = self.returns.copy()
        positions: pd.DataFrame = self.positions.copy()

        # Pour le calcul de la valeur du portefeuille, les valeurs manquantes sont remplacées par des 0
        returns.replace(np.nan, 0, inplace=True)
        returns.replace(np.inf, 0, inplace=True)
        returns.replace(-np.inf, 0, inplace=True)

        # Vérifications élémentaires
        if positions.isna().any().any():
            raise ValueError("Les positions contiennent des NaN. Vérifiez les entrées dans `positions`.")
        if returns.isna().any().any():
            raise ValueError("éLes prix contiennent des valeurs manquanters. Vérifiez les entrées dans 'returns'. ")

        # Création d'une série pour stocker les NAV du portefeuille (première valeur = 100 par défaut)
        portfolio_value: list = [initial_value]

        # Boucle sur les dates à partir de t=1 car nous avons besoin des rendements en t
        for t in range(1, len(positions)):

            # récupération de l'indice pour les rendements (rendement en t + window - 1 pour la NAV en t)
            ret_idx: int = 36 + (t - 1)

            # Récupération des rendements de la période précédente
            list_asset_returns: list = returns.iloc[ret_idx, :].values.tolist()
            # Récupération des rendements de la période précédente
            list_weight: list = positions.iloc[t - 1].values.tolist()
            # Calcul du rendement du portefeuille à la période précédente
            weighted_returns: float = np.sum(np.multiply(list_weight, list_asset_returns))
            # Calcul de la nouvelle valeur du portefeulle
            portfolio_value.append(portfolio_value[-1] * (1+weighted_returns))
        return portfolio_value


