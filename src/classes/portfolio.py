import pandas as pd
import numpy as np
import datetime as datetime
from operator import add

from dateutil.relativedelta import relativedelta

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
    Classe qui s'occupe des différentes métriques et de la composition des portefeuilles pour chaque pilier.
    Arguments : 
        - asset_prices : dataframe contenant les prix des actifs de l'univers considéré pour ce pilier, 
        - periodicity : période pour le calcul des rendements (daily, weekly, monthly, quarterly, yearly)
        - rebalancement : fréquence pour le calcul de la date de rebalancement
        - method : méthode pour le calcul des rendements (discret, continu)
        - list_strat : liste des stratégies à appliquer dans la poche d'alpha
        - list_weighting : liste des schémas de pondération à appliquer pour chaque stratégie de la poche d'alpha
        - calculation_window : taille de la fenêtre à utiliser pour les stratégies d'alpha
        - asset_class : classe d'actif pour laquelle on effectue le backtest (equity, bond, ...)
    """

    def __init__(self, df_prices: pd.DataFrame, universe:pd.DataFrame, bench:pd.DataFrame, dict_sector: dict,
                 periodicity: str, rebalancement: str, method: str,
                 strat: str, weighting: str, calculation_window: int, quantile: float, segmentation:str = None):

        # Vérifications de cohérence
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

        # Calcul de l'allocation du portefeuille à chaque date
        self.positions: pd.DataFrame = pd.DataFrame()

        # Calcul de la NAV du portefeuille (= valeur du pilier)
        self.portfolio_value: pd.Series = pd.Series()

    def run_backtest(self):
        """
        Méthode permettant de réaliser le backtest et de récupérer les positions et la NAV du fonds
        """

        # 1ere étape : calcul des poids
        portfolio_positions: pd.DataFrame = self.build_portfolio()
        self.positions = portfolio_positions

        # 2eme étape : Calcul de la NAV du portefeuille
        portfolio_value: list = self.compute_portfolio_value()
        self.portfolio_value = pd.DataFrame(portfolio_value, columns = ["Nav"], index = self.positions.index)

    def rebalancing_date_2(self, serie_date:pd.Series, index:int) -> datetime:
        """
        Méthode permettant de déterminer la prochaine date de rebalancement à partir d'une liste de données mensuelles
        contenant toutes les dates de la période d'étude
        :param serie_date:
        :param index:
        :return:
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


    def rebalancing_date(self, prec_date: datetime = None) -> datetime:
        """
        Méthode permettant de calculer la nouvelle date de rebalancement
        à partir de la date de rebalancement précédente
        """

        # On rebalance à la première itération du backtest (=> à voir, peut équipondérer sinon)
        if self.rebalancement == Portfolio.MONTHLY_LABEL:
            next_rebalancing_date: datetime = prec_date + relativedelta(months=1)
        elif self.rebalancement == Portfolio.QUARTERLY_LABEL:
            next_rebalancing_date: datetime = prec_date + relativedelta(months=3)
        elif self.rebalancement == Portfolio.YEARLY_LABEL:
            next_rebalancing_date: datetime = prec_date + relativedelta(years=1)
        else:
            raise Exception("La fréquence de rebalancement souhaitée n'est pas implémentée")

        return next_rebalancing_date

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
        Méthode  pour calculer la valeur du portefeuille
        strategy : Momentum
        """

        # récupération de l'indice des rebalancements
        int_rebal_freq: int = self._rebal_to_index()

        # On récupère le nombre de rendements disponibles
        length: int = self.returns.shape[0]

        # Récupération de la première date de rebalancement
        rebalancing_date: datetime = self.returns.index[self.calculation_window]
        # premier indice à partir duquel on réalise les calculs
        # index_strat: int = self.calculation_window * 3
        index_strat: int = 36

        # Initialisation du dataframe de positions / poids
        positions: pd.DataFrame = pd.DataFrame(0.0, index=self.returns.index,
                                               columns=self.returns.columns)

        # Boucle pour construire le portefeuille à partir de la première date de rebalancement
        for idx in range(index_strat, length):

            # Récupération de la date courante
            date: datetime = self.returns.index[idx]

            # 1ere étape : Calcul des poids du portefeuille
            # 1er cas : date >= date de rebalancement ==> on rebalance
            if date >= rebalancing_date:

                # Récupération du premier indice pour le calcul des rendements
                # begin_idx: int = idx - self.calculation_window * 3
                begin_idx: int = idx - 36

                # Calcule de la nouvelle date de rebalancement tant qu'on est pas à la dernière date
                if idx < length - int_rebal_freq:
                    rebalancing_date = self.rebalancing_date_2(self.returns.index, idx)

                # Récupération des signaux associés à cette date selon les stratégies / schémas implémentés
                positions.iloc[idx, :] = self._compute_portfolio_position(
                    self.returns.iloc[begin_idx:idx, :], self.strategy,
                    self.weighting)

            # 2e cas : la date n'est pas une date de rebalancement
            # Récupération des poids précédents et calcul des nouveaux poids
            else:
                prec_weights: list = positions.iloc[idx - 1, :]
                positions.iloc[idx, :] = self.compute_portfolio_derive(idx, prec_weights)

        # On ne conserve pas les dates inférieures avant la première date de rebalancement
        positions = positions.iloc[index_strat:, ]
        return positions

    def _compute_portfolio_position(self, returns_to_use: pd.DataFrame, strategy:str, weighting:str) -> list:
        """
        Méthode permettant de déterminer le poids du portefeuille à chaque période de rebalancement
        en appliquant une éventuelle segmentation sectorielle
        :param returns_to_use:
        :param strategy:
        :param weighting:
        :return:
        """

        # Gestion des erreurs pour déterminer si l'utilisateur souhaite réaliser une segmentation sectorielle ou non
        if self.segmentation != Portfolio.SECTOR_LABEL and self.segmentation is not None:
            raise Exception("Segmentation non implémentée")
        elif self.segmentation is None:
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
                # Récupération des titres à utiliser
                df_prices_sector: pd.DataFrame = value

                # Calcul des rendements et alignement pour ne conserver que les dates de la période d'étude
                ret_sectors : pd.DataFrame = self.utils.compute_asset_returns(df_prices_sector, self.periodicity, self.method)
                ret_sectors = ret_sectors[ret_sectors.index.isin(returns_to_use.index)]

                # Calcul des poids pour la stratégie
                list_weights_sector: list = self._get_weight(ret_sectors, strategy, weighting)

                # Ajout à la liste des poids finaux du portefeuille pour la date considéré
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

        # distinction selon les différentes stratégies implémentées : Momentul
        if strategy == Portfolio.MOMENTUM_LABEL:
            strategy_instance: Momentum = Momentum(returns_to_use.iloc[-self.calculation_window:], self.universe,
                                                    weighting, self.quantile)
        # Stratégie Momentum idiosyncratique
        elif strategy == Portfolio.MOMENTUM_IDIOSYNCRATIC_LABEL:
            # Calcul des rendements du benchmark (= facteur de risque du momentum idiosyncratique)
            ret_bench: pd.DataFrame = self.utils.compute_asset_returns(self.bench, self.periodicity, self.method)
            strategy_instance: IdiosyncraticMomentum = IdiosyncraticMomentum(returns_to_use, self.universe,ret_bench,
                                                    weighting, self.quantile)
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
        :param idx:
        :param list_prec_weights:
        :return:
        """

        # récupération des rendements en remplaçant les valeurs manquantes par des 0 pour les calculs
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

        # Etape 3 : calcul des poids
        for i in range(len(list_prec_weights)):
            old_weight: float = list_prec_weights[i]
            tot_ret_asset: float = 1 + list_prec_ret[i]
            weights.append(old_weight * tot_ret_asset / tot_ret)

        return weights

    def compute_portfolio_value(self, initial_value: float = 100.0) -> list:
        """
        Méthode permettant de calculer la NAV du portefeuille à chaque période
        :param initial_value: valeur initiale du portefeuille (base 100)
        :return:
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

        # Création d'une série pour stocker les NAV du portefeuille
        portfolio_value: list = [initial_value]

        # Boucle sur les dates à partir de t=1 car nous avons besoin des rendements en t
        for t in range(1, len(positions)):

            # récupération de l'indice pour les rendements (rendement en t + window - 1 pour la NAV en t)
            ret_idx: int = self.calculation_window + (t - 1)

            # Pb avec le calcul de la valeur (= multiplication de matrice)
            list_asset_returns: list = returns.iloc[ret_idx, :].values.tolist()
            list_weight: list = positions.iloc[t - 1].values.tolist()
            weighted_returns: float = np.sum(np.multiply(list_weight, list_asset_returns))
            portfolio_value.append(portfolio_value[-1] * (1+weighted_returns))
        return portfolio_value

    # old
    """
    def compute_ptf_weight(self, returns_to_use: pd.DataFrame, list_strategies: list, list_weighting: list) -> list:

        Méthode permettant de générer les poids d'un portefeuille
        à une date t à partir des poids générés par plusieurs stratégies.

        Hypothèse : chaque stratégie représente le même poids dans le portefeuille
        final (peut être optimisé)

        arguments :
        - list_strategies : liste contenant les stratégies à appliquer au sein du portefeuille
        - list_weighting : liste contenant les schémas de pondérations à appliquer au sein du portefeuille

        # pondération de chaque stratégie dans le portefeuille
        # a adapter pour la semgnetation
        scaling_factor: float = 1.0 / len(list_strategies)

        # Liste pour stocker les poids du portefeuille
        list_weights_ptf: list = [0] * returns_to_use.shape[1]

        # boucle sur les stratégies
        for i in range(len(list_strategies)):
            # récupération de la stratégie et du schéma de pondération associé
            strat: str = list_strategies[i]
            weighting: str = list_weighting[i]

            if strat not in Portfolio.STRAT_LABELS:
                raise Exception("Stratégie non implémentée")

            if weighting not in Portfolio.WEIGHTING_LABELS:
                raise Exception("Schéma de pondération non implémenté")

            # distinction selon les différentes stratégies implémentées
            if strat == Portfolio.STRAT_LABELS[0]:
                strategy_instance: Momentum = Momentum(returns_to_use,
                                                       weighting, self.quantile)
            # Ajouter un cas pour chaque stratégie
            else:
                raise Exception("To be implemented")

            list_weight_strat: list = strategy_instance.get_position()
            if len(list_weight_strat) != len(list_weights_ptf):
                raise Exception(
                    "Les séries de poids générés par les stratégies / le portefeuille doivent avoir la même longueur")

            # Mettre à jour les poids au sein du portefeuille
            list_weight_strat = [weight * scaling_factor for weight in list_weight_strat]
            list_weights_ptf = list(map(add, list_weights_ptf, list_weight_strat))

        return list_weights_ptf
    """



