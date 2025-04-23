import numpy as np
import pandas as pd
from src.classes.utilitaire import Utils

class Metrics:

    DISCRET_LABEL = "discret"
    CONTINU_LABEL = "continu"
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"

    """
    Classe permettant de calculer un ensemble de métriques de performance et de risque pour un portefeuille
    """
    def __init__(self, ptf_nav:pd.DataFrame, method:str, frequency:str, benchmark:pd.DataFrame = None):
        """
        :param ptf_nav: DataFrame contenant la NAV du portefeuille à chaque date
        :param method: méthode de calcul des rendements
        :param frequency: fréquence des données pour le calcul des rendements
        :param benchmark: DataFrame contenant la NAV du benchmark du portefeuille
        """
        self.nav: pd.DataFrame = ptf_nav
        self.bench: pd.DataFrame = benchmark
        self.method: str = method
        self.frequency: str = frequency

        self.utils: Utils = Utils()

        self.annualization_factor: int = self.compute_annualization_factor()

    def compute_annualization_factor(self):
        """
        Méthode permettant de déterminer le coefficient d'annualisation
        à utiliser selon la périodicité des données
        """
        if self.frequency == Metrics.MONTHLY_LABEL:
            return 12
        elif self.frequency == Metrics.QUARTERLY_LABEL:
            return 4
        elif self.frequency == Metrics.YEARLY_LABEL:
            return 1
        else:
            raise Exception("Les calculs pour une périodicité autre ne sont pas implémentés")

    def compute_performance(self)->dict:
        """
        Méthode permettant de calculer le rendement annualisé et le rendement total d'une stratégie.
        :return: dictionnaire contenant le rendement annualisé et le rendement total
        """

        # Récupération des rendements
        ret_ptf: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)

        # Calcul du total return
        total_return: float = (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

        # Calcul du rendement annualisé
        annualized_return: float = (1+total_return) ** (self.annualization_factor / ret_ptf.shape[0]) - 1
        return {"total_return": total_return, "annualized_return": annualized_return}

    def compute_annualized_vol(self)->float:
        """
        Méthode permettant de calculer la volatilité annualisée d'un portefeuille
        :return: volatilité annualisée d'un portefeuille
        """
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        vol: float = np.std(ptf_ret)
        return vol * np.sqrt(self.annualization_factor)

    def compute_sharpe_ratio(self, rf:float = 0)->float:
        """
        Méthode permettant de calculer le Sharpe ratio d'un portefeuille
        :param rf: taux sans risque (0 par hypothèse)
        :return: sharpe ratio du portefeuille
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        ann_vol: float = self.compute_annualized_vol()
        sharpe:float = (ann_ret - rf)/ann_vol
        return sharpe

    def compute_downside_vol(self)->float:
        """
        Méthode permettant de calculer la volatilité à la baisse du portefeuille
        :return: volatilité à la baisse du portefeuille
        """

        # Etape 1 : calcul de la différence entre rendement et taux sans risuqe (= rendement) et
        # récupération des cas où la diff est négative
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        neg_ptf_ret: pd.DataFrame = ptf_ret[ptf_ret < 0]

        # Etape 2 : Calcul de la volatilité sur les rendements negatifs
        downside_vol: float = np.std(neg_ptf_ret)

        # Etape 3 : Récupération de la downside vol annualisée
        return downside_vol * np.sqrt(self.annualization_factor)

    def compute_sortino(self, rf:float = 0)->float:
        """
        Méthode permettant de calculer le ratio de Sortino du portefeuille
        :return: ratio de Sortino du portefeuille
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        downside_vol: float = self.compute_annualized_vol()
        if downside_vol == 0:
            raise Exception("Calcul impossible pour une volatilité à la baisse nulle")
        sortino:float = (ann_ret - rf)/downside_vol
        return sortino

    def compute_beta_and_alpha(self) -> dict:
        """
        Méthode permettant de calculer le beta et l'alpha du portefeuille
        :return: Dictionnaire contenant le beta et l'alpha du portefeuille
        """

        # Etape 1 : Calcul des rendements de l'indice et du portefeuille
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        bench_ret, ptf_ret = self.utils.compute_asset_returns(self.bench, self.frequency, self.method).align(ptf_ret, join="inner")

        # Etape 3 : Calcul de la covariance et de la variance
        cov: float = np.cov(bench_ret, ptf_ret)[0, 1]
        var:float = np.var(bench_ret)
        if var == 0:
            raise Exception("La variance du benchmark ne peut pas être nul")

        # Etape 4 : Calcul du beta
        beta: float = cov/var

        # Etape 5 : Calcul de l'alpha
        alpha: float = (ptf_ret.mean() - beta * bench_ret.mean()) * self.annualization_factor

        return {'alpha':alpha, 'beta':beta}

    def compute_tracking_error(self)->float:
        """
        Méthode permettant de calculer la tracking error du fonds
        :return: Tracking error du fonds
        """

        # Etape 1 : Calcul des rendements du portefeuille et du bench
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        bench_ret: pd.DataFrame = self.utils.compute_asset_returns(self.bench, self.frequency, self.method)

        # Etape 2 : Calcul de la différence de rendements p/R au portefeuiille
        excess_ret: pd.DataFrame = ptf_ret - bench_ret

        # Etape 3 : Calcul de la TE
        tracking_error: float = excess_ret.std() * np.sqrt(self.annualization_factor)
        return tracking_error

    def compute_max_draw_down(self)->float:
        """
        Méthode permettant de calculer le max draw down
        d'un portefeuille
        :return: MaxDrawDown du portefeuille
        """

        # Etape 1 : initialisation du hwm et de la liste contenant les drawdowns
        list_drawdowns: list = []
        hwm:float = self.nav[0]

        # Etape 2 : boucle pour calculer le drawdown a chaque date
        for i in range(self.nav.shape[0]):
            # Mise du hwm si nouveaux plus haut
            if self.nav[i] > hwm:
                hwm = self.nav[i]

            # Calcul du drawdown
            drawdown: float = (self.nav[i] - hwm)/hwm
            list_drawdowns.append(drawdown)

        # Etape 3 : calcul du max drawdown
        return min(list_drawdowns)

    def synthesis(self,nom_ptf:str, df_stats_bench: pd.DataFrame = None)->pd.DataFrame:
        """
        Méthode de comparer les performances de plusieurs portefeuilles entre eux
        :param nom_ptf: Nom du portefeuille étudié
        :param df_stats_bench: DataFrame contenant statistiques des autres portefeuilles de références
        :return: DataFrame contenant les statistiques de tous les portefeuilles
        """
        # Calcul des statistiques pour le portefeuille
        df_stats_fond:pd.DataFrame = self.display_stats()
        df_stats_fond.rename(columns={0:f"Performances du {nom_ptf}"}, inplace=True)

        # Cas où l'utilisateur souhaite avoir la performance du portefeuille uniquement
        if df_stats_bench is None:
            return df_stats_fond

        # Comparaison par rapport au benchmark
        df_results: pd.DataFrame = pd.concat([df_stats_bench, df_stats_fond], axis=1)
        return df_results

    def display_stats(self)->pd.DataFrame:
        """
        Méthode permettant de calculer les statistiques descriptives du portefeuille
        :return: DataFrame contenant les statistiques descriptives du portefeuille
        """
        #
        dict_perf: dict = self.compute_performance()
        dict_alpha: dict = self.compute_beta_and_alpha()

        ann_ret: float = dict_perf["annualized_return"]
        ann_vol: float = self.compute_annualized_vol()
        tot_ret: float = dict_perf["total_return"]
        sharpe: float = self.compute_sharpe_ratio()
        downside_vol: float = self.compute_downside_vol()
        sortino: float = self.compute_sortino()
        alpha:float = dict_alpha['alpha']
        beta:float = dict_alpha['beta']
        te: float = self.compute_tracking_error()
        mdd:float = self.compute_max_draw_down()

        stats_dict: dict = {
            "Rendement annualisé en % ":round(ann_ret * 100, 2),
            "Total return en %":round(tot_ret * 100, 2),
            "Volatilité annualisée en %":round(ann_vol * 100, 2),
            "Sharpe ratio annualisé":round(sharpe,2),
            "Downside Vol annualisée en %":round(downside_vol*100, 2),
            "Ratio de Sortino annualisé":round(sortino,2),
            "Alpha":round(alpha,2),
            "Beta":round(beta, 2),
            "Tracking-error du portefeuille en %":round(te*100,2),
            "MaxDrawDown":round(mdd,2)
        }
        return pd.DataFrame.from_dict(stats_dict, orient='index')
