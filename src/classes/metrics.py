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
    Inputs : 
        - ptf_nav, une liste contenant l'évolution de la NAV du portefeuille dans le temps
        - method : méthode pour calculer les rendements
        - frequency : fréquence à utiliser pour le calcul des métriques
        - benchmark : dataframe contenant le benchmark si on souhaite utiliser cette classe
        pour faire des comparaisons portefeuille / benchmark
    """
    def __init__(self, ptf_nav:pd.DataFrame, method:str, frequency:str, benchmark:pd.DataFrame = None):
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
        Méthode permettant de calculer le rendement annualisé d'une stratégie.
        Intérêt dans le cadre du projet : comparer la performance du nouveau pilier (action / obligation) avec
        celui de la grille de départ
        """

        # Récupération des rendements
        ret_ptf: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)

        # Calcul du total return
        total_return: float = (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

        # Calcul du rendement annualisé
        annualized_return: float = (1+total_return) ** (self.annualization_factor / ret_ptf.shape[0]) - 1
        return {"total_return": total_return, "annualized_return": annualized_return}

    def compute_annualized_vol(self):
        """
        Méthode permettant de calculer la volatilité annualisée d'une stratégie
        """
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        vol: float = np.std(ptf_ret)
        return vol * np.sqrt(self.annualization_factor)

    def compute_sharpe_ratio(self, rf:float = 0):
        """
        Méthode permettant de calculer le sharpe ratio de l'investissement
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        ann_vol: float = self.compute_annualized_vol()
        sharpe:float = (ann_ret - rf)/ann_vol
        return sharpe

    def compute_downside_vol(self)->float:
        """
        Méthode permettant de calculer la volatilité à la baisse
        :return:
        """

        # Etape 1 : calcul de la différence entre rendement et taux sans risuqe (= rendement) et
        # récupération des cas où la diff est négative
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        neg_ptf_ret: pd.DataFrame = ptf_ret[ptf_ret < 0]

        # Etape 2 : Calcul de la volatilité sur les rendements negatifs
        downside_vol: float = np.std(neg_ptf_ret)

        # Etape 3 : Récupération de la downside vol annualisée
        return downside_vol * np.sqrt(self.annualization_factor)

    def compute_sortino(self, rf:float = 0):
        """
        Méthode permettant de calculer le ratio de Sortino
        :return:
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        downside_vol: float = self.compute_annualized_vol()
        if downside_vol == 0:
            raise Exception("Calcul impossible pour une volatilité à la baisse nulle")
        sortino:float = (ann_ret - rf)/downside_vol
        return sortino

    def compute_beta_and_alpha(self) -> dict:
        """
        Méthode permettant de calculer le beta entre l'indice et le portefeuille
        :return:
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

    def compute_tracking_error(self):
        """
        Méthode permettant de calculer la tracking error du fonds
        :return:
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
        :return:
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

    """
    A maj : plus nécessaire d'avoir bench et ptf
    """

    def synthesis(self,nom_ptf:str, df_stats_bench: pd.DataFrame = None):
        """
        Permet de comparer les performances d'un portefeuille avec celles d'un benchmark
        précisé par l'utilisateur
        """
        # Calcul des statistiques pour le portefeuille
        df_stats_fond:pd.DataFrame = self.display_stats()
        df_stats_fond.rename(columns={0:f"Performances du {nom_ptf}"}, inplace=True)

        # Cas où l'utilisateur souhaite avoir la performance du portefeuille uniquement
        if df_stats_bench is None:
            return df_stats_fond

        # Comparaison par rapport au benchmark
        df_results: pd.DataFrame = pd.concat([df_stats_fond, df_stats_bench], axis=1)
        return df_results

    def display_stats(self):
        """
        Méthode permettant d'afficher les différentes statistiques descriptives pour un portefeuille
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
        """
        print(f"Rendement annualisé du portefeuille (en %) : {round(ann_ret * 100, 2)}.")
        print(f"Volatilité annualisée du portefeuille (en %) : {round(ann_vol * 100, 2)}.")
        print(f"Sharpe ratio annualisé du portefeuille : {round(sharpe,2)}.")
        print(f"Total return du portefeuille (en %) : {round(tot_ret * 100, 2)}")
        print(f"Vol à la baisse du portefeuille annualisée : {round(downside_vol*100, 2)}")
        print(f"Ratio de sortino annualisé du portefeuille : {round(sortino,2)}")
        print(f"Alpha du portefeuille annualisé : {alpha}")
        print(f"Beta du portefeuille : {beta}")
        print(f"Tracking-error du portefeuille : {te}")
        print(f"Max Draw Down du portefeuille : {mdd}")
        """
        return pd.DataFrame.from_dict(stats_dict, orient='index')

    """
    def display_bench_stats(self):

        #Méthode pour afficher les statistiques du benchmark
        if self.bench.empty:
            raise Exception("Aucun benchmark renseigné")

        bench_ret: pd.DataFrame = self.compute_returns(self.bench)
        ann_ret: float = self.compute_performance(self.bench, bench_ret)["annualized_return"]
        ann_vol: float = self.compute_annualized_vol(bench_ret)
        tot_ret: float = self.compute_performance(self.bench, bench_ret)["total_return"]
        sharpe: float = self.compute_sharpe_ratio(self.bench, bench_ret)

        print(f"Rendement annualisé du benchmark (en %) : {round(ann_ret * 100, 2)}.")
        print(f"Volatilité annualisée du benchmark (en %) : {round(ann_vol * 100, 2)}.")
        print(f"Sharpe ratio du benchmark : {round(sharpe,2)}.")
        print(f"Total return du benchmark (en %) : {round(tot_ret * 100, 2)}")
    """
