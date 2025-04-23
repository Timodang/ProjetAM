import pandas as pd
import numpy as np

from src.classes.metrics import Metrics
from src.classes.portfolio import Portfolio

"""
#####################
# Main du projet ####
#####################
"""

"""
Le calcul de l'indice MSCI world est réalisé indépendamment directement depuis Excel.
Les résultats sont importés dans ce programme pour être utilisé
"""

# Fonction pour calculer l'indice ===> old
"""
def calc_index(prices:pd.DataFrame, weights:pd.DataFrame) -> pd.DataFrame:

    Fonction permettant de calculer la valeur du MSCI world date par date
    :param prices:
    :param weights:
    :return:

    # Remplacements des NA par des 0
    prices.fillna(0, inplace=True)
    weights.fillna(0, inplace=True)

    # Test pour vérifier que les calculs sont réalisables
    if prices.shape[0] != weights.shape[0]:
        raise Exception("Il doit y avoir le même nombre de dates pour réaliser les calculs")
    if prices.shape[1] != weights.shape[1]:
        raise Exception("Il doit y avoir le même nombre de tickers pour réaliser les calculs")

    # Création du dataframe pour stocker les valeurs du MSCI
    df_msci: pd.DataFrame = pd.DataFrame(columns = ['MSCI World'], index=prices.index)

    # Boucle pour calculer les valeurs du MSCI world
    for i in range(prices.shape[0]):

        # Récupération des valeurs pour la date en cours
        vect_price: np.ndarray = prices.iloc[i,:]
        vect_weight: np.ndarray = weights.iloc[i, :]

        # Calcul et stockage de la valeur du MSCI world
        msci_value: float = np.matmul(vect_weight, vect_price.T)
        df_msci.iloc[i, 0] = msci_value

    return df_msci
"""

# Fonction permettant de réaliser un mapping sectoriel
def data_to_sector(prices: pd.DataFrame, df_sector:pd.DataFrame) -> dict:
    """
    Fonction permettant de réaliser un mapping sectoriel pour construire des stratégies segmentées
    :param prices:
    :param df_sector:
    :return:
    """

    # création d'un dictionnaire vide pour stocker les tickers par secteur
    dict_sector: dict = dict()

    # Tous les tickers sans secteur reçoivent le ticker "other"
    df_sector.replace(np.nan, "other", inplace=True)

    # Récupération des secteurs présents dans l'univers d'investissement
    sector_array: np.array = pd.unique(df_sector.iloc[0])

    # Boucle sur chaque secteur
    for i in range(len(sector_array)):

        # Récupération du secteur qui sera utilisé comme clé
        sector_key: str = sector_array[i]

        # Récupération sous forme de booléen de tous les tickers qui sont rattachés à ce secteur
        sector_bool_array: np.array(bool) = df_sector.iloc[0].eq(sector_key)

        # Filtre sur les tickers rattachés à ce secteurs
        df_prices_sector: pd.DataFrame = prices.loc[:, sector_bool_array]
        df_prices_sector.fillna(0, inplace=True)

        # Ajout au dictionnaire
        dict_sector[sector_key] = df_prices_sector

    # Suppression des others
    del dict_sector["other"]
    return dict_sector

"""
Import des données
"""

# Import des données contenant les compositions mensuelles du S&P 500
df_compo:pd.DataFrame = pd.read_excel('data/Master 272 - AM - Projet indices.xlsx', sheet_name="Composition MSCI World")
df_compo.set_index("Dates", inplace=True)
df_compo.fillna(0, inplace=True)

# Import des données contenant les prix des stocks du MSCI et retraitements
df_msci_stocks: pd.DataFrame = pd.read_excel('data/Final Database MSCI stocks.xlsx')
df_msci_stocks.set_index("Dates", inplace=True)
df_msci_stocks = df_msci_stocks.apply(lambda series: series.loc[:series.last_valid_index()].ffill())
# Les valeurs manquantes sont remplacées par des 0 pour réaliser les traitements ultérieures
df_msci_stocks.replace(np.nan, 0, inplace=True)
df_msci_stocks.fillna(0, inplace=True)

# Import des données contenant la segmentation sectorielle
df_sector: pd.DataFrame = pd.read_excel('data/Master 272 - AM - Projet indices.xlsx', sheet_name="Secteurs")
df_sector.set_index("Ticker", inplace = True)

# Réalisation du mapping sectoriel
dict_tickers_sectors: dict = data_to_sector(df_msci_stocks, df_sector)

# Choix de sauvegarder ou non les données
save: bool = False

"""
Import du MSCI World et présentation des performances
"""

# Import du MSCI World
msci_index: pd.DataFrame = pd.read_excel("data/NAV MSCI World.xlsx")
msci_index.set_index("Dates", inplace=True)

# Calcul des performances
msci_perf: Metrics = Metrics(msci_index['MSCI World'], "discret", "monthly", msci_index['MSCI World'])
print("Affichage des performances pour le MSCI world : ")
df_msci_stats: pd.DataFrame = msci_perf.synthesis("Benchmark")

# Visualisation graphique
# Visualisation.plot_cumulative_returns(msci_perf.compute_returns(msci_index))

# Sauvegarde des résultats
if save:
    msci_index['MSCI World'].to_excel("NAV MSCI World.xlsx")

"""
Paramètres à utiliser pour le backtest de toutes les stratégies
"""
list_rebalancing: list = ['monthly','quarterly']
list_decile_ranking: list = [0.1, 0.2, 0.25]
list_segmentation: list = [None,"sectorial"]
method: str = "discret"
weighting: str = 'ranking'


"""
Première question : Stratégie Momentum
"""

# Etape 1 : définition de la stratégie à mettre en oeuvre
strat = 'momentum'
calculation_window:int = 12
periodicity: str = 'quarterly'

# Etape 2 : initialisation des listes utiliser pour stocker les résultats du backtest pour plusieurs configuration
list_nav_max_sharpe: list = []
list_metrics_max_sharpe: list = []

# Etape 3 : boucle pour faire les calculs sur toutes les configurations souhaitées
ptf_momentum: Portfolio = Portfolio(df_msci_stocks,
                                    universe=df_compo,
                                    bench=msci_index,
                                    dict_sector=dict_tickers_sectors,
                                    periodicity=periodicity,
                                    rebalancement=list_rebalancing[1],
                                    method=method,
                                    strat=strat,
                                    weighting=weighting,
                                    calculation_window=calculation_window,
                                    quantile=list_decile_ranking[0],
                                    segmentation=None)
ptf_momentum.run_backtest()
ptf_momentum_metrics: Metrics = Metrics(ptf_momentum.portfolio_value["Nav"], method,
                                        frequency=periodicity,
                                        benchmark = msci_index['MSCI World'])
stat_ptf_momentum = ptf_momentum_metrics.synthesis("Portefeuille momentum test", df_msci_stats)

# Voir pour passer en liste et graph
a = 3
# Décile / Quintile / quartile ok
# Intra sectoriel ou non
# Rebalancement Monthly ou Quarterly ok
# 1 an ok (à revoir)
# revoir MSCI world
# Tableau perf vs bench ok
# Graph


"""
Deuxième question : Stratégie Momentum idiosyncratique
"""

"""
strat: str = 'momentum idiosyncratique'
calculation_window:int = 12
periodicity: str = 'quarterly'

# Etape 2 : initialisation des listes utiliser pour stocker les résultats du backtest pour plusieurs configuration
list_nav_momentum_id: list = []
list_metrics_momentum_id: list = []

# Etape 3 : boucle pour faire les calculs sur toutes les configurations souhaitées
ptf_momentum_id: Portfolio = Portfolio(df_msci_stocks,
                                    universe=df_compo,
                                    bench=msci_index,
                                    dict_sector=dict_tickers_sectors,
                                    periodicity=periodicity,
                                    rebalancement=list_rebalancing[0],
                                    method=method,
                                    strat=strat,
                                    weighting=weighting,
                                    calculation_window=calculation_window,
                                    quantile=list_decile_ranking[0],
                                    segmentation="sectorial")
ptf_momentum_id.run_backtest()
ptf_momentum_id_metrics: Metrics = Metrics(ptf_momentum_id.portfolio_value, method,
                                        frequency="monthly",
                                        benchmark = msci_index['MSCI World'])
stat_ptf_id_momentum = ptf_momentum_id_metrics.synthesis("Portefeuille momentum test", df_msci_stats)
a=3
"""
"""
Troisième question : graphique
"""

"""
Quatrième question : Momentum contrarian
"""

"""
Cinquième question : Momentum idiosyncratique contrarian
"""

"""
Sixième question : graphique
"""

"""
Septième question : Max Sharpe
"""
"""
strat: str = 'max sharpe'
calculation_window:int = 12
periodicity: str = 'monthly'

# Etape 2 : initialisation des listes utiliser pour stocker les résultats du backtest pour plusieurs configuration
list_nav_max_sharpe: list = []
list_metrics_max_sharpe: list = []

# Etape 3 : boucle pour faire les calculs sur toutes les configurations souhaitées
ptf_max_sharpe: Portfolio = Portfolio(df_msci_stocks,
                                    universe=df_compo,
                                    bench=msci_index,
                                    dict_sector=dict_tickers_sectors,
                                    periodicity=periodicity,
                                    rebalancement=list_rebalancing[0],
                                    method=method,
                                    strat=strat,
                                    weighting=weighting,
                                    calculation_window=calculation_window,
                                    quantile=list_decile_ranking[0],
                                    segmentation="sectorial")
ptf_max_sharpe.run_backtest()
ptf_max_sharpe_metrics: Metrics = Metrics(ptf_max_sharpe.portfolio_value, method,
                                        frequency="monthly",
                                        benchmark = msci_index['MSCI World'])
stat_ptf_max_sharpe = ptf_max_sharpe_metrics.synthesis("Portefeuille momentum test", df_msci_stats)
"""