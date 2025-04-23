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

# Fonction permettant de réaliser un mapping sectoriel
def data_to_sector(prices: pd.DataFrame, sectors:pd.DataFrame) -> dict:
    """
    Fonction permettant de réaliser un mapping sectoriel pour construire des stratégies segmentées
    :param prices: DataFrame contenant les prix de tous les titres de l'univers d'investissement entre 2007 et 2024
    :param sectors: DataFrame contenant le secteur de chaque titre de l'univers d'investissement
    :return: Un dictionnaire qui associe à chaque secteur un dataframe contenant tous les titres qui lui sont rattachés
    """

    # création d'un dictionnaire vide pour stocker les tickers par secteur
    dict_sector: dict = dict()

    # Tous les tickers sans secteur reçoivent le ticker "other"
    sectors.replace(np.nan, "other", inplace=True)

    # Récupération des secteurs présents dans l'univers d'investissement
    sector_array: np.array = pd.unique(sectors.iloc[0])

    # Boucle sur chaque secteur
    for i in range(len(sector_array)):

        # Récupération du secteur qui sera utilisé comme clé
        sector_key: str = sector_array[i]

        # Récupération sous forme de booléen de tous les tickers qui sont rattachés à ce secteur
        sector_bool_array: np.array(bool) = sectors.iloc[0].eq(sector_key)

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
# Les valeurs manquantes sont remplacées par des 0
df_compo.fillna(0, inplace=True)

# Import des données contenant les prix des stocks du MSCI et retraitements
df_msci_stocks: pd.DataFrame = pd.read_excel('data/Final Database MSCI stocks.xlsx')
df_msci_stocks.set_index("Dates", inplace=True)
df_msci_stocks = df_msci_stocks.apply(lambda series: series.loc[:series.last_valid_index()].ffill())
# Les valeurs manquantes sont remplacées par des 0 pour réaliser les traitements ultérieures
df_msci_stocks.replace(np.nan, 0, inplace=True)
df_msci_stocks.fillna(0, inplace=True)

# Import des données relatives au secteur de chaque ticker
df_sector: pd.DataFrame = pd.read_excel('data/Master 272 - AM - Projet indices.xlsx', sheet_name="Secteurs")
df_sector.set_index("Ticker", inplace = True)

# Réalisation du mapping sectoriel
dict_tickers_sectors: dict = data_to_sector(df_msci_stocks, df_sector)

# Booléan pour déterminer s'il faut exporter les données obtenues ou non
save: bool = True

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

"""
Configuration des éléments du backtest qui sont utilisés pour toutes les stratégies
"""

list_rebalancing: list = ['monthly','quarterly']
list_decile_ranking: list = [0.1, 0.2, 0.25]
list_segmentation: list = [None,"sectorial"]
method: str = "discret"
weighting: str = 'ranking'
periodicity: str = 'monthly'

# Liste contenant le nom de chaque portefeuille généré
list_ptf_name: list = ["Monthly decile without segmentation",
                       "Monthly quintile without segmentation",
                       "Monthly quartile without segmentation",
                       "Quarterly decile without segmentation",
                       "Quarterly quintile without segmentation",
                       "Quarterly quartile without segmentation",
                       "Monthly decile with sectorial segmentation",
                       "Monthly quintile with sectorial segmentation",
                       "Monthly quartile with sectorial segmentation",
                       "Quarterly decile with sectorial segmentation",
                       "Quarterly quintile with sectorial segmentation",
                       "Quarterly quartile with sectorial segmentation",
                       ]

"""
Stratégie à backtester
selon plusieurs configurations possibles
"""

# Etape 1 : définition de la stratégie à mettre en oeuvre
strat = 'momentum idiosyncratique'
# Dans le cas d'une stratégie Mean Reverting, il faut utiliser un fenêtre d'un mois
calculation_window:int = 1
# Nom de la stratégie
nom: str = "Momentum Idiosyncratique Mean Revert 1 mois - "

# Etape 2 : Initialisation du dataframe utilisé pour stocker les NAV
df_nav: pd.DataFrame = pd.DataFrame()

# Etape 3 : Calcul des NAV et de la performance de la stratégie pour toutes les configurations sélectionnées
# Boucle sur la segmentation
cpt = 0
# Boucle sur les règles de sélection
for segmentation in list_segmentation:
    # Boucle sur les fréquences de rebalancement
    for rebalancing in list_rebalancing:
        # Boucle sur les déciles utilisés
        for decile in list_decile_ranking:
            # Instanciation du portefeuille
            ptf: Portfolio = Portfolio(df_msci_stocks,
                                    universe=df_compo,
                                    bench=msci_index,
                                    dict_sector=dict_tickers_sectors,
                                    periodicity=periodicity,
                                    rebalancement=rebalancing,
                                    method=method,
                                    strat=strat,
                                    weighting=weighting,
                                    calculation_window=calculation_window,
                                    quantile=decile,
                                    segmentation=segmentation)

            # Réalisation du backtest
            ptf.run_backtest()

            # Calcul des métriques de risque et de performance
            ptf_metrics: Metrics = Metrics(ptf.portfolio_value["Nav"], method,
                                        frequency=periodicity,
                                        benchmark = msci_index['MSCI World'])

            # Récupération du nom du portefeuille
            nom_ptf: str = nom + list_ptf_name[cpt]

            # Stockage de la NAV
            df_nav[nom_ptf] = ptf.portfolio_value['Nav']

            # Stockage des métriques
            if cpt == 0:
                stat_ptf:pd.DataFrame = ptf_metrics.synthesis(nom_ptf, df_msci_stats)
            else:
                stat_ptf:pd.DataFrame = ptf_metrics.synthesis(nom_ptf, stat_ptf)
            cpt += 1


# Sauvegarde et export des résultats
if save:
    df_nav.to_excel("Résultats stratégie Momentum idiosyncratique Mean Revert.xlsx")
    stat_ptf.to_excel("Performances stratégie Momentum idiosyncratique Mean Revert.xlsx")
