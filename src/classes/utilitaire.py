import pandas as pd
import numpy as np

class Utils:
    """
    Classe contenant des méthodes utilisées dans plusieurs autres classes
    Méthode :
    - compute_asset_returns : méthode utilisée pour calculer des rendements discrets ou continus
    """

    @staticmethod
    def compute_asset_returns(asset_prices: pd.DataFrame, periodicity_returns: str, method:str) -> pd.DataFrame:
        """
        Méthode permettant de calculer un ensemble de revenus à partir d'un jeu de données contenant des prix
        :param asset_prices: le dataframe contenant les prix des actifs pour lesquels on souhaite calculer le rendement
        :param periodicity_returns: la périodicité des données (monthly / quarterly / yearly
        :param method: la méthode pour le calcul des rendements (discret vs continu)
        :return: un dataframe contenant les rendements pour tous les actifs
        """

        # Test pour vérifier la périodicité des données
        if periodicity_returns not in ["monthly", "quarterly", "yearly"]:
            raise Exception(f"La périodicité {periodicity_returns} n'est pas implémentée. Veuillez modifier ce paramètre" )

        # Test pour vérifier la méthode de calcul des rendements
        if method not in ["discret","continu"]:
            raise Exception(f"La méthode {method} n'est pas implémentée pour le calcul des rendements. Veuillez modifier ce paramètre")

        # Récupération des données de fin de période
        period_resampling: dict = {
            "monthly": 'ME',
            "quarterly": 'QE',
            "yearly": 'YE'
        }
        resampled_data: pd.DataFrame = asset_prices.resample(period_resampling[periodicity_returns]).last()

        # Les valeurs manquantes sont conservés pour prendre en compte des entrées / sorties de titres de l'indice
        returns: pd.DataFrame = asset_prices.pct_change() if (method == 'discret') else np.log(asset_prices).diff()

        # Retraitement des rendements
        returns.replace(-1, np.nan, inplace=True)
        returns.replace(np.inf, np.nan, inplace=True)

        # La première ligne n'est pas conservée (pas de rendements)
        return returns.iloc[1:returns.shape[0], ]
