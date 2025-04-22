import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualisation:
    """
    Cette classe sert exclusivement à centraliser les éléments d'affichages graphiques.
    """

    def __init__(self):
        pass

    @staticmethod
    def heatmap_correlation(returns: pd.DataFrame, annot: bool = True, cmap: str = 'coolwarm',
                            vmin: float = -1, vmax: float = 1,
                            title: str = "Matrice de corrélation") -> None:
        """
        Affiche la matrice de corrélation des rendements
        """
        correlations = returns.corr()
        plt.figure(figsize=(12, 6))

        heatmap = sns.heatmap(
            correlations,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cumulative_returns(returns: pd.DataFrame,
                                title: str = "Rendements cumulés") -> None:
        """
        Calcule et affiche les rendements cumulés pour chaque actif.
        """
        cumulative_returns: pd.DataFrame = (1 + returns).cumprod()
        cumulative_returns.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Valeur")
        plt.legend(title="Actifs", frameon=True, fontsize=8)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_weights(weights: pd.DataFrame, title="Évolution des poids du portefeuille",
                     invert_xaxis: bool = False) -> plt.Axes:
        if weights.empty:
            raise ValueError("Les positions du portefeuille n'ont pas été calculées. Exécutez d'abord run_backtest().")

        # Création du graphique à aire empilée
        ax = weights.plot.area(figsize=(12, 6), alpha=0.7, stacked=True)

        # Si on veut inverser l'axe des abscisses ==> grille de désensibilisation
        if invert_xaxis:
            ax.invert_xaxis()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Allocation (%)', fontsize=12)
        ax.legend(title='Actifs', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        for y in np.arange(0.2, 1.0, 0.2):
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.tight_layout()
        return ax

