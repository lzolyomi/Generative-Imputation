import pandas as pd 
import numpy as np

import plotly.graph_objects as go 
from plotly.subplots import make_subplots

class Evaluator:

    def __init__(self, csv_path) -> None:
        #NOTE: check if path exists 
        self.df = pd.read_csv(csv_path, index_col=0)
        self.gen_loss = self.df['Generator'].values 
        self.discr_loss = self.df['Discriminator'].values 
        self.sum = self.df['Sum'].values 
        self.filename = csv_path.split('/')[-1]

    def print_summary(self):
        """Prints a textual summary of all variables
        """
        cols = self.df.columns 
        prints = []
        for col in cols:
            values = self.df[col].values 
            argmin, best = self.get_best(values)
            infos = f'### {col.upper():<20}  $Lowest value {best:<8} $At epoch {argmin:<8}'
            prints.append(infos)

        for item in prints:
            print(item)

    def make_plot(self, col1:list, col2:list):
        """Returns plotly figure with dual y axis

        Args:
            col1 (list): List of columns to appear on LEFT y-axis
            col2 (list): List of columns to appear on RIGHT y-axis

        Returns:
            plotly.Figure: Plotly figure
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        assert type(col1) == list, 'col1 should be a list of column names(str)'
        assert type(col2) == list, 'col2 should be a list of column names(str)'
        # Add traces
        for c1 in col1:
            fig.add_trace(
                go.Scatter(x=self.df.index, y=self.df[c1], name=c1),
                secondary_y=False,
            )
        for c2 in col2:
            fig.add_trace(
                go.Scatter(x=self.df.index, y=self.df[c2], name=c2),
                secondary_y=True,
            )

        # Add figure title
        fig.update_layout(
            title_text="{} and {} for file {}".format(col1, col2, self.filename)
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Epochs")

        # Set y-axes titles
        fig.update_yaxes(title_text=str(col1), secondary_y=False, type='log')
        fig.update_yaxes(title_text=str(col2), secondary_y=True)
        self.fig = fig
        return fig


    @staticmethod
    def get_best(array):
        """Returns lowest element and its index

        Args:
            array (list/np array): array to use

        Returns:
            tuple: lowest element's index, lowest element
        """
        argmin = np.argmin(array)
        best = array[argmin]
        return argmin, best