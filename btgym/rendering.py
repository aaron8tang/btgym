###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import logging
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _matplotlib = True

except:
    _matplotlib = False


class BTgymRendering():
    """
    Executes rendering of BTgym Environment.
    """
    # Output elements:
    state = None  # values to plot,  type=np.array.
    title = ''  # figure title, type=str.
    box_text = ''  # inline text block, type=str.

    # Plotting controls, can be passed as kwargs:
    plot_type = 'plot'
    figsize = (9, 7)
    plot_style = 'seaborn'
    color_map = 'PRGn'
    xlabel = 'Relative timestep'
    ylabel = 'Value'
    title_template = 'State observation min: {:.4f}, max: {:.4f}'
    box_params = dict(
        fontsize=12,
        fontweight='bold',
        color='w',
        bbox={'facecolor': 'k', 'alpha': 0.3, 'pad': 3},
    )

    def __init__(self, **kwargs):
        self.log = logging.getLogger('Plotter')
        logging.getLogger().setLevel(logging.WARNING)

        self.matplotlib = _matplotlib
        if not self.matplotlib:
            self.log.warning('Matplotlib not loaded. Plotting features disabled.')

        # Update, if any:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_string(self, dictionary):
        """
        Converts given dictionary to more-or-less good looking string.
        """
        text = ''
        for k, v in dictionary.items():
            if type(v) in [float]:
                v = '{:.4f}'.format(v)
            text += '{}: {}\n'.format(k, v)
        return text

    def parse_response(self, env_response):
        """
        Converts environment response to plotting attributes:
        state, title, text.
        """
        try:
            (state, reward, done, info) = env_response
            try:
                # What we plot:
                self.state = np.asarray(state)
                assert len(self.state.shape) == 2

            except:
                raise NotImplementedError('Only 2D observation state can be plotted')

        except:
            raise AssertionError('Cant render given environment response')

        # Figure out how to deal with info field:
        try:
            assert type(info[-1]) == dict
            info_dict = info[-1]

        except:
            try:
                assert type(info) == dict
                info_dict = info

            except:
                try:
                    info_dict = {'info': str(dict)}

                except:
                    info_dict = {}

        # Add records:
        info_dict.update(reward=reward, is_done=done)

        # Convert:
        self.box_text = self.to_string(info_dict)

        # Make title:
        self.title = self.title_template.format(self.state.min(), self.state.max())

    def render_state(self, env_response):
        """
        Decides how to get job done.
        """
        self.parse_response(env_response)

        if self.matplotlib:
            if self.plot_type == 'plot':
                self.as_plot()

            else:
                self.as_image()

        else:
            self.as_text()

    def as_text(self):
        """
        Ascetic: just text output.
        """
        raise NotImplementedError('For python''s sake, install that Matplotlib!')

    def as_plot(self):
        """
        Visualises environment state as 2d line plot.
        """

        plt.figure(figsize=self.figsize)
        #plt.figure(figsize=(6,6))
        plt.style.use(self.plot_style)
        plt.title(self.title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(self.state.shape[-1] - 1, 0, int(self.state.shape[-1]), dtype=int)
        plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in plt.xticks()[1][::5]:
           tick.set_visible(True)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(True)

        # Add Info box:
        plt.text(1, self.state.T.min(), self.box_text, **self.box_params)

        plt.plot(self.state.T)
        plt.show()

    def as_image(self):
        """
        Visualises environment state as image.
        """
        plt.figure(figsize=self.figsize)
        #plt.style.use(self.plot_style)
        plt.title(self.title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(self.state.shape[-1] - 1, 0, int(self.state.shape[-1]), dtype=int)
        plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in plt.xticks()[1][::5]:
            tick.set_visible(True)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(True)

        # Add Info box:
        plt.text(1, self.state.T.min(), self.box_text, **self.box_params)

        plt.imshow(self.state, aspect='auto', cmap=self.color_map)
        plt.show()