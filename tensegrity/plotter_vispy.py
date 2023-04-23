from .plotter import Plotter
from .structure import Structure

from vispy import app, scene


class VisPyPlotter(Plotter):

    defaults = {
        'plot_nodes': True,
        'plot_members': True,
        'node_marker': 'o',
        'node_markersize': .05,
        'node_linewidth': 2,
        'node_linestyle': 'none',
        'node_facecolor': (1, 1, 1),
        'node_edgecolor': (1, 1, 1)
    }

    def __init__(self, s: Structure):
        super().__init__(s)

        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.ArcballCamera(fov=0)
        self.view.camera.scale_factor = 10

    def plot(self, **kwargs):

        # process options
        defaults = VisPyPlotter.defaults.copy()
        defaults.update(kwargs)

        # plot nodes
        nodes = self.s.nodes
        if defaults['plot_nodes']:
            # Create and show visual
            vis = scene.visuals.Markers(
                pos=nodes.transpose(),
                size=defaults['node_markersize'],
                antialias=0,
                face_color=defaults['node_facecolor'],
                edge_color=defaults['node_edgecolor'],
                edge_width=0,
                scaling=True,
                spherical=True,
            )
            vis.parent = self.view.scene

        # plot members
        members = self.s.members
        if defaults['plot_members']:
            for j in range(self.s.get_number_of_members()):
                if self.s.has_member_tag(j, 'string'):
                    # plot strings as lines
                    kwargs = self.s.get_member_properties(j, ['facecolor', 'linewidth']).to_dict()
                    kwargs['color'] = kwargs['facecolor']
                    del kwargs['facecolor']
                    kwargs['width'] = kwargs['linewidth']
                    del kwargs['linewidth']
                    line = nodes[:, [members[0, j], members[1, j]]].transpose()
                    vis = scene.visuals.Line(line, **kwargs)
                    vis.parent = self.view.scene
                else:
                    # plot others as solid elements
                    kwargs = self.s.get_member_properties(j, ['facecolor']).to_dict()
                    kwargs['color'] = kwargs['facecolor']
                    del kwargs['facecolor']
                    line = nodes[:, [members[0, j], members[1, j]]].transpose()
                    vis = scene.visuals.Tube(line, radius=.025, **kwargs)
                    vis.parent = self.view.scene

        # return self.fig, self.ax
