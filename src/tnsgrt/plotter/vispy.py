from tnsgrt.structure import Structure
from .plotter import Plotter

from vispy import scene as vispyscene


class VisPyPlotter(Plotter):
    """
    VisPy based structure plotter

    :param camera: dict with camera settings
    :param scene: dict with vispy scene settings
    """

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

    def __init__(self, camera=None, scene=None):

        scene_kwargs = {
            'keys': 'interactive',
            'size': (600, 600)
        }
        if scene:
            scene_kwargs.update(scene)

        camera_kwargs = {
            'fov': 0,
            'scale_factor': 1
        }
        if camera:
            camera_kwargs.update(camera if camera else {})

        # Create canvas and view
        self.canvas = vispyscene.SceneCanvas(**scene_kwargs)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = vispyscene.cameras.ArcballCamera(**camera_kwargs)

    def get_canvas(self):
        """
        :return: vispy canvas 
        """
        return self.canvas

    def plot(self, *s: Structure, **kwargs):

        # loop if more than one is given
        if len(s) != 1:
            for si in s:
                self.plot(si, **kwargs)
            return
        else:
            s = s[0]

        # process options
        defaults = VisPyPlotter.defaults.copy()
        defaults.update(kwargs)

        # plot nodes
        nodes = s.nodes
        if defaults['plot_nodes']:
            # Create and show visual
            vis = vispyscene.visuals.Markers(
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
        members = s.members
        if defaults['plot_members']:
            for j in range(s.get_number_of_members()):
                if s.member_properties.loc[j, 'visible']:
                    if s.has_member_tag(j, 'string'):
                        # plot strings as lines
                        kwargs = s.get_member_properties(j, 'facecolor',
                                                         'linewidth').to_dict()
                        kwargs['color'] = kwargs['facecolor']
                        del kwargs['facecolor']
                        kwargs['width'] = kwargs['linewidth']
                        del kwargs['linewidth']
                        line = nodes[:, [members[0, j], members[1, j]]].transpose()
                        vis = vispyscene.visuals.Line(line, **kwargs)
                        vis.parent = self.view.scene
                    else:
                        # plot others as solid elements
                        kwargs = s.get_member_properties(j, 'facecolor',
                                                         'radius').to_dict()
                        kwargs['color'] = kwargs['facecolor']
                        del kwargs['facecolor']
                        line = nodes[:, [members[0, j], members[1, j]]].transpose()
                        vis = vispyscene.visuals.Tube(line, **kwargs)
                        vis.parent = self.view.scene
