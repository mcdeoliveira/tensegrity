import numpy as np
from .structure import Structure


class Snelson(Structure):

    @staticmethod
    def __init__(self, height: float = 1, radius: float = 1, alpha: float = np.pi/6, label: str = None):
        # height
        # radius
        # alpha = twist angle (in degrees)

        lambdaBar = 1
        delta = 0.5
        zeta = 0
        delH = 0
        doSimple = True

        # convert alpha to radians
        # alpha = alpha * np.pi / 180
        beta = 2 * np.pi / 3

        n1 = np.array([radius, 0])
        rotation_beta = Snelson.rotation_2d(beta)
        rotation_alpha = Snelson.rotation_2d(alpha)
        bar_nodes = np.vstack((n1, np.dot(rotation_beta, n1), np.dot(rotation_beta @ rotation_beta, n1),
                              np.dot(rotation_beta @ rotation_alpha, n1),
                              np.dot(rotation_beta @ (rotation_beta @ rotation_alpha), n1),
                              np.dot(rotation_alpha, n1))).transpose()
        btm_rot = (1 - zeta) * (delta * np.eye(2) + (1 - delta) * rotation_beta) + zeta * rotation_beta @ rotation_beta
        top_rot = (1 - zeta) * (delta * np.eye(2) + (1 - delta) * rotation_beta @ rotation_beta) + zeta * rotation_beta
        string_nodes = np.hstack((np.dot(btm_rot, bar_nodes[:, :3]), np.dot(top_rot, bar_nodes[:, 4:6]),
                                 np.dot(top_rot, bar_nodes[:, [3]])))
        z_pos = np.hstack((-(height / 2) * np.ones((1, 3)), (height / 2) * np.ones((1, 3))))
        z_pos = np.hstack((z_pos, (1 - delH) * z_pos))
        nodes = np.vstack((np.hstack((bar_nodes, string_nodes)), z_pos))

        # member indices are given starting at 1 below:
        members = np.array([
            [1, 7], [2, 8], [3, 9],        # btm long(1: 3)
            [1, 9], [2, 7], [3, 8],        # btm short(4: 6)
            [4, 12], [5, 10], [6, 11],     # top long(7: 9)
            [4, 10], [5, 11], [6, 12],     # top short(8: 12)
            [1, 6], [2, 4], [3, 5],        # vert(13: 15)
            [1, 5], [2, 6], [3, 4],        # diag(16: 18)
            [7, 8], [8, 9], [7, 9],        # btm  overlap(19: 21)
            [10, 11], [11, 12], [10, 12],  # top overlap(22: 24)
            [1, 11], [2, 12], [3, 10],     # reach1(25: 27)
            [7, 11], [8, 12], [9, 10],     # reach2(28: 30)
            [7, 12], [8, 10], [9, 11],     # reach3(31: 33)
            [1, 4], [2, 5], [3, 6]         # bars(34: 36)
        ]).transpose() - 1
        number_of_strings = 33

        # string tags
        member_tags = {
            'bottom': np.hstack((np.arange(6), 18 + np.arange(3))),      # [1:6 19:21]
            'top': np.hstack((6 + np.arange(6), 21 + np.arange(3))),     # [7:12, 22:24]
            'long': np.hstack((np.arange(3), 6 + np.arange(3))),         # [1: 3, 7: 9];
            'short': np.hstack((3 + np.arange(3), 9 + np.arange(3))),    # [4: 6, 10: 12];
            'vertical': 12 + np.arange(3),                               # [13: 15];
            'diagonal': 15 + np.arange(3),                               # [16: 18];
            'thick': np.hstack((12 + np.arange(6), 24 + np.arange(9))),  # [13: 18, 25: 33];
            'reach': 24 + np.arange(9),                                  # [25: 33];
            'string2string': 27 + np.arange(6),                          # [28: 33];
            'overlap': 18 + np.arange(6)                                 # [19: 24]
        }

        # call super
        super().__init__(nodes=nodes, members=members, number_of_strings=number_of_strings,
                         member_tags=member_tags, label=label)

        # self.equilibrium()

        # barDiam = 4e-3
        # stringDiam = 2e-3
        # modulus = 230e9  # Carbon        Fiber
        # yld = 2e9
        # density = 1850

        # s = addSubStructure(s, label);
        # s = addLocalNodes(s, 1, nodeIndex);
        # s = addMembers(s, 1, memberIndex, numberOfStrings);
        # s = computeEquil(s, lambdaBar);
        #
        #
        # for index = 1:size(tags, 1)
        # ind = makeIndex(s, 1, tags
        # {index, 2});
        # s = add(s, ind, tags
        # {index, 1});
        # end
        #
        # sp.height = h;
        # sp.radius = r;
        # sp.alpha = alpha * 180 / pi;
        # sp.delta = delta;
        # sp.zeta = zeta;
        # sp.delH = delH;
        #
        # if (doSimple)
        #     sp.delta = [];
        #     sp.zeta = [];
        #     s = removeSlackMembers(s);
        #     s = removeDependentNodes(s);
        #     s = removeUnusedNodes(s);
        # end
        #
        # s = materialProperties(s, barDiam, stringDiam, modulus, modulus,
        # yield, yield, density, density );
        # s = updateMemberStiffness(s);
        # s = updateMemberMass(s);
        #
        # s =
        #
        # class(sp, 'snelson', s);
        #
        # property = {'color', {[1 0 0]}, 'top';
        # 'color', {[0 0 1]}, 'bottom';
        # 'color', {[1 1 0]}, 'thick';
        # 'width', 2, 'bar';
        # 'width', 1.5, 'string'};
        #
        # for index = 1:size(property, 1)
        # ind = makeIndex(s, 'all', property
        # {index, 3});
        # s = set(s, property
        # {index, 1}, ind, property
        # {index, 2} );
        # end

