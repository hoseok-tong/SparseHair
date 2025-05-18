import json
import numpy as np
import vedo
import pyqtgraph as pg
from PyQt5 import QtGui


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class Skeleton:
    def __init__(self, config_path, color=None, init_pose=None):
        self.config = load_config(config_path)
        self.skeleton = {'bones': [], 'joints': []}
        self.initialize(init_pose, color)
    
    def initialize(self, init_pose, color=None):
        if init_pose is None:
            num_joints = len(self.config['joint_names'])
            init_pose = np.zeros([num_joints, 3])
        
        for pair in self.config['bone_indices']:
            is_right = ['Right' in self.config['joint_names'][jidx] for jidx in pair]
            if color is None:
                bone_lr = 0 if any(is_right) else 1
                bone_color = [val / 255. for val in self.config['limb_colors'][bone_lr]]
            else:
                bone_color = [val / 255. for val in color]
            bone = vedo.Line(init_pose[pair], lw=self.config['bone_radius'], c=bone_color)
            self.skeleton['bones'].append(bone)
        
        # Add joint actors to joints list
        if color is None:
            joint_color = [val / 255. for val in self.config['joint_color']]
        else:
            joint_color = [val / 300. for val in color]
        for coords in init_pose:
            joint = vedo.Sphere(coords, r=self.config['joint_radius'], c=joint_color)
            self.skeleton['joints'].append(joint)
    
    def add(self, vp, at=None):
        for key in self.skeleton:
            vp.add(self.skeleton[key], at=at)
    
    def remove(self, vp, at=None):
        for key in self.skeleton:
            vp.remove(self.skeleton[key], at=at)
    
    def update(self, pose):
        # world_pose = pose @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for bone, pair in zip(self.skeleton['bones'], self.config['bone_indices']):
            bone.points(pose[pair])
        for joint, coords in zip(self.skeleton['joints'], pose):
            joint.pos(*coords)


class Mesh:
    def __init__(self, vertices, faces, alpha=0.7):
        self.mesh = None
        self.initialize(vertices, faces, alpha)
    
    def initialize(self, vertices, faces, alpha):
        self.mesh = vedo.Mesh([vertices, faces], c=[225, 225, 225], alpha=alpha)
        # self.mesh.subdivide(n=4)
        # self.mesh.compute_normals()
        # self.mesh.lighting(ambient=0.6, diffuse=0.5)
    
    def add(self, vp, at=None):
        vp.add(self.mesh, at=at)

    def remove(self, vp, at=None):
        vp.remove(self.mesh, at=at)
    
    def update(self, vertices):
        self.mesh.points(vertices)


class Strand:
    def __init__(self, vertices, alpha=1.0):
        self.strand = None
        self.initialize(vertices, alpha)
    
    def initialize(self, vertices, alpha):
        self.strand = vedo.Points(vertices, c=[225, 225, 225], alpha=alpha)
    
    def add(self, vp, at=None):
        vp.add(self.strand, at=at)

    def remove(self, vp, at=None):
        vp.remove(self.strand, at=at)
    
    def update(self, vertices):
        self.strand.points(vertices)


class ROM:
    def __init__(self, endpoints, roms, joint_idx):
        # TODO: Only use roms
        self.rom = []
        self.initialize(endpoints, roms, joint_idx)
    
    def initialize(self, endpoints, roms, joint_idx):
        fp_points = endpoints[joint_idx, 0].transpose(1, 0)
        sp_points = endpoints[joint_idx, 1].transpose(1, 0)
        sr_points = endpoints[joint_idx, 2].transpose(1, 0)
        
        # Add Swing ROM
        if joint_idx in [idx - 1 for idx in [4, 5, 10, 11, 18, 19]]:
            min_angle, max_angle = roms.tolist()[joint_idx][1]
            self.rom.append(self.draw_arc(min_angle, max_angle))
        elif joint_idx in [idx - 1 for idx in [22, 23]]:
            min_angle, max_angle = roms.tolist()[joint_idx][0]
            self.rom.append(self.draw_arc(min_angle, max_angle, frontal=True))
        else:
            normals = []
            for i in range(4):
                fp_point = fp_points[i % 2]
                sp_point = sp_points[i // 2]
                if fp_point[1] == -1 or sp_point[1] == -1:
                    continue
                
                x_normal = (1, 0, 0) if fp_point[0] >= 0 else (-1, 0, 0)
                z_normal = (0, 0, 1) if sp_point[2] >= 0 else (0, 0, -1)
                
                normal = np.cross(fp_point, sp_point) if fp_point[0] * sp_point[2] >= 0 \
                    else np.cross(sp_point, fp_point)
                if sr_points[0, 1] > 0:
                    normal *= -1
                normals.append((x_normal, z_normal, normal / np.linalg.norm(normal + 1e-8)))
            # Create a sphere
            spheres = []
            for x_normal, z_normal, normal in normals:
                sphere = vedo.Sphere().alpha(0.6)
                sphere.cut_with_plane(normal=x_normal)
                sphere.cut_with_plane(normal=z_normal)
                sphere.cut_with_plane(normal=normal.tolist())
                spheres.append(sphere)
            self.rom.append(spheres)
        
        # Add Twist ROM
        min_angle, max_angle = roms.tolist()[joint_idx][2]
        self.rom.append(self.draw_arc(min_angle, max_angle, horizontal=True))
    
    def add(self, vp):
        for at, rom in enumerate(self.rom):
            vp.add(rom, at=at)

    def remove(self, vp):
        for at, rom in enumerate(self.rom):
            vp.remove(rom, at=at)
    
    @staticmethod
    def draw_arc(min_angle, max_angle, steps=100, horizontal=False, frontal=False):
        angles = np.linspace(min_angle, max_angle, num=steps)
        if horizontal:
            points = np.stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)], axis=-1)
        elif frontal:
            points = np.stack([np.sin(angles), -np.cos(angles), np.zeros_like(angles)], axis=-1)
        else:
            points = np.stack([np.zeros_like(angles), -np.cos(angles), np.sin(angles)], axis=-1)
        return vedo.Lines(points[1:], points[:-1], c='r')


class ROMPlotter:
    def __init__(self, shape=(1, 2), qt_widget=None):
        self.shape = shape
        self.vp = vedo.Plotter(shape=self.shape, qt_widget=qt_widget, bg=(30, 30, 30))
        # Set plotter viewpoint
        dist, azim, elev = 5, np.pi / 6, np.pi / 6
        camera_position = np.array(
            [dist * np.sin(azim) * np.cos(elev),
             dist * np.cos(azim) * np.cos(elev),
             dist * np.sin(elev)])
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.camera.SetPosition(camera_position)
            self.vp.add(vedo.Sphere(r=0.01, c='k'), at=at)
            self.vp.add(vedo.Text2D('O', pos=(0.5, 0.5)), at=at)
            self.vp.add(vedo.Sphere(res=360).c((225, 225, 225), alpha=0.05), at=at)
        self.roms = []
        self.points = []
    
    def init_roms(self, endpoints, roms):
        for idx, _ in enumerate(roms):
            self.roms.append(ROM(endpoints, roms, idx))
    
    def update_roms(self, endpoints, roms):
        self.roms = []
        for idx, _ in enumerate(roms):
            self.roms.append(ROM(endpoints, roms, idx))
    
    def init_points(self):
        for at in range(self.shape[0] * self.shape[1]):
            point_unrefined = vedo.Point(r=20, c='r')
            point_refined = vedo.Point(r=20, c='g')
            self.vp.add(point_unrefined, at=at)
            self.vp.add(point_refined, at=at)
            self.points.append([point_unrefined, point_refined])
    
    def add_rom(self, joint_idx):
        # for rom in self.roms:
        #     rom.add(self.vp)
        self.roms[joint_idx].add(self.vp)
        self.vp.render()
    
    def add_points(self, swing_point, twist_point):
        self.vp.add(vedo.Point(swing_point, r=20, c='g'), at=0)
        self.vp.add(vedo.Point(twist_point, r=20, c='g'), at=1)
        self.vp.render()
    
    def update(self, unrefined_point, refined_point, idx):
        self.points[idx][0].pos(*unrefined_point)
        self.points[idx][1].pos(*refined_point)
        self.vp.render()
    
    def remove_rom(self, joint_idx):
        # for rom in self.roms:
        #     rom.remove(self.vp)
        self.roms[joint_idx].remove(self.vp)
        self.vp.render()
    
    def show(self, **kwargs):
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.show(at=at, **kwargs)


class PosePlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.skeleton = None
    
    def init_skeleton(self, config_path, init_pose=None):
        self.skeleton = Skeleton(config_path, init_pose)
    
    def add_skeleton(self):
        self.skeleton.add(self.vp)
        self.vp.render()
    
    def remove_skeleton(self):
        self.skeleton.remove(self.vp)
        self.vp.render()
    
    def show(self):
        self.vp.show()
    
    def update(self, pose):
        self.skeleton.update(pose)
        self.vp.render()


class MeshPlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.mesh = None
    
    def init_mesh(self, vertices, faces, alpha=0.7):
        self.mesh = Mesh(vertices, faces, alpha)
    
    def add_mesh(self):
        self.mesh.add(self.vp)
        self.vp.render(resetcam=1)
    
    def remove_mesh(self):
        self.mesh.remove(self.vp)
        self.vp.render()

    def show(self):
        self.vp.show()
    
    def update(self, vertices, restcam=False):
        self.mesh.update(vertices)
        self.vp.render(restcam)


class PoseMeshPlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.mesh = None
    
    def init_mesh(self, vertices, faces):
        self.mesh = Mesh(vertices, faces)
    
    def init_skeleton(self, config_path, init_pose=None):
        self.skeleton = Skeleton(config_path, init_pose)
    
    def add_mesh(self):
        self.mesh.add(self.vp)
        self.vp.render(resetcam=1)
    
    def remove_mesh(self):
        self.mesh.remove(self.vp)
        self.vp.render()
    
    def add_skeleton(self):
        self.skeleton.add(self.vp)
        self.vp.render()

    def remove_skeleton(self):
        self.skeleton.remove(self.vp)
        self.vp.render()

    def show(self):
        self.vp.show()
    
    def update(self, vertices, pose, restcam=False):
        self.mesh.update(vertices)
        self.skeleton.update(pose)
        self.vp.render(restcam)


class MultiPosePlotter:
    def __init__(self, shape=(1, 1), qt_widget=None):
        self.shape = shape
        self.vp = vedo.Plotter(shape=shape, qt_widget=qt_widget, bg=(30, 30, 30))
        self.skeleton_lists = [[] for _ in range(self.shape[0] * self.shape[1])]
    
    def init_skeleton(self, at, config_path, color=None, init_pose=None):
        self.skeleton_lists[at].append(Skeleton(config_path, color, init_pose))
    
    def add_skeletons(self):
        for at, skeleton_list in enumerate(self.skeleton_lists):
            for skeleton in skeleton_list:
                skeleton.add(self.vp, at=at)
        self.vp.render(resetcam=True)
    
    def remove_skeletons(self):
        for at, skeleton_list in enumerate(self.skeleton_lists):
            for skeleton in skeleton_list:
                skeleton.remove(self.vp, at=at)
        self.vp.render()
    
    def show(self, **kwargs):
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.show(at=at, **kwargs)


class MultiMeshPlotter:
    def __init__(self, shape=(1, 1), qt_widget=None):
        self.shape = shape
        self.vp = vedo.Plotter(shape=shape, qt_widget=qt_widget, bg=(30, 30, 30))
        self.mesh_lists = [[] for _ in range(self.shape[0] * self.shape[1])]
    
    def init_mesh(self, at, vertices, faces, alpha=0.7):
        self.mesh_lists[at].append(Mesh(vertices, faces, alpha))
    
    def add_meshes(self):
        for at, mesh_list in enumerate(self.mesh_lists):
            for mesh in mesh_list:
                mesh.add(self.vp, at=at)
        self.vp.render(resetcam=True)
    
    def remove_meshes(self):
        for at, mesh_list in enumerate(self.mesh_lists):
            for mesh in mesh_list:
                mesh.remove(self.vp, at=at)
        self.vp.render()
    
    def show(self, **kwargs):
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.show(at=at, **kwargs)
    
    def update(self, at, vertices, restcam=False):
        for mesh in self.mesh_lists[at]:
            mesh.update(vertices)
        self.vp.render(restcam)


class MultiPoseMeshPlotter:
    def __init__(self, shape=(1, 1), qt_widget=None):
        self.shape = shape
        self.vp = vedo.Plotter(shape=shape, qt_widget=qt_widget, bg=(30, 30, 30))
        self.meshes = {}
        self.skeletons = {}
        # for at in range(self.shape[0] * self.shape[1]):
        #     light1 = vedo.Light([5, -2, 0], c='w', intensity=0.1)
        #     light2 = vedo.Light([0, -2, 5], c='w', intensity=0.1)
        #     light3 = vedo.Light([-5, -2, 0], c='w', intensity=0.1)
        #     light4 = vedo.Light([0, -2, -5], c='w', intensity=0.1)
        #     self.vp.add([light1, light2, light3, light4], at=at)
    
    def init_strand(self, key, at, vertices, faces, alpha=0.7):
        self.meshes[key] = (Mesh(vertices, faces, alpha), at)
    
    def init_skeleton(self, key, at, config_path, color=None, init_pose=None):
        self.skeletons[key] = (Skeleton(config_path, color, init_pose), at)
    
    def add_meshes(self):
        for key in self.meshes:
            mesh, at = self.meshes[key]
            mesh.add(self.vp, at=at)
        self.vp.render(resetcam=True)
    
    def add_skeletons(self):
        for key in self.skeletons:
            skeleton, at = self.skeletons[key]
            skeleton.add(self.vp, at=at)
        self.vp.render(resetcam=True)
    
    def remove_meshes(self):
        for key in self.meshes:
            mesh, at = self.meshes[key]
            mesh.remove(self.vp, at=at)
        self.vp.render()
    
    def remove_skeletons(self):
        for key in self.skeletons:
            skeleton, at = self.skeletons[key]
            skeleton.remove(self.vp, at=at)
        self.vp.render()
    
    def replace_mesh(self, key, at, vertices, faces, alpha=0.7):
        self.meshes[key][0].remove(self.vp, at=at)
        self.meshes[key] = (Mesh(vertices, faces, alpha), at)
        self.meshes[key][0].add(self.vp, at=at)
        self.vp.render()
    
    def show(self, **kwargs):
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.show(at=at, **kwargs)
    
    def update(self, vertices, poses, restcam=False):
        for key in self.meshes:
            if key == 'bodyscan':
                continue
            mesh, _ = self.meshes[key]
            mesh.update(vertices[key])
        for key in self.skeletons:
            skeleton, _ = self.skeletons[key]
            skeleton.update(poses[key])
        self.vp.render(restcam)


class StrandPlotter:
    def __init__(self, shape=(1, 1), qt_widget=None):
        self.shape = shape
        self.vp = vedo.Plotter(shape=shape, qt_widget=qt_widget, bg=(30, 30, 30))
        self.vp.add(vedo.Point([0.0, 0.0, 0.0], c='r'))
        self.vp.add(vedo.Arrow([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], c='r'))
        self.vp.add(vedo.Arrow([0.0, 0.0, 0.0], [0.0, 0.1, 0.0], c='g'))
        self.vp.add(vedo.Arrow([0.0, 0.0, 0.0], [0.0, 0.0, 0.1], c='b'))
        self.strand = {}
    
    def init_strand(self, key, at, vertices, alpha=0.7):
        self.strand[key] = (Strand(vertices, alpha), at)
    
    def add_strand(self):
        for key in self.strand:
            strand, at = self.strand[key]
            strand.add(self.vp, at=at)
        self.vp.render(resetcam=True)
    
    def remove_strand(self):
        for key in self.strand:
            strand, at = self.strand[key]
            strand.remove(self.vp, at=at)
        self.vp.render()
    
    def replace_strand(self, key, at, vertices, alpha=0.7):
        self.strand[key][0].remove(self.vp, at=at)
        self.strand[key] = (Strand(vertices, alpha), at)
        self.strand[key][0].add(self.vp, at=at)
        self.vp.render()
    
    def show(self, **kwargs):
        for at in range(self.shape[0] * self.shape[1]):
            self.vp.show(at=at, **kwargs)
    
    def update(self, vertices, restcam=False):
        for key in self.strand:
            strand, _ = self.strand[key]
            strand.update(vertices.squeeze())
        self.vp.render(restcam)

class MultiGraphPlotter(pg.GraphicsLayoutWidget):
    def __init__(self, plot_configs, num_rows=1, parent=None):
        super().__init__()
        # TODO: Add arguments for background color and font
        self.setBackground((30, 30, 30))
        self.font = QtGui.QFont('Times New Roman')
        self.widgets = {}
        for idx, (widget_key, plot_config) in enumerate(plot_configs.items()):
            title, xlabel, ylabel, logy = plot_config
            row, col = idx // (num_rows + 1), idx % (num_rows + 1)
            self.widgets[widget_key] = \
                self.create_plot(row, col, title, xlabel, ylabel, logy)
        # Initialize plot and data holders
        self.plots = {}
        self.texts = {}
        self.xdata = {}
        self.ydata = {}
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        self.color_index = 0
    
    def create_plot(self, row, col, title, xlabel, ylabel, logy):
        plot = self.addPlot(row=row, col=col)
        plot.setTitle(title)
        plot.titleLabel.item.setFont(self.font)
        plot.setLabel('bottom', xlabel)
        plot.setLabel('left')
        plot.getAxis('bottom').label.setFont(self.font)
        # plot.getAxis('left').label.setFont(self.font)
        plot.getAxis('bottom').setStyle(tickFont=self.font)
        plot.getAxis('left').setStyle(tickFont=self.font)
        plot.setLogMode(y=logy)
        return plot
    
    def clear_plots(self):
        for widget_key in self.widgets:
            if self.texts:
                for plot_key in self.texts[widget_key]:
                    self.widgets[widget_key].scene().removeItem(self.texts[widget_key][plot_key])
            self.widgets[widget_key].clear()
            self.plots = {}
            self.texts = {}
            self.xdata = {}
            self.ydata = {}
    
    def _prepare_plot(self, data_config, widget_key, plot_key):
        color, item_pos, parent_pos = data_config
        self.plots[widget_key][plot_key] = self.widgets[widget_key].plot()
        if color is not None:
            self.plots[widget_key][plot_key].setPen(color=color, width=2)
        else:
            color = self.colors[self.color_index % len(self.colors)]
            self.plots[widget_key][plot_key].setPen(color=color, width=2)
            self.color_index += 1
        self.texts[widget_key][plot_key] = pg.LabelItem('{}: '.format(plot_key), color=color)
        self.texts[widget_key][plot_key].setParentItem(self.widgets[widget_key].graphicsItem())
        self.texts[widget_key][plot_key].anchor(itemPos=item_pos, parentPos=parent_pos)
        self.xdata[widget_key][plot_key] = []
        self.ydata[widget_key][plot_key] = []
    
    def prepare_plots(self, data_configs, xranges):
        self.clear_plots()
        for widget_key in data_configs:
            self.texts[widget_key] = {}
            self.plots[widget_key] = {}
            self.xdata[widget_key] = {}
            self.ydata[widget_key] = {}
            self.widgets[widget_key].getViewBox().setXRange(*xranges[widget_key], padding=0.1)
            for plot_key, data_config in data_configs[widget_key].items():
                self._prepare_plot(data_config, widget_key, plot_key)

    def _update_plot(self, plot_data, widget_key, plot_key):
        xdata, ydata = plot_data[widget_key][plot_key]
        self.xdata[widget_key][plot_key].append(xdata)
        self.ydata[widget_key][plot_key].append(ydata)
        self.plots[widget_key][plot_key].setData(self.xdata[widget_key][plot_key],
                                                 self.ydata[widget_key][plot_key])
        self.texts[widget_key][plot_key].setText('{}: {:.4f}'.format(plot_key, ydata))
    
    def update_plots(self, plot_data):
        for widget_key in plot_data:
            for plot_key in plot_data[widget_key]:
                self._update_plot(plot_data, widget_key, plot_key)
