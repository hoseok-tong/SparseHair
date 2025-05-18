# Reference: https://github.com/c-he/perm/blob/main/src/hair/io.py
import os
import yaml
import struct
import numpy as np
from types import SimpleNamespace

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

def load_hair(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1]
    assert ext == '.data', "only support loading hair data in .data format"

    with open(path, 'rb') as f:
        data = f.read()
    num_strands = struct.unpack('i', data[:4])[0]
    strands = []
    idx = 4
    for _ in range(num_strands):
        num_points = struct.unpack('i', data[idx:idx + 4])[0]
        points = struct.unpack('f' * num_points * 3, data[idx + 4:idx + 4 + 4 * num_points * 3])
        strands.append(list(points))
        idx = idx + 4 + 4 * num_points * 3
    strands = np.array(strands, dtype=np.float32).reshape((num_strands, -1, 3))

    return strands


def save_hair(path: str, data: np.ndarray, color: np.ndarray = None) -> None:
    ext = os.path.splitext(path)[1]
    assert ext in ['.data', '.obj'], "only support saving hair data in .data and .obj format"

    if ext == '.data':
        _save_hair_data(path, data)
    else:
        save_hair2pc(path, data, color)


def _save_hair_data(path: str, data: np.ndarray) -> None:
    num_strands, num_points = data.shape[:2]
    with open(path, 'wb') as f:
        f.write(struct.pack('i', num_strands))
        for i in range(num_strands):
            f.write(struct.pack('i', num_points))
            f.write(struct.pack('f' * num_points * 3, *data[i].flatten().tolist()))


def save_hair2pc(path, vert, color=None):
    """
    Save a 3D model in the OBJ format.
    Args:
        path (str): Path to save the OBJ file.
        vert (np.ndarray): Vertices of the model.
        color (np.ndarray, optional): Colors for the vertices.
    """
    with open(path, 'w') as f:
        if color is not None:
            for v, c in zip(vert, color):
                data = 'v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2])
                f.write(data)
        else:
            for v in vert:
                data = 'v %f %f %f\n' % (v[0], v[1], v[2])
                f.write(data)

                
def save_obj(path, vert, face=None, color=None):
    """
    vert N x 3
    color N x 3
    """
    with open(path, 'w') as f:
        if color is not None:
            for v, c in zip(vert, color):
                data = 'v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2])
                f.write(data)
        else:
            for v in vert:
                data = 'v %f %f %f\n' % (v[0], v[1], v[2])
                f.write(data)
        if face is not None:
            for f_ in face:
                data = 'f %d %d %d\n' % (f_[0], f_[1], f_[2])
                f.write(data)


def read_obj(filename):
    with open(filename) as file:
        content = file.readlines()
        
        v, vn, vt, tmp = [], [], [], []
        for i in range(len(content)):
            line = content[i]
            # if line[0] == '#' or 'o':
            #     continue

            if line[0:2] == 'vn':
                v_normal_info = line.replace(' \n', '').replace('vn ', '')
                vn.append(np.array(v_normal_info.split(' ')).astype(float))
            elif line[0:2] == 'v ':
                vertex_info = line.replace(' \n', '').replace('v ', '')
                v.append(np.array(vertex_info.split(' ')).astype(float))
            elif line[0:2] == 'vt':
                texture_info = line.replace(' \n', '').replace('vt ', '')
                vt.append(np.array(texture_info.split(' ')).astype(float))
            elif line[0:2] == 'f ':
                face_info = line.replace(' \n', '').replace('f ', '')
                for i in range(3):
                    tmp.append(np.array(face_info.split(' ')[i].split('/')).astype(int))

        tmp = np.array(tmp).reshape((-1, 3, 3))
        tmp2 = tmp[:, :, 0:2].reshape((-1, 2))
        v_vt_dict = dict()
        for i in tmp2:
            v_vt_dict[i[0]-1] = i[1]-1

    return v_vt_dict, np.array(v), np.array(vt)


def read_yaml(config_path):
    def dict_to_namespace(d):
        namespace = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            else:
                setattr(namespace, k, v)
        return namespace
        
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg_dict)

    return cfg
