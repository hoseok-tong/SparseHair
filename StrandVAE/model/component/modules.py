import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pytorch3d import _C
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes


_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


def find_value_from_uv_mappeing(root_uv, uv_map):
    # root_uv -> B, N, 2
    # uv_map -> C, H, W
    _, H, W = uv_map.shape
    root_uv = torch.stack([root_uv[:, :, 0] * (W - 1),
                           root_uv[:, :, 1] * (H - 1)
                           ], dim=-1)
    
    device = root_uv.device
    coord0 = torch.stack([torch.floor(root_uv[:, :, 0]), torch.floor(root_uv[:, :, 1])], dim=-1).type(
        torch.LongTensor).to(device)
    coord1 = torch.stack([torch.floor(root_uv[:, :, 0]), torch.ceil(root_uv[:, :, 1])], dim=-1).type(
        torch.LongTensor).to(device)
    coord2 = torch.stack([torch.ceil(root_uv[:, :, 0]), torch.floor(root_uv[:, :, 1])], dim=-1).type(
        torch.LongTensor).to(device)
    coord3 = torch.stack([torch.ceil(root_uv[:, :, 0]), torch.ceil(root_uv[:, :, 1])], dim=-1).type(
        torch.LongTensor).to(device)

    weight0 = F.pairwise_distance(root_uv, coord0)
    weight1 = F.pairwise_distance(root_uv, coord1)
    weight2 = F.pairwise_distance(root_uv, coord2)
    weight3 = F.pairwise_distance(root_uv, coord3)

    weight = weight0 + weight1 + weight2 + weight3
    weight0 = weight0 / weight
    weight1 = weight1 / weight
    weight2 = weight2 / weight
    weight3 = weight3 / weight

    zg0 = uv_map[:, coord0[:, :, 1], coord0[:, :, 0]].permute(1, 2, 0) * weight0[:, :, None]
    zg1 = uv_map[:, coord1[:, :, 1], coord1[:, :, 0]].permute(1, 2, 0) * weight1[:, :, None]
    zg2 = uv_map[:, coord2[:, :, 1], coord2[:, :, 0]].permute(1, 2, 0) * weight2[:, :, None]
    zg3 = uv_map[:, coord3[:, :, 1], coord3[:, :, 0]].permute(1, 2, 0) * weight3[:, :, None]
    
    return zg0 + zg1 + zg2 + zg3


def compute_barycentric_coords(P, A, B, C):
    """
    Compute barycentric coordinates for a point P with respect to triangle ABC.
    Args:
        P (torch.Tensor): Point to compute barycentric coordinates for.
        A (torch.Tensor): Vertex A of the triangle.
        B (torch.Tensor): Vertex B of the triangle.
        C (torch.Tensor): Vertex C of the triangle.
    Returns:
        torch.Tensor: Barycentric coordinates (alpha, beta, gamma).
    """
    # Compute the normal of the plane defined by A, B, C
    normal = torch.cross(B - A, C - A, dim=-1)
    normal = normal / torch.norm(normal)  # Normalize the normal vector
    
    # Project P onto the plane
    d = torch.dot(normal, A - P)
    P_projected = P + d * normal

    v0 = C - A
    v1 = B - A
    v2 = P_projected  - A

    # Compute dot products
    dot00 = torch.dot(v0, v0)
    dot01 = torch.dot(v0, v1)
    dot11 = torch.dot(v1, v1)
    dot20 = torch.dot(v2, v0)
    dot21 = torch.dot(v2, v1)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    alpha = (dot11 * dot20 - dot01 * dot21) / denom
    beta = (dot00 * dot21 - dot01 * dot20) / denom
    gamma = 1.0 - alpha - beta

    return gamma, beta, alpha   # (TODO) Why?


class ClosestPointUV2Mesh(object):
    def __init__(self, mesh_path, points_path, output_path):
        self.mesh_path = mesh_path
        self.points_path = points_path
        self.output_path = output_path

    def load_points(self):
        _, file_extension = os.path.splitext(self.points_path)
        if file_extension == ".obj":
            return load_obj(self.points_path)[0].reshape(-1,100,3)
        elif file_extension == ".npz":
            npzfile = np.load(self.points_path)
            # vertsS_tan = torch.from_numpy(npzfile['vertsS_tan'])
            # TBNs = torch.from_numpy(npzfile['TBNs'])
            # roots = torch.from_numpy(npzfile['roots'])
            # return vertsS_tan, TBNs, roots
            roots = torch.from_numpy(npzfile['roots'])
            return roots            
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
    def compute_uv_coordinates(self, v_r, tris, idxs, vt, f_h):
        """
        Compute UV coordinates for given vertices.
        Args:
            v_r (torch.Tensor): Vertices to compute UVs for.
            tris (torch.Tensor): Triangles of the mesh.
            idxs (torch.Tensor): Indices of closest triangles for each vertex in v_r.
            vt (np.ndarray): Vertex UVs.
            f_h (trimesh.base.Trimesh): Face data.
        Returns:
            np.ndarray: Computed UV coordinates.
        """
        uv_coords = []
        for i, point in enumerate(v_r):
            triangle = tris[idxs[i]]
            weights = compute_barycentric_coords(point, triangle[0], triangle[1], triangle[2])
            uv_triangle = [vt[vertex_idx] for vertex_idx in f_h.textures_idx[idxs[i]]]
            uv_point = sum(w * uv for w, uv in zip(weights, uv_triangle))
            if uv_point[0] * uv_point[1] == 0:
                print(i, uv_point)
                print(weights)
                print(triangle)
            uv_coords.append(np.clip(uv_point.numpy(), 0, 1))
        return np.array(uv_coords)

    def get_hair_root_uv(self):
        """
        Main function to load a 3D mesh, compute UV coordinates for a set of points, 
        and save the results as an OBJ file and a scatter plot.
        Args:
            mesh_path (str): Path to the input mesh OBJ file.
            points_path (str): Path to the input points OBJ file.
        """
        v_h, f_h, aux_h = load_obj(self.mesh_path)
        head_mesh = Meshes(verts=v_h.unsqueeze(0), faces=f_h.verts_idx.unsqueeze(0))

        head_verts_packed = head_mesh.verts_packed()
        head_faces_packed = head_mesh.faces_packed()
        vt_h = aux_h.verts_uvs.numpy()

        tris = head_verts_packed[head_faces_packed]
        tris_first_idx = head_mesh.mesh_to_faces_packed_first_idx()
        
        v_r = self.load_points()    # get root positions
        # _, _, v_r = self.load_points()    # get root positions
        # from StrandVAE.util.utils import save_hair2pc
        # save_hair2pc('./tmp.obj', v_r)
        
        # v_r = v_r[:100]    # 100 개만 보자  torch.Size([100,3])

        _, idxs = _C.point_face_dist_forward(v_r, tris_first_idx, tris, tris_first_idx, len(v_r), _DEFAULT_MIN_TRIANGLE_AREA)
        
        uv_r = self.compute_uv_coordinates(v_r, tris, idxs, vt_h, f_h)   # UV coord of the roots on the head mesh, torch.Size(10000, 2)

        # plt.scatter(uv_r[:, 0], uv_r[:, 1], s=1, c='b')
        # plt.scatter(vt_h[:, 0], vt_h[:, 1], s=3, c='r')
        # plt.xlabel('U')
        # plt.ylabel('V')
        # plt.title('UV Scatter Plot')
        # plt.savefig(os.path.join(self.output_path, './roots.png'))
        # plt.close()

        return uv_r