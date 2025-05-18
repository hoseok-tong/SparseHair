import os
import cv2
import numpy as np
import torch
import matplotlib.cm as cm

from pytorch3d.io import load_obj
from scipy.interpolate import griddata

from StrandVAE.model.component.modules import ClosestPointUV2Mesh



class ExtractShapeTexture(object):
    def __init__(self, mesh_path, hair_npz_path, uv_map_size, interp_method, model, output_path):
        self.mesh_path = mesh_path
        self.hair_npz_path = hair_npz_path
        self.strand_feature = "latent"
        self.uv_map_size = uv_map_size
        self.interp_method = interp_method
        self.model = model
        self.output_path = output_path


    def process(self):
        output_filename = os.path.basename(self.hair_npz_path).split('.')[0]
        output_filedir = os.path.join(self.output_path, f'{self.strand_feature}')
        os.makedirs(output_filedir, exist_ok=True)

        uv_mapper = ClosestPointUV2Mesh(self.mesh_path, self.hair_npz_path, self.output_path)
        uv_r = uv_mapper.get_hair_root_uv()
        vertsS_tan, _, _ = self.load_points()
        
        # # 모근 100개만 보고자 할때.
        # hairstyle = hairstyle[:100]
        # TBNs = TBNs[:100]
        # roots = roots[:100]

        if self.strand_feature == "length":
            strand_length = self.compute_strand_length(vertsS_tan)
            strand_length_uv_map = self.create_feature_uv_map(uv_r, strand_length, method=self.interp_method)
            self.save_feature_uv_map_as_image(strand_length_uv_map, os.path.join(output_filedir, f'{output_filename}.png'))
        
        elif self.strand_feature == "latent":
            strand_latent = self.compute_strand_latent(vertsS_tan, self.model)

            strand_latent_uv_map = self.create_feature_uv_map(uv_r, strand_latent, uv_map_size=self.uv_map_size, method=self.interp_method)
            self.save_feature_uv_map_as_image(torch.sqrt((strand_latent_uv_map**2).mean(dim=2)), os.path.join(output_filedir, f'{output_filename}.png'), uv_map_size=self.uv_map_size)
            self.save_feature_uv_map_as_pt(strand_latent_uv_map, os.path.join(output_filedir, f'{output_filename}.pt'))
        
            return strand_latent_uv_map
        

    def load_points(self):
        _, file_extension = os.path.splitext(self.hair_npz_path)
        if file_extension == ".obj":
            return load_obj(self.hair_npz_path)[0].reshape(-1,100,3)
        elif file_extension == ".npz":
            npzfile = np.load(self.hair_npz_path)
            vertsS_tan = torch.from_numpy(npzfile['vertsS_tan'])
            TBNs = torch.from_numpy(npzfile['TBNs'])
            roots = torch.from_numpy(npzfile['roots'])
            return vertsS_tan, TBNs, roots
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


    @staticmethod
    def compute_strand_length(pointcloud):
        """
        Compute the length of each strand in the pointcloud.
        
        Args:
            pointcloud (torch.Tensor): Tensor of size [num_strands, num_points_per_strand, 3] 
                                    representing the pointcloud.
                                    
        Returns:
            torch.Tensor: Tensor of size [num_strands, 1] representing the length of each strand.
        """
        directions = pointcloud[:, 1:, :] - pointcloud[:, :-1, :]
        distances = torch.norm(directions, dim=2)
        lengths = distances.sum(dim=1).unsqueeze(1)
        return lengths


    @staticmethod
    def compute_strand_latent(pointcloud, model):
        """
        Compute the length of each strand in the pointcloud.
        
        Args:
            pointcloud (torch.Tensor): Tensor of size [num_strands, num_points_per_strand, 3] 
                                    representing the pointcloud.
                                    
        Returns:
            torch.Tensor: Tensor of size [num_strands, 1] representing the length of each strand.
        """
        pointcloud = pointcloud.to(model.layer1.weight.device)
        _, _, _, z = model(pointcloud)
        return z
    

    @staticmethod
    def create_feature_uv_map(uv_r, feature, uv_map_size, method='linear'):
        """
        Create and interpolate a UV map based on root UV coordinates and strand lengths.
        
        Args:
            uv_r (torch.Tensor): UV coordinates of strand roots.
            feature (torch.Tensor): feature of strands.
            uv_map_size (int): Size of the UV map to create.
            
        Returns:
            torch.Tensor: Interpolated UV map.
        """
        grid_x, grid_y = np.mgrid[0:1:uv_map_size*1j, 0:1:uv_map_size*1j]
        if method == 'no_interp':
            # Interpolation 없이 알고 있는 GT strand latent 값만으로 neural texture를 만들 때
            uv_map = np.zeros((uv_map_size,uv_map_size,32))
            tmp = (uv_r // (1/uv_map_size)).astype(int)
            uv_map[tmp[:,0],tmp[:,1],:] = feature.cpu()
            uv_map = torch.tensor(uv_map, dtype=torch.float32)
        else:
            uv_map = griddata(uv_r, feature.cpu().detach().squeeze(), (grid_x, grid_y), method=method)
        return torch.tensor(uv_map, dtype=torch.float32)


    @staticmethod
    def create_root_uv_map(uv_r, uv_map_size):
        """
        Create a UV map based on root UV coordinates.
        
        Args:
            uv_r (torch.Tensor): UV coordinates of strand roots.
            uv_map_size (int): Size of the UV map to create.
            
        Returns:
            torch.Tensor: UV map with uv_r points in red and other pixels in white.
        """
        # Initialize the UV map with white color
        uv_map = torch.ones((uv_map_size, uv_map_size, 3))
        
        # Convert UV coordinates to pixel coordinates
        uv_pixel_coords = np.clip(np.round(uv_r * (uv_map_size - 1)), 0, uv_map_size-1).astype(int)
        
        # Set the uv_r points to red color
        for i in range(uv_r.shape[0]):
            u, v = uv_pixel_coords[i]
            uv_map[u, v] = torch.tensor([1, 0, 0])  # Red color
        
        return uv_map


    @staticmethod
    def save_feature_uv_map_as_image(feature_uv_map, filename, uv_map_size):
        """
        Save a UV map as a colorized image.
        
        Args:
            uv_map (torch.Tensor): UV map to save.
            filename (str): Path to save the image.
        """
        colorized_feature_uv_map = cm.viridis(feature_uv_map.numpy())
        colorized_feature_uv_map_pixel = (colorized_feature_uv_map * (uv_map_size-1)).astype(np.uint8)
        colorized_feature_uv_map_bgr = cv2.cvtColor(colorized_feature_uv_map_pixel, cv2.COLOR_RGB2BGR)
        colorized_feature_uv_map_bgr_rotated = cv2.rotate(colorized_feature_uv_map_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(filename, colorized_feature_uv_map_bgr_rotated)
        # print(f"{filename}: UV map image saved successfully!")

    @staticmethod
    def save_feature_uv_map_as_pt(feature_uv_map, filename):
        """
        Save a UV map as a colorized image.
        
        Args:
            uv_map (torch.Tensor): UV map to save.
            filename (str): Path to save the image.
        """
        torch.save(feature_uv_map.cpu(), filename)
        # print(f"{filename}: UV map pt saved successfully!")



    # @staticmethod
    # def save_root_uv_map_as_image(root_uv_map, filename, uv_map_size):
    #     """
    #     Save a UV map as a colorized image.
        
    #     Args:
    #         uv_map (torch.Tensor): UV map to save.
    #         filename (str): Path to save the image.
    #     """
    #     root_uv_map_255 = (root_uv_map * (uv_map_size-1)).numpy().astype(np.uint8)
    #     root_uv_map_bgr = cv2.cvtColor(root_uv_map_255, cv2.COLOR_RGB2BGR)
    #     root_uv_map_bgr_rotated = cv2.rotate(root_uv_map_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     cv2.imwrite(filename, root_uv_map_bgr_rotated)
    #     # print(f"{filename}: root UV map image saved successfully!")
    



