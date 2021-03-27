#!/usr/bin/python3
import cv2
import numpy as np
import os

class CubeStep:
    def __init__(self, cube_shape=(8,8,8), video_shape=None):
        self.video_shape = video_shape
        self.depth, self.width, self.height = self.cube_shape = cube_shape
        self.colors = 3  # Purely for clarity
        self.cube_size = 3
        for d in cube_shape:
            self.cube_size *= d
    
    def to_cubes(self, images):
        """ Shatter a series of frames into 8x8x8x3 tensors

            Accepts
            -------
            images : np.array of shape (frame_count, width, height, colors) and dtype uint8
                The sequence of images to analyze.
                The image shape must be a multiple of the cube size, no padding will be done.
            
            Returns
            -------
            np.array of (d, w, h, self.depth, self.width, self.height, self.colors)
                where d, w, h are the time, width, and height of the video measured in cubes
        """
        if self.video_shape is not None and self.video_shape != images.shape:
            raise ValueError("Video shape does not match input shape.")
        else:
            self.video_shape = images.shape
        frame_count, width, height, colors = self.video_shape
        t = images.reshape((
            frame_count // self.depth,
            self.depth,
            width // self.width,
            self.width,
            height // self.height,
            self.height,
            colors
        ))
        # Make it a (x,y,z,8,8,8,3) tensor so it's a prism of cubes, by color
        return np.moveaxis(t, [0,2,4,1,3,5,6], [0,1,2,3,4,5,6])

    def to_matrix(self, cubes):
        """ Flatten a cube of cubes into a single matrix

            Accepts
            -------
            cubes : np.array of (d, w, h, self.depth, self.width, self.height, self.colors)
                where d, w, h are the time, width, and height of the video measured in cubes
            
            Returns
            -------
            np.array of (cube_count, cube_volume)
                where
                cube_count = d*w*h,
                d, w, h = the time, width, and height of the video measured in cubes,
                cube_volume = self.depth * self.width * self.height * self.colors
        """
        # Copy improves locality later
        return cubes.reshape((-1, self.cube_size)).copy()

    def from_cubes(self, cubes):
        """ Convert cubes back into normal frames
        
            Accepts
            -------
            cubes : np.array of (d, w, h, self.depth, self.width, self.height, self.colors)
                where d, w, h are the time, width, and height of the video measured in cubes
            
            Returns
            -------
            np.array of shape (frame_count, width, height, colors) and dtype float32 (not uint8)
                The sequence of images to analyze.
                The image shape must be a multiple of the cube size, no padding will be done.
        """
        d,w,h = cubes.shape[:3]
        uncube = np.moveaxis(cubes, [0,1,2,3,4,5,6], [0,2,4,1,3,5,6])
        return uncube.reshape((d*self.depth, w*self.width, h*self.height, self.colors))
    
    def from_matrix(self, samples):
        """ Convert cubes back into normal frames
        
            Accepts
            -------
            samples : np.array of (cube_count, cube_volume)
                where
                cube_count = d*w*h,
                d, w, h = the time, width, and height of the video measured in cubes,
                cube_volume = self.depth * self.width * self.height * self.colors
                
            
            Returns
            -------
            np.array of (d, w, h, self.depth, self.width, self.height, self.colors)
                where d, w, h are the time, width, and height of the video measured in cubes
        """
        if self.video_shape is None:
            raise ValueError("Video shape unspecified when converting from matrix.")
        frame_count, width, height, colors = self.video_shape
        return samples.reshape((
            frame_count // self.depth,
            width // self.width,
            height // self.height,
            self.depth,
            self.width,
            self.height,
            colors
        ))

def quality_check(original, compressed, num_embed):
    """ Compare pixel to pixel quality of each frame

        Accepts
        -------
        original (str): filename for original video
        compressed (str): regenerated video from decomposition
        num_embed (int): dimensionality of embedding

    """
    vd1 = cv2.VideoCapture()
    vd1.open(original)
    images1 = np.stack([vd1.read()[1] for i in range(24)])    

    vd2 = cv2.VideoCapture()
    vd2.open(compressed)
    images2 = np.stack([vd2.read()[1] for i in range(24)])    

    comp_ratio = os.stat("proj.arr").st_size / os.stat(original).st_size
    mse = ((images2 - images1)**2).mean(axis=None)

    with open("logs.dat", "a") as out:
        out.write(f"{num_embed} {comp_ratio} {mse}\n")



def webm_forward():
    vc = cv2.VideoCapture()
    vc.open("snippet.webm")
    images = np.stack([vc.read()[1] for i in range(24)])
    # Divide into 8x8x8 cubes
    frame_count, width, height, colors = images.shape
    
    assert width % 8 == 0 and height % 8 == 0, "The image dimensions must be divisible by 8"
    vc.release()
    return images.astype(np.float32)

def webm_backward(images):
    """ Save images to a video file as VP8

        Accepts
        -------
        images : np.array of shape (frame_count, width, height, colors) and dtype uint8
            The sequence of images to analyze.
            The width, height, and frame count must all be multiples of 8.
    """
    d,w,h = images.shape[:3]
    vw = cv2.VideoWriter("undemo.mkv", cv2.VideoWriter_fourcc(*'VP80'), 5, (h,w))
    for frameid in range(d):
        vw.write(images[frameid])
    vw.release()




def svd_prep(samples, embed=50):
    """ Create a base to summarize video segments

        Accepts
        -------
        samples : np.array of (cube_count, cube_volume)
            where
            cube_count = d*w*h,
            d, w, h = the time, width, and height of the video measured in cubes,
            cube_volume = self.depth * self.width * self.height * self.colors
        
        Returns
        -------
        np.array of shape (cube_volume, embed)
    """

    # Try a basic SVD approximation
    #base = np.random.uniform(0, 1, (8*8*8*colors, embed))
    #breakpoint()
    #base = np.linalg.qr(np.dot(samples, base))[0]
    #base = np.linalg.qr(np.dot(samples, base))[0]
    
    return np.linalg.svd(samples[::13], full_matrices=False)[2][:embed].T

def svd_forward(base, cubes):
    """ Encode images with eigenvectors, using an existing base

        Accepts
        -------
        base : np.array of shape (cube_volume, embed)
            Linear basis for description of video segments, generated by svd_prep()
        cubes : np.array of (cube_count, cube_volume)
            where
            cube_count = d*w*h,
            d, w, h = the time, width, and height of the video measured in cubes,
            cube_volume = self.depth * self.width * self.height * self.colors

        Returns
        -------
        np.array of (cube_count, embed)
            where
            cube_count = d*w*h,
            d, w, h = the time, width, and height of the video measured in cubes,
            cube_volume = self.depth * self.width * self.height * self.colors
    """
    return np.dot(cubes, base)

def svd_backward(base, proj):
    """ Decode cubes back from embedding
        
        Accepts
        -------
        base : np.array of shape (cube_volume, embed)
            Linear basis for description of video segments, generated by svd_prep()
        proj : np.array of (cube_count, embed)
            where
            cube_count = d*w*h,
            d, w, h = the time, width, and height of the video measured in cubes,
            cube_volume = self.depth * self.width * self.height * self.colors
        
        Returns
        -------
        np.array of (cube_count, cube_volume)
            where
            cube_count = d*w*h,
            d, w, h = the time, width, and height of the video measured in cubes,
            cube_volume = self.depth * self.width * self.height * self.colors
        
    """
    reproj = np.dot(proj, base.T)
    return np.clip(reproj, 0, 255)

def buf_forward(proj):
    """ Save projections to a bytestring for later reconstruction
        
        Accepts
        -------
        proj : np.array
            The projection to store as a byte array
        
        Returns
        -------
        A byte array representing that file. In machine order.
    """
    sign = np.sign(proj)
    proj = sign * np.clip(np.log1p(np.abs(proj)) * 14, 0, 127)
    return (
        np.array(proj.shape).astype(np.uint32).tobytes().ljust(32, b"\x00")
        + proj.astype(np.int8).tobytes()
    )

def buf_backward(buf):
    """ load projections from a bytestring
        
        Accepts
        -------
        proj : np.array
            The projection to store as a byte array
        
        Returns
        -------
        proj : np.array
            the projection previously saves
    """
    shape = tuple(x for x in np.frombuffer(buf[:32], dtype=np.uint32) if x > 0)
    proj = (
        np.frombuffer(buf, dtype=np.int8, offset=32)
        .reshape(shape)
        .astype(np.float32)
    )
    return np.sign(proj) * np.expm1(np.abs(proj)/14)

def run_pipeline(num_embed):
    cube_step = CubeStep()
    cubes = cube_step.to_matrix(cube_step.to_cubes(webm_forward()))
    base = svd_prep(cubes, embed=num_embed)
    proj = svd_forward(base, cubes)
    buf = buf_forward(proj)
    base_buf = buf_forward(base)
    open("proj.arr", "wb").write(buf)
    open("base.arr", "wb").write(base_buf)
    
    # At this point everything is stored in buf and base_buf
    proj = buf_backward(buf)
    #base = buf_backward(base_buf)
    reproj = svd_backward(base, proj)
    newimages = cube_step.from_cubes(cube_step.from_matrix(reproj))
    webm_backward(newimages.astype(np.uint8))
    quality_check("snippet.webm", "undemo.mkv", num_embed)

    
if __name__ == "__main__":
    for n in range(1,50,1):
        run_pipeline(n)
        print(f"Finished {n}")

