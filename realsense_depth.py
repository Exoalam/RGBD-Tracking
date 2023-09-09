import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    align = ""
    depth_scale = ""
    decimation = rs.decimation_filter() 
    spatial = rs.spatial_filter()  
    temporal = rs.temporal_filter()  
    hole_filling = rs.hole_filling_filter()

    def __init__(self):
        global align
        global depth_scale
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        align_to = rs.stream.color
        align = rs.align(align_to)

    def get_frame(self):
        global align

        global depth_scale
        Global_point = (0,0)
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return False, None, None
        
        # aligned_frames = align.process(frames)

        # # Get aligned frames
        # depth_frame = aligned_frames.get_depth_frame()
        # color_frame = aligned_frames.get_color_frame()

        # Get depth frame as numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        

        return True, depth_image, color_image, depth_frame

    def Global_points(self,x,y):
        global align

        global depth_scale
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

    
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()

        # Get depth frame as numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Get intrinsic properties of the depth image
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        global_point = []
        # Iterate over the depth image (assuming depth_image is a 2D numpy array)

        depth = depth_image[y, x] * depth_scale
                # Convert depth pixel to global coordinates
        global_point.append(rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth))

        return global_point
    
    def actual_depth(self,x,y):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        distance = depth_frame.get_distance(x, y)
        return distance
    
    def release(self):
        self.pipeline.stop()

    