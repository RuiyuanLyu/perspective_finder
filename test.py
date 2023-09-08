import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d 
 
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame() 
vis = o3d.visualization.Visualizer() 
vis.create_window(visible=True)
vis.add_geometry(mesh) 
vis.poll_events() 
vis.update_renderer() 
vis.run()

color = vis.capture_screen_float_buffer(True) 
depth = vis.capture_depth_float_buffer(True) 
vis.destroy_window() 
color = np.asarray(color) 
depth = np.asarray(depth) 
print(np.max(color), np.min(color))
plt.imshow(color) 
plt.show() 
plt.imshow(depth) 
plt.show()  