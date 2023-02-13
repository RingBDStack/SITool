# Visualize as Encoding Tree and plot in hierarchical_tree.svg
# Use RGB color space for correct node color
#%%

import igraph
import numpy as np

from seg_model import ImageSeg

import PIL

#%%
img=np.array(PIL.Image.open("img.png"))
# img = np.array(ds_test[0][0])
imgH = img.shape[0]
imgW = img.shape[1]
#%%

model = ImageSeg()
img, g = model.process(img, use_lab=False, target_num=1, intercode=True, start_intercode_seg=500)
es, et, c, s = g
es, et, c, s = es.cpu().numpy(), et.cpu().numpy(), c.cpu().numpy(), s.cpu().numpy()
# c[:, 1] += .5
# c[:, 2] += .5
# print(c[:, 0].mean()* 255, c[:, 1].mean()* 255, c[:, 2].mean()* 255)
# c = cv2.cvtColor(((c * 255).reshape(c.shape[0], 1, 3)).clip(0, 255).astype(np.uint8), cv2.COLOR_Lab2RGB)[:, 0, :]
# print(c.mean())
graph = igraph.Graph()
graph.add_vertices(s.shape[0])
graph.add_edges([(es[i], et[i]) for i in range(es.shape[0])])
smx = (s**0.5).max()
layout = graph.layout_reingold_tilford_circular(root = [et[-1]])
igraph.plot(graph, "hierarchical_tree.svg",layout = layout,  vertex_color=[(c[i] ).tolist() for i in range(s.shape[0])],
            vertex_size = [(s[i] ** 0.5) / smx * 40 for i in range(s.shape[0])],edge_width=0.1)
#
# plt.tight_layout()
# plt.show()


