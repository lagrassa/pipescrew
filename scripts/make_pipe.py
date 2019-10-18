from __future__ import division
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

"returns list of vertices"
def make_polygon(n, side_length):
    int_angle = np.pi/n
    hyp = side_length/(2*np.sin(int_angle))
    theta = int_angle
    pts = []
    for i in range(n):
        new_pt = (hyp*np.cos(theta), hyp*np.sin(theta))
        pts.append(new_pt)
        theta += 2*int_angle
    return pts

def make_cylinder(n, side_length, height, width):
    verts = make_polygon(n, side_length)
    verts = np.vstack(verts)
    shape_indices = [p.createCollisionShape(p.GEOM_BOX, halfExtents = [side_length/2., height/2., width/2.]) for _ in verts]
    #shape_indices = shape_indices[0:2]

    angle = ((n-2)*180)/n
    angle = np.deg2rad(angle)
    phi = np.pi-angle
    link_orientations = [p.getQuaternionFromEuler((0,angle,0)) for i in shape_indices]
    link_positions = []
    link_orientations=[]
    h = ((side_length/2)**2+(width/2)**2)**0.5
    rel_x = side_length/2+h*np.cos(phi)
    rel_y = width/2+h*np.sin(phi)
    curr_angle = angle
    curr_x = 0
    curr_y = 0
    for i in range(len(shape_indices)):
        #link_positions.append((verts[1],0, verts[0]))
        #link_positions.append((rel_y, rel_x,0))
        #link_positions.append((0.9, 0,-1.1))
        #link_positions.append((side_length-width+(i/10.), 0,-(side_length+width)))
        #curr_x += side_length*np.cos(angle)
        #curr_y += side_length*np.sin(angle)
        #curr_angle += angle
        if i == len(shape_indices)-1:
            midpt =np.mean(np.vstack([verts[i], verts[0]]),axis=0) 
            diff = verts[0]-verts[i]
        else:
            midpt =np.mean(np.vstack([verts[i], verts[i+1]]),axis=0) 
            diff = verts[i+1]-verts[i]
        curr_angle = np.arctan2(diff[1],diff[0])-(np.pi/2.0)
        link_orientations.append(p.getQuaternionFromEuler((0,curr_angle,0)))
        x_width_shift = width
        y_width_shift = width
        link_positions.append((midpt[1]+y_width_shift,0,midpt[0]+x_width_shift))

    #shape_indices = shape_indices[0:2]
    #link_orientations = link_orientations[0:2]
    #link_positions = link_positions[0:2]
    parent_indices = [0]*len(shape_indices)
    assert(len(shape_indices)==len(link_orientations)==len(link_positions))
    #p.createMultiBody(linkCollisionShapeIndices=shape_indices, linkPositions=link_positions, linkOrientations=link_orientations)
    sphereUid = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=-1,
                                  basePosition=[0,0,0],
                                  baseOrientation=[1,0,0,0],
                                  linkMasses=[0 for i in shape_indices],
                                  linkCollisionShapeIndices=shape_indices,
                                  linkVisualShapeIndices=[-1 for i in shape_indices],
                                  linkPositions=link_positions,
                                  linkOrientations = link_orientations,
                                  linkInertialFramePositions=[[0,0,0,0] for _ in shape_indices], 
                                  linkInertialFrameOrientations=[[1,0,0,0] for _ in shape_indices],
                                  linkParentIndices=parent_indices,
                                  linkJointTypes=[p.JOINT_FIXED for _ in shape_indices], 
                                  linkJointAxis=[[1,0,0] for _ in shape_indices])

#verts = make_polygon(16, 1)
#p.connect(p.GUI)
#make_cylinder(12,1,3,0.1)


#xs = [pt[0] for pt in verts]
#ys = [pt[1] for pt in verts]
#plt.plot(xs, ys)
#plt.show()
