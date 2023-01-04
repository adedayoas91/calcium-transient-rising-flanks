import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation


def repopulate(inf,traces,idx):
    """
    To repopulate the inferred matrix into the big matrix of all neurons in the data,
    to facilitate easy extraction of centers for the ROIs
    :param inf: shape [n x n]. This is the inferred matrix your algorithm return.
    :param traces: The whole data, just to extract the shape for big matrix initialization.
    :param idx: This is the list of emitter and receiver ROIs concatenated together
    :return: (sparse) matrix after filling the spaces adequately.
    """
    inferred_ = np.zeros((traces.shape[0],traces.shape[0]))
    p = np.transpose(np.where(inf!=0))
    for i in range(len(p)):
        inferred_[idx[p[i,0]],idx[p[i,1]]] = 1
    return inferred_


def xyz_maker(inferred,topography):
    """
    A func to make coordinates for each ROIs
    :param inferred: shape [n x n] inferred matrix
    :param topography: the position of ROIs given from data
    :return: coordinates for each ROIs and the edges [t]
    """

    t = np.transpose(np.where(inferred>0))
    point_1 = np.zeros_like(t)
    point_2 = np.zeros_like(t)

    for i in range(len(t)):
        point_1[i]=topography[t[i,0],[0,1]]
        point_2[i]=topography[t[i,1],[0,1]]

    x_val,y_val,z_val = [],[],[]
    for i in range(len(point_1)):
        x_val.append([point_1[i,0],point_2[i,0]])
        y_val.append([point_1[i,1],point_2[i,1]])
        z_val.append([topography[t[i,0],2],topography[t[i,1],2]])

    return x_val,y_val,z_val, t


def rotate(angle):
    """
    Basically helper function for the rotation
    This should be in the jupyter notebook
    """
    ax.view_init(azim=angle)


def vertixDegree(inferred_):
    """
    Out from emitters and in to recievers
    :param inferred_: Big matrix from repopulate
    :return:
    """
    nodes_out,nodes_in = {},{}
    for i in range(inferred_.shape[0]):
        nodes_out[i] = np.where(inferred_[i]!=0)[0]
        nodes_in[i] = np.where(inferred_.T[i]!=0)[0]
    return nodes_out,nodes_in

def counter_(out_,from_,emitter,reciever):
    """
    To count the number of in and out edges from each node
    :param out_:
    :param from_:
    :param emitter:
    :param reciever:
    :return:
    """
    n_out,n_in = len(out_),len(from_)
    for el in out_:
        if el in emitter:
            n_out -= 1
    for el in from_:
        if el in reciever:
            n_in -= 1
    return n_out, n_in


if __name__ == '__main__':

    all_positions = np.load('220127_F4_F4_run2_after_dec_cells_positions.npy')
    emitter_idx = np.load('220127_F4_F4_run2_after_dec_emitter_cells.npy')
    reciever_idx = np.load('220127_F4_F4_run2_after_dec_receiver_cells.npy')
    idx = np.r_[emitter_idx, reciever_idx]
    arr = list(idx)

    # % matplotlib notebook

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for i in arr:
        if i in emitter_idx:
            out_, _ = counter_(edges_out[i], edges_in[i], emitter_idx, reciever_idx)
            if out_ > 0:
                ax.scatter(all_positions[i,0], all_positions[i,1], all_positions[i,2], color='red', s=30 * out_ + 1,
                           alpha=.45, marker='*')
            else:
                ax.scatter(all_positions[i,0], all_positions[i,1], all_positions[i,2], color='red', s=30, alpha=.45,
                           marker='*')
        else:
            _, in_ = counter_(edges_out[i], edges_in[i], emitter_idx, reciever_idx)
            if in_ > 0:
                ax.scatter(all_positions[i,0], all_positions[i,1], all_positions[i,2], color='gray', s=30 * in_ + 1,
                           alpha=.75, marker='o')
            else:
                ax.scatter(all_positions[i,0], all_positions[i,1], all_positions[i,2], color='gray', s=30, alpha=.75,
                           marker='o')

    x_val, y_val, z_val, prec = xyz_maker(inferred_, all_positions)   # inferred_ is the big matrix from repopulate()

    for a in range(len(x_val)):
        if prec[a][0] in emitter_idx and prec[a][1] in emitter_idx:
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.4, alpha=.5, c='brown')
        elif prec[a][0] in emitter_idx and prec[a][1] in reciever_idx:
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.5, alpha=.75, c='green')
        elif prec[a][0] in reciever_idx and prec[a][1] in reciever_idx:
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.4, alpha=.4, c='cyan')
        elif prec[a][0] in reciever_idx and prec[a][1] in emitter_idx:
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.5, alpha=.55, c='blue')
    ax.grid(False)
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    rot_animation.save('ER_OUT-IN_new.gif', dpi=80, writer='imagemagick')     # to save the animation
