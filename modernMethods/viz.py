import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def Gen_Spirals(length, dims=2):
    theta_range = np.random.randint(1,10)
    theta = np.linspace(-theta_range * np.pi, theta_range * np.pi, length)
    z_range = np.random.randint(15,45)
    z = np.random.uniform(1,3)*np.sin(np.linspace(0, z_range, length))
    rx = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    ry = np.abs(z) ** np.random.uniform(1.5,3)*np.abs(np.random.rand())  + 1
    x = rx**1.5 * np.sin(theta)
    y = ry**1.5 * np.cos(theta)

    return np.stack([x,y,z], axis=1) + 5*np.random.uniform(low=-5, high=5, size=[1,dims])

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[:num, 0:2].T)
        line.set_3d_properties(data[:num, 2].T)
    return lines

def visualize(data, length):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    N = np.shape(data)[1]
    data_list = []
    for n in range(N):
        data_list.append(np.squeeze(data[:, n, :].numpy()))
    lines = [ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2])[0] for dat in data_list]

    ax.set_xlim3d([-1, 1])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, 1])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1, 1])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, length, fargs=(data_list, lines),
                                       interval=50, blit=False)

    plt.show()

def old():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Fifty lines of random 3-D lines
    data = [Gen_Spirals(200, 3) for index in range(10)]

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d([-20, 20])
    ax.set_xlabel('X')

    ax.set_ylim3d([-20, 20])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-20, 20])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, 200, fargs=(data, lines),
                                       interval=50, blit=False)

    plt.show()

#old()