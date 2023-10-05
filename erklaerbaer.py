import numpy as np
import matplotlib.pyplot as plt
from singular_angles import SingularAngles
import seaborn as sns
import matplotlib as mpl

# set fonttype so Avenir can be used with pdf format
mpl.rcParams['pdf.fonttype'] = 42
sns.set(font='Avenir', style="ticks")

singular_angles = SingularAngles()


def plot_vectors(vectors, color, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for vector in vectors:
        ax.arrow(x=0, y=0, dx=vector[0], dy=vector[1], head_width=0.1, head_length=0.1, length_includes_head=True,
                 color=color)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    return ax


# --- define matrices and unit vectors ---
matrix_a = np.array([[1, 1], [0.1, 1]])
matrix_b = np.array([[1.2, 0.8], [0.3, 1.3]])

e0 = np.array([1, 0])
e1 = np.array([0, 1])
ax = plot_vectors([e0, e1], color='#DDAA33')
ax.set_title(r'$\vec{e}$')
plt.savefig('plots/erklaerbaer/e.pdf', bbox_inches='tight')


vectors_a_M_e = [np.dot(matrix_a, e0), np.dot(matrix_a, e1)]
vectors_b_M_e = [np.dot(matrix_b, e0), np.dot(matrix_b, e1)]


# --- decomposition ---
U_a, S_a, V_at = np.linalg.svd(matrix_a)
U_b, S_b, V_bt = np.linalg.svd(matrix_b)

S_a = np.diag(S_a)
S_b = np.diag(S_b)

# --- V transformation ---
vectors_a_V_e = [np.dot(V_at, e0), np.dot(V_at, e1)]
vectors_b_V_e = [np.dot(V_bt, e0), np.dot(V_bt, e1)]
ax = plot_vectors(vectors_a_V_e, color='#004488')
ax = plot_vectors(vectors_b_V_e, color='#BB5566', ax=ax)
ax.set_title(r'$V^{T} \vec{e}$')
plt.savefig('plots/erklaerbaer/Ve.pdf', bbox_inches='tight')

# --- S transformation ---
vectors_a_S_e = [np.dot(S_a, e0), np.dot(S_a, e1)]
vectors_b_S_e = [np.dot(S_b, e0), np.dot(S_b, e1)]
vectors_a_S_V_e = [np.dot(S_a, vectors_a_V_e[0]), np.dot(S_a, vectors_a_V_e[1])]
vectors_b_S_V_e = [np.dot(S_b, vectors_b_V_e[0]), np.dot(S_b, vectors_b_V_e[1])]
ax = plot_vectors(vectors_a_S_e, color='#004488')
ax = plot_vectors(vectors_b_S_e, color='#BB5566', ax=ax)
ax.set_title(r'$\Sigma \vec{e}$')
plt.savefig('plots/erklaerbaer/Se.pdf', bbox_inches='tight')
ax = plot_vectors(vectors_a_S_V_e, color='#004488')
ax = plot_vectors(vectors_b_S_V_e, color='#BB5566', ax=ax)
ax.set_title(r'$\Sigma V^{T} \vec{e}$')
plt.savefig('plots/erklaerbaer/SVe.pdf', bbox_inches='tight')

# --- U transformation ---
vectors_a_U_e = [np.dot(U_a, e0), np.dot(U_a, e1)]
vectors_b_U_e = [np.dot(U_b, e0), np.dot(U_b, e1)]
vectors_a_U_S_V_e = [np.dot(U_a, vectors_a_S_V_e[0]), np.dot(U_a, vectors_a_S_V_e[1])]
vectors_b_U_S_V_e = [np.dot(U_b, vectors_b_S_V_e[0]), np.dot(U_b, vectors_b_S_V_e[1])]
ax = plot_vectors(vectors_a_U_e, color='#004488')
ax = plot_vectors(vectors_b_U_e, color='#BB5566', ax=ax)
ax.set_title(r'$U \vec{e}$')
plt.savefig('plots/erklaerbaer/Ue.pdf', bbox_inches='tight')
ax = plot_vectors(vectors_a_U_S_V_e, color='#004488')
ax = plot_vectors(vectors_b_U_S_V_e, color='#BB5566', ax=ax)
ax.set_title(r'$U \Sigma V^{T} \vec{e}$')
plt.savefig('plots/erklaerbaer/USVe.pdf', bbox_inches='tight')
ax = plot_vectors(vectors_a_M_e, color='#004488')
ax = plot_vectors(vectors_b_M_e, color='#BB5566', ax=ax)
ax.set_title(r'$M \vec{e}$')
plt.savefig('plots/erklaerbaer/Me.pdf', bbox_inches='tight')


# angles_noflip = (singular_angles.angle(U_a, U_b, method='columns')
#                  + singular_angles.angle(V_at, V_bt, method='rows')) / 2
# angles_flip = np.pi - angles_noflip
# angles = np.minimum(angles_noflip, angles_flip)
# weights = (S_a + S_b) / 2
# weights /= np.sum(weights)
# smallness = 1 - angles / (np.pi / 2)
# weighted_smallness = smallness * weights
# similarity_score = np.sum(weighted_smallness)
