import numpy as np
from load_data import BFM
from face3d import mesh


def render(facemodel, chi):
    fitted_R = mesh.transform.angle2matrix(chi[0:3])

    fitted_s = chi[3] 
    fitted_t = chi[4:7].copy()
    fitted_t[2] = 1.0
    fitted_ep = np.expand_dims(chi[7:71], 1)
    fitted_sp = np.expand_dims(facemodel.sp, 1)
    tex_coeff = np.expand_dims(facemodel.tex, 1)
    expression1 = facemodel.exBase.dot(fitted_ep)
	
    gamma = np.expand_dims(facemodel.gamma, 0)
	

    vertices = facemodel.meanshape.T + facemodel.idBase.dot(fitted_sp) + expression1
    vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
    print(vertices.shape, vertices.dtype)

    # face_norm = Compute_norm(np.expand_dims(vertices,0),facemodel)
    # face_norm_r = np.matmul(face_norm,np.expand_dims(fitted_R, 0))

    # colors = facemodel.meantex.T + facemodel.texBase.dot(tex_coeff)
    # colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T


facemodel = BFM()
nver = facemodel.idBase.shape[0]/3
ntri = facemodel.tri.shape[0]
n_shape_para = facemodel.idBase.shape[1]
n_exp_para = facemodel.exBase.shape[1]
n_tex_para = facemodel.texBase.shape[1]

kpt_ind = facemodel.keypoints
triangles = facemodel.tri

## Load the audio driven BFM parameters
real_params = "../audio2face/data/train3.npz"
realparams = np.load(open(real_params, 'rb'))
realparams = realparams['face']

idparams = realparams[0,71:151]
texparams = realparams[0,151:231]
gammaparams = realparams[0,231:]


facemodel.sp = idparams
facemodel.tex = texparams
facemodel.gamma = gammaparams


render(facemodel, realparams[:, 0])