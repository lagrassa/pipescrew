import numpy as np
import os
import GPy as gpy
from autolab_core import RigidTransform
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn import preprocessing

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize

def in_region_with_modelfailure(pt, good_model_states, bad_model_states, data_folder="test",retrain=False):
    #print("Always in region of model failure for debugging. Turn off for experiments")
    if bad_model_states is None:
        return False  #not enough info
    gp = None
    if retrain or gp is None:
        bad =  bad_model_states
        good = good_model_states
        if len(bad) == 0 and len(good) == 0:
            return False
        #good = np.array([[0.1, 0.1], [0.2, 0.2]])
        #bad = np.array([[-0.1,-0.1], [-0.2, -0.2]])
        max_xs = np.array([.8,.8,.8,1,1,1,1])
        min_xs = -max_xs
        lengthscale = max_xs - min_xs
        lengthscale_bound = np.vstack([[0.001*lengthscale],[1*lengthscale]])
        X = np.vstack([good, bad])
        Y = np.vstack([np.ones((len(good),1)), -np.ones((len(bad),1))])
        #Y = Y.flatten()
        #X = preprocessing.scale(X)
        if gp is None:
            #gp = GPy.models.GPRegression(X,Y)
            k = gpy.kern.Matern52(len(max_xs), ARD=True, lengthscale=lengthscale)
            gp = gpy.models.GPRegression(X, Y, k)
            for i in range(len(max_xs)):
                gp.kern.lengthscale[i:i+1].constrain_bounded(lengthscale_bound[0][i],
                lengthscale_bound[1][i], warning=False)
            gp['.*variance'].constrain_bounded(1e-1,2., warning=False)
            gp['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
            # These GP hyper parameters need to be calibrated for good uncertainty predictions.
            for i in range(20):
                gp.optimize(messages=True)
            #kernel = Matern()
            #gp = GaussianProcessClassifier(kernel = kernel, n_restarts_optimizer=10)
            #kernel = 1.0 * RBF([1.0])
            #gp.fit(X,Y)
            pred_bad = gp.predict(bad)
            pred_good = gp.predict(good)
            #print("Max bad", np.max(pred_bad[0]))
            #print("Min good", np.min(pred_good[0]))
            print("bad", pred_bad[0])
            print("good", pred_good[0])
            #print("good", pred_good[0].round(2))
    
    pred = gp.predict(np.hstack([pt.translation, pt.quaternion]).reshape(1,-1))[0]
    print(pred, "gp pred")
    return not bool(np.round(pred.item()))


def setup_stilde_training_data(data_folder, training=False):
    good_model_states = []
    bad_model_states = []
    model_folder = "data/"+data_folder+"/"
    for fn in os.listdir(model_folder):
        if "good" in fn:
            data = np.load(model_folder+fn, allow_pickle=True)
            if data.all() is not None and len(data) > 0:
                good_model_states.append(data)
        elif "bad" in fn:
            data = np.load(model_folder+fn, allow_pickle=True)
            if data.all() is not None and len(data) > 0:
                if len(data.shape) == 1:
                    data = data.reshape(1,-1)
                bad_model_states.append(data)
    good_model_states = np.concatenate(good_model_states,axis=0)
    bad_model_states = np.concatenate(bad_model_states,axis=0)
    return good_model_states, bad_model_states

def test_gp(data_folder):
    good_model_states, bad_model_states = setup_stilde_training_data(data_folder)
    #pt = [ 0.41101035,  0.01295252,  0.05013599, -0.01704243,  0.99902863,
    #    0.03855948, -0.01282647]
    pt = RigidTransform()
    pt.translation = bad_model_states[2,:3]
    in_region_with_modelfailure(pt, good_model_states, bad_model_states)

if __name__ == "__main__":
    import sys
    data_folder = sys.argv[1]
    test_gp(data_folder)
